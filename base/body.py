from __future__ import annotations
import numpy as np
from typing import List, Type
from attrs import define, field
from pathlib import Path
from icecream import ic
from .world import World, AbstractBody
from .pose import SE3, SO3
from .data import *
from .utils import generate_temp_urdf, generate_frame_urdf
from contextlib import contextmanager
import trimesh


@define
class Geometry(AbstractBody):
    shape: Shape

    @classmethod
    def make_geometry_body(
        cls, name:str, world:World, vis_id:int, col_id:int, 
        mass:float, shape:Shape):
        if name in world.bodies:
            ic("Body name already exists.")
            return world.bodies[name]
            #world.remove_body(name)

        uid = world.createMultiBody(
            baseVisualShapeIndex=vis_id,
            baseCollisionShapeIndex=col_id,
            baseMass=mass)
        body = cls(
            world=world, 
            uid=uid, 
            name=name, 
            mass=mass, 
            ghost=shape.ghost, 
            shape=shape)
        world.bodies[name] = body
        return body




@define(repr=False)
class URDF(AbstractBody):
    path: str
    fixed: bool
    scale: float
    dof: int
    joint_info: List[JointInfo]
    movable_joints: np.ndarray
    pos_ctrl_gain_p: List[float]
    pos_ctrl_gain_d: List[float]
    max_torque: List[float]
    T_com: SE3 = field(init=False)

    def __attrs_post_init__(self):
        self.T_com = super().get_pose()
        self.set_joint_angles(self.neutral)

    @classmethod
    def create(
        cls,
        name,
        world:World,
        path:Path|str,
        fixed:bool = True,
        scale:float = 1.,
        ghost: bool = False  
    ):
        if name in world.bodies:
            ic("Body name already exists.")
            return world.bodies[name]
            #world.remove_body(name)

        #flags = p.URDF_USE_INERTIA_FROM_FILE if mass is not None else 0
        uid = world.loadURDF(
            fileName=str(path),
            useFixedBase=fixed,
            globalScaling=scale,
            #flags=flags, # TODO
        )
        dof = world.getNumJoints(uid)

        joint_info = []
        movable_joints = []
        for i in range(dof):
            info = JointInfo(*world.getJointInfo(uid, i))
            joint_info.append(info)
            if info.movable:
                movable_joints.append(i)
        movable_joints = np.array(movable_joints)
        pos_ctrl_gain_p = [0.01] * len(movable_joints)
        pos_ctrl_gain_d = [1.0] * len(movable_joints)
        max_torque = [250] * len(movable_joints)
        mass = None
        body = cls(
            world=world, 
            uid=uid, 
            name=name, 
            mass=mass,
            ghost=ghost,
            path=path, 
            fixed=fixed, 
            scale=scale, 
            dof=dof, 
            joint_info=joint_info, 
            movable_joints=movable_joints,
            pos_ctrl_gain_p=pos_ctrl_gain_p, 
            pos_ctrl_gain_d=pos_ctrl_gain_d, 
            max_torque=max_torque
        )
        if ghost:
            # TODO -> setCollisionFilterGroupMask
            # TODO: Do something here if collision behavior should be changed
            world.ghosts[name] = body
        else:
            world.bodies[name] = body
        return body
    
    @classmethod
    def from_trimesh(cls, name:str, world:World, mesh:trimesh.Trimesh, fixed:bool, rgba=[1,1,1,1]):
        import tempfile
        with tempfile.TemporaryDirectory() as tempdir:
            urdf_path = generate_temp_urdf(mesh, tempdir, rgba)
            obj = cls.create(
                name, world, 
                urdf_path, fixed=fixed, scale=1.)
        return obj
    
    @property
    def lb(self):
        return np.array([joint.joint_lower_limit  for joint in self.joint_info if joint.movable])
    
    @property
    def ub(self):
        return np.array([joint.joint_upper_limit  for joint in self.joint_info if joint.movable])
    
    @property
    def neutral(self):
        return (self.lb + self.ub)/2
    
    @contextmanager
    def set_joint_angles_context(self, q):
        joints_temp = self.get_joint_angles()
        self.set_joint_angles(q)
        yield
        self.set_joint_angles(joints_temp)
    
    def set_pose(self, pose):
        """ This is because, simply setting pose is setting a pose of base inertia frame(CoM).
        This differs from initial load state of the URDF."""
        super().set_pose(pose @ self.T_com)
    
    def get_pose(self):
        return super().get_pose() @ self.T_com.inverse()
    
    def get_joint_states(self):
        return [JointState(*s) 
                for s in self.world.getJointStates(self.uid, self.movable_joints)]
    
    def get_joint_angles(self):
        return np.array([s.pos for s in self.get_joint_states()])
    
    def set_joint_angle(self, i, angle):
        self.world.resetJointState(
            self.uid, jointIndex=i, targetValue=angle)
        
    def set_joint_angles(self, angles):
        assert len(angles) == len(self.movable_joints), f"num_angle is not matched: {len(angles)} vs {len(self.movable_joints)}"
        for i, angle in zip(self.movable_joints, angles):
            self.set_joint_angle(i, angle)
    
    def get_joint_frame_pose(self, joint_idx):
        assert len(self.joint_info) > joint_idx
        parent_link_idx = joint_idx - 1
        parent_link_pose = self.get_link_pose(parent_link_idx)
        pos = self.joint_info[joint_idx].parent_frame_pos
        quat = self.joint_info[joint_idx].parent_frame_orn
        rel_pose = SE3(SO3.from_quat(quat), pos)
        return parent_link_pose @ rel_pose

    def get_link_pose(self, link_idx):
        assert len(self.joint_info) > link_idx
        if link_idx == -1:
            return super().get_pose()
        pos, xyzw = self.world.getLinkState(self.uid, link_idx)[:2]
        return SE3(SO3.from_quat(xyzw), pos)
    
    def forward_kinematics(self, q:ArrayLike, link_idx:int):
        with self.set_joint_angles_context(q):
            pose = self.get_link_pose(link_idx)
        return pose
    
    def inverse_kinematics(
        self, target_pose:SE3, link_idx:int, 
        validate=True, max_iter=10, pos_tol=1e-3
    ):
        solved = False
        q = self.get_joint_angles()
        with self.set_joint_angles_context(q):
            for _ in range(max_iter):    
                ik_sol = self.world.calculateInverseKinematics(
                    self.uid, link_idx, target_pose.trans, target_pose.rot.as_quat())
                self.set_joint_angles(ik_sol) # update initial joint angles to ik solution
                if not validate: 
                    solved = True
                    break
                
                pose_sol = self.forward_kinematics(ik_sol, link_idx)
                if np.linalg.norm(pose_sol.trans-target_pose.trans) < pos_tol:
                    solved = True
                    break
        return np.array(ik_sol) if solved else None
    
    def set_ctrl_target_joint_angles(self, q):
        assert len(q) == len(self.movable_joints)
        self.world.setJointMotorControlArray(
            self.uid, 
            jointIndices=self.movable_joints, 
            controlMode=p.POSITION_CONTROL, 
            targetPositions=q,
            forces=self.max_torque,
            positionGains=self.pos_ctrl_gain_p,
            velocityGains=self.pos_ctrl_gain_d,
        )
    
    def get_jacobian(self, q, link_idx, local_position=[0,0,0]):
        jac_trans, jac_rot = self.world.calculateJacobian(
            bodyUniqueId=self.uid,
            linkIndex=link_idx,
            localPosition=local_position,
            objPositions=q.tolist(),
            objVelocities=np.zeros_like(q).tolist(),
            objAccelerations=np.zeros_like(q).tolist()
        )
        return np.vstack([jac_trans, jac_rot])
    
class Frame(URDF):
    @classmethod
    def create(cls, name:str, world:World, length=0.05, radius=0.005):
        import tempfile
        with tempfile.TemporaryDirectory() as tempdir:
            urdf_path = generate_frame_urdf(tempdir, length, radius)
            # Note: "frame.urdf" has no collision shape
            frame = super().create(name, world, urdf_path, fixed=True, ghost=True)
        return frame