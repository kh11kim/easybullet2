from __future__ import annotations
import numpy as np
from typing import List
from attrs import define, field
from pathlib import Path
from icecream import ic
from .world import World, AbstractBody
from .pose import SE3, SO3
from .data import *
from .utils import generate_temp_urdf
from contextlib import contextmanager
import trimesh


@define
class Geometry(AbstractBody):
    shape: Shape
    ghost: bool


@define
class Bodies:
    """Body container"""
    bodies: List[AbstractBody]
    relative_poses: List[AbstractBody]
    pose: SE3 = field(factory=lambda : SE3.identity())

    @classmethod
    def from_bodies(cls, bodies:List[AbstractBody]):
        """ The first body will be the reference body"""
        rel_poses = [body.get_pose() for body in bodies]
        ref_pose = rel_poses[0]
        rel_poses = [ref_pose.inverse()@pose for pose in rel_poses]
        return cls(bodies, rel_poses, ref_pose)
    
    def get_pose(self):
        return self.pose
    
    def set_pose(self, pose:SE3):
        self.pose = pose
        poses = [self.pose@pose for pose in self.relative_poses]
        for pose, body in zip(poses, self.bodies):
            body.set_pose(pose)


@define
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

    @classmethod
    def create(
        cls,
        name,
        world:World,
        path:Path|str,
        mass:float=0.5,
        fixed:bool = True,
        scale:float = 1.,
    ):
        if name in world.bodies:
            if cls is not type(world.bodies[name]):
                raise ValueError(f"Body name already exists with different type!")
            ic("Body name already exists. Return the existing one")
            return world.bodies[name]
        
        uid = world.loadURDF(
            fileName=str(path),
            useFixedBase=fixed,
            globalScaling=scale)
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
        
        return cls(
            world, uid, name, mass,
            path, fixed, scale, dof, joint_info, movable_joints,
            pos_ctrl_gain_p, pos_ctrl_gain_d, max_torque)
    
    @classmethod
    def from_trimesh(cls, name:str, world:World, mesh:trimesh.Trimesh, fixed:bool):
        import tempfile
        with tempfile.TemporaryDirectory() as tempdir:
            urdf_path = generate_temp_urdf(mesh, tempdir)
            obj = cls.create(
                name, world, 
                urdf_path, fixed=fixed, scale=1.)
        return obj

    
    @property
    def lb(self):
        return np.array([joint.joint_lower_limit  for joint in self.joint_info if joint.movable])
    
    def ub(self):
        return np.array([joint.joint_upper_limit  for joint in self.joint_info if joint.movable])
    @property
    def neutral(self):
        return (self.lb + self.ub)/2
    
    @contextmanager
    def apply_joints(self, q):
        joints_temp = self.get_joint_angles()
        self.set_joint_angles(q)
        yield
        self.set_joint_angles(joints_temp)
    
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
    
    def get_link_pose(self, link_idx):
        assert len(self.joint_info) > link_idx
        pos, xyzw = self.world.getLinkState(self.uid, link_idx)[:2]
        return SE3(SO3(xyzw), pos)
    
    def forward_kinematics(self, q:ArrayLike, link_idx:int):
        with self.apply_joints(q):
            pose = self.get_link_pose(link_idx)
        return pose
    
    def inverse_kinematics(
        self, target_pose:SE3, link_idx:int, 
        validate=True, max_iter=10, pos_tol=1e-3):
        
        solved = False
        for _ in range(max_iter):    
            ik_sol = self.world.calculateInverseKinematics(
                self.uid, link_idx, target_pose.trans, target_pose.rot.as_quat())
            if not validate: 
                solved = True
                break
            
            pose_sol = self.forward_kinematics(ik_sol)
            if np.allclose(pose_sol.trans, target_pose.trans, atol=pos_tol):
                solved = True
                break
        return ik_sol if solved else None
    
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
    
