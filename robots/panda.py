from attrs import define
import numpy as np
import pybullet as p
from ..base import URDF, World, ConstraintInfo, SE3, SO3
from ..assets import PANDA_URDF
from scipy.interpolate import CubicSpline

def cubic_spline(qs:np.ndarray, num):
    dof = qs.shape[1]
    tt = np.linspace(0, 1, len(qs), endpoint=True)
    cs = [CubicSpline(tt, qs[:,i], bc_type='clamped') for i in range(dof)]
    ttt = np.linspace(0, 1, num, endpoint=True)
    qs_interp = np.vstack([cs[i](ttt) for i in range(dof)]).T
    return qs_interp

@define
class Panda(URDF):
    max_width:float = 0.08
    ee_idx:int = 10
    finger_force:float = 50.
    qdot_mag:float = 0.5
    v_mag:float = 0.5

    @property
    def neutral(self):
        q = super().neutral
        q[-2:] = self.max_width / 2
        return q
    
    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.pos_ctrl_gain_p = np.array([0.01] * len(self.movable_joints))
        self.pos_ctrl_gain_d = np.array([1.0] * len(self.movable_joints))
        self.max_torque = np.array([250] * len(self.movable_joints))
        self.max_torque[-2:] = self.finger_force
        finger_constr_info = ConstraintInfo(
            self.uid, 8,
            self.uid, 9,
            p.JOINT_GEAR, [1,0,0],
            (0.,0.,0.), (0.,0.,0.,1.),
            (0.,0.,0.), (0.,0.,0.,1.)
        )
        self.world.create_constraint("panda_finger_constr", finger_constr_info)
        self.world.change_constraint("panda_finger_constr", **{"gearRatio":-1, "erp":0.1, "maxForce":self.finger_force})
        self.set_dynamics_info(dict(lateralFriction=1.), 8)
        self.set_dynamics_info(dict(lateralFriction=1.), 9)

    def grasp(self, duration=1.):
        q_target = self.get_joint_angles()
        q_target[-2:] = 0.
        timesteps = int(duration / self.world.dt)
        for _ in range(timesteps):
            self.set_ctrl_target_joint_angles(q_target)
            self.world.step()
    
    def open(self, duration=1.):
        q_target = self.get_joint_angles()
        q_target[-2:] = self.max_width/2
        timesteps = int(duration / self.world.dt)
        for _ in range(timesteps):
            self.set_ctrl_target_joint_angles(q_target)
            self.world.step()
    
    @classmethod
    def create(cls, name:str, world:World):
        return super().create(name, world, PANDA_URDF, fixed=True)
    
    def move_to_pose_target(
        self, 
        pose_goal:SE3, 
        v_mag=None, 
        is_grasped=False, 
        converge_time=3.,
        gain=1.,
    ):
        ''' task-space control (position)'''
        if v_mag is None: v_mag = self.v_mag
        pose = self.get_link_pose(self.ee_idx)
        pos_diff = pose_goal.trans - pose.trans
        duration = (np.linalg.norm(pos_diff) / v_mag) + converge_time # to converge
        timesteps = int(duration / self.world.dt)
        
        traj = cubic_spline(
            np.stack([pose.trans, pose_goal.trans]), timesteps)
        
        for pos_d in traj:
            #pose = self.get_link_pose(self.ee_idx)
            #if np.linalg.norm(pose.trans - pose_goal.trans, ord=np.inf) < 0.01: return
            q = self.get_joint_angles()    
            pose.trans = pos_d
            q_delta, pos_err = self.calculate_approach_control(pose.as_matrix())
            
            #q_delta_norm = np.linalg.norm(q_delta)
            # if q_delta_norm > max_qdelta:
            #     q_delta = q_delta / q_delta_norm * max_qdelta
            q = q + q_delta * gain

            if is_grasped: q[-2:] = 0.
            else: q[-2:] = 0.04
            self.set_ctrl_target_joint_angles(q)
            self.world.step()
        
        for _ in range(int(converge_time/self.world.dt)):
            pose = self.get_link_pose(self.ee_idx)
            if np.linalg.norm(pose.trans - pose_goal.trans, ord=np.inf) < 0.001:
                break
            self.world.step()

    def follow_trajectory(
        self,
        traj: np.ndarray,
        qdot_mag=None, 
        is_grasped=False, 
        converge_time=3.
    ):
        if qdot_mag is None: qdot_mag = self.qdot_mag
        ''' joint-space control (position)'''
        #q_diff = traj[-1] - self.get_joint_angles()
        duration = np.sum([
            np.linalg.norm(q1-q2) / qdot_mag for q1, q2 in zip(traj[:-1], traj[1:])
        ])
        timesteps = int(duration // self.world.dt)
        traj = cubic_spline(traj, timesteps)
        for qd in traj:
            if len(qd) == 7: qd = np.r_[qd, 0, 0.]
            if is_grasped: qd[-2:] = 0.
            else: qd[-2:] = 0.04
            self.set_ctrl_target_joint_angles(qd)
            self.world.step()
        
        for _ in range(int(converge_time/self.world.dt)):
            q = self.get_joint_angles()
            if np.linalg.norm(q - traj[-1], ord=np.inf) < 0.001:
                break
            self.world.step()

    def move_to_config_target(
        self, 
        q_goal:np.ndarray,
        qdot_mag=None, 
        is_grasped=False, 
        converge_time=3.
    ):
        ''' joint-space control (position)'''
        if qdot_mag is None: qdot_mag = self.qdot_mag
        q = self.get_joint_angles()
        q_diff = q_goal - q
        duration = (np.linalg.norm(q_diff) / qdot_mag)# to converge
        timesteps = int(duration // self.world.dt)
        
        traj = cubic_spline(
            np.stack([q, q_goal]), timesteps)
        for qd in traj:
            if len(qd) == 7: qd = np.r_[qd, 0, 0.]
            if is_grasped: qd[-2:] = 0.
            else: qd[-2:] = 0.04
            self.set_ctrl_target_joint_angles(qd)
            self.world.step()
        
        for _ in range(int(converge_time/self.world.dt)):
            q = self.get_joint_angles()
            if np.linalg.norm(q - q_goal, ord=np.inf) < 0.001:
                break
            self.world.step()

    def calculate_approach_control(self, target_pose):
        def skew(v):
            v1, v2, v3 = v
            return np.array([[0, -v3, v2], [v3, 0., -v1], [-v2, v1, 0.]])
        def to_ec(rot_mat):
            return SO3.from_matrix(rot_mat).as_rotvec()
        def get_rotvec_angvel_map_np(v, eps=1e-8):            
            vmag = (np.linalg.norm(v) + eps)
            vskew = skew(v)
            alpha = 0.5 * vmag * 1 / np.tan(vmag/2)
            return np.eye(3) - 0.5*skew(v) \
                + (1 - alpha) * vskew @ vskew / vmag**2
        
        ee_pose = self.get_link_pose(10).as_matrix()
        q_curr = self.get_joint_angles()
        jac = self.get_jacobian(q_curr, 10)
        goal_rot = ee_pose[:3, :3]
        
        pos_err = ee_pose[:3, :3] @ (ee_pose[:3, 3] - target_pose[:3, 3])
        rot_err = ee_pose[:3, :3] @ to_ec(goal_rot.T @ ee_pose[:3, :3])
        error = np.hstack([pos_err, rot_err])
        E = get_rotvec_angvel_map_np(rot_err)
        jac_rotvec = np.vstack([ee_pose[:3, :3]@jac[:3], E@jac[3:]])
        return - np.linalg.pinv(jac_rotvec) @ error, pos_err