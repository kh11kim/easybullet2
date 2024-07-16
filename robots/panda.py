from attrs import define
import numpy as np
import pybullet as p
from ..base import URDF, World, ConstraintInfo, SE3
from ..assets import PANDA_URDF

@define
class Panda(URDF):
    max_width:float = 0.08
    ee_idx:int = 10
    finger_force:float = 50.

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

    def grasp(self, duration=1.):
        q_target = self.get_joint_angles()
        q_target[-2:] = 0.
        timesteps = int(duration / self.world.dt)
        for _ in range(timesteps):
            self.set_ctrl_target_joint_angles(q_target)
            self.world.step()
    
    @classmethod
    def create(cls, name:str, world:World):
        return super().create(name, world, PANDA_URDF, fixed=True)
    
    def move_to_pose_target(self, pose:SE3):
        pass

    def move_to_config_target(self, q:np.ndarray):
        pass