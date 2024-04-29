from attrs import define, field
from ..base import *

@define
class PandaHand(BodyContainer):
    max_width = 0.08
    z_offset = 0.105
    grasping_box: Box = field(init=False)

    @property
    def hand(self)->URDF: return self.bodies[0]
    @property
    def swept_vol(self): return self.bodies[1]
    @property
    def T_tcp_base(self): return SE3(trans=[0,0,-self.z_offset])

    def __attrs_post_init__(self):
        self.reset()
        super().__attrs_post_init__()

    def is_grasp_candidate(self, target_obj:AbstractBody):
        self.world.step(no_dynamics=True)
        is_col_gripper = self.hand.is_collision_with(target_obj)
        is_in_swept_vol = self.swept_vol.is_collision_with(target_obj)
        return is_in_swept_vol and not is_col_gripper

    def is_grasped(self, target_obj:AbstractBody):
        base_col = any(self.world.get_distance_info(self.hand, target_obj, -1, -1))
        col = self.hand.is_collision_with(target_obj)
        return not base_col and col
    
    def set_pose(self, pose:SE3):
        base_pose = pose @ self.T_tcp_base
        super().set_pose(base_pose)
    
    def get_pose(self):
        base_pose = super().get_pose()
        return base_pose @ self.T_tcp_base.inverse()
    
    def reset(self, pose=SE3(), width=None):
        if width is None: width = self.max_width
        self.set_pose(pose)
        self.hand.set_joint_angles([width/2,  width/2])

    def grasp(self, duration=0.5):
        q_target = np.zeros(2)
        timesteps = int(duration / self.world.dt)
        self.swept_vol.set_pose(SE3(trans=[0,0,-10]))
        for _ in range(timesteps):
            self.hand.set_ctrl_target_joint_angles(q_target)
            self.world.step()

    

    @classmethod
    def create(cls, name:str, world:World):
        if name in world.bodies:
            ic("Body name already exists.")
            return world.bodies[name]
        hand_urdf_path = Path("../assets/panda/hand.urdf")
        hand = URDF.create(name, world, hand_urdf_path, fixed=True)
        box_half_extents = [0.0085, 0.04, 0.0085]
        swept_vol = Box.create(
            name=f"{name}_swept_vol", 
            world=world, 
            half_extents=box_half_extents, 
            rgba=(0, 1, 0, 0.4))
        swept_vol.set_pose(SE3(trans=[0,0,0.105]))
        return cls.from_bodies(name, [hand, swept_vol])
