import numpy as np
from typing import List
from attrs import define, field

from .world import World, Body
from .pose import SE3, SO3
from .data import *

# #urdf
# if self.body_type == "urdf":
#     link_num = len(self.joint_info) #TODO
#     lowers, uppers = [], []
#     for link in [-1, *range(link_num)]:
#         lower, upper = \
#             self.client.getAABB(self.uid, linkIndex=link)
#         lowers.append(lower)
#         uppers.append(upper)
#     lower = np.min(lowers, axis=0)
#     upper = np.max(uppers, axis=0)

@define
class Bodies:
    """Body container"""
    bodies: List[Body]
    relative_poses: List[Body]
    pose: SE3 = field(factory=lambda : SE3.identity())

    @classmethod
    def from_bodies(cls, base_body:Body, other_bodies:List[Body]):
        bodies = [base_body, *other_bodies]
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