from typing import Tuple
from attrs import define, field
import pybullet as p
import abc
from .pose import SE3
from numpy.typing import ArrayLike
import numpy as np

"""Information Structure
"""
@define
class JointState:
    pos: float
    vel: float
    wrench: Tuple[float] #reaction force
    torque: float # applied motor torque

@define
class JointInfo:
    joint_index: int
    joint_name: str
    joint_type: int
    q_index: int
    u_index: int
    flags: int
    joint_damping: float
    joint_friction: float
    joint_lower_limit: float
    joint_upper_limit: float
    joint_max_force: float
    joint_max_velocity: float
    link_name: str
    joint_axis: Tuple[float]
    parent_frame_pos: Tuple[float]
    parent_frame_orn: Tuple[float]
    parent_index: int

    @property
    def movable(self): return self.joint_type != p.JOINT_FIXED

@define
class DistanceInfo:
    contact_flag: bool
    bodyA: int
    bodyB: int
    linkA: int
    linkB: int
    position_on_A: Tuple[float]
    position_on_B: Tuple[float]
    contact_normal_on_B: Tuple[float]
    contact_distnace: float
    normal_force: float
    lateral_frictionA: float
    lateral_friction_dirA: Tuple[float]
    lateral_frictionB: float
    lateral_friction_dirB: Tuple[float]

@define
class ContactInfo:
    contactFlag: int # nothing
    bodyA: int
    bodyB: int
    link_indexA: int
    link_indexB: int
    position_on_A: Tuple[float]
    position_on_B: Tuple[float]
    contact_normal_on_B: Tuple[float]
    contact_distance: float
    normal_force: float
    lateral_friction1: float
    lateral_friction_dir1: Tuple[float]
    lateral_friction2: float
    lateral_friction_dir2: Tuple[float]

@define
class DynamicsInfo:
    mass: float
    lateral_friction: float
    local_inertial_diagonal: Tuple[float]
    local_inertial_pos: Tuple[float]
    local_inertial_orn: Tuple[float]
    restitution: float
    rolling_friction: float
    spinning_friction: float
    contact_damping: float
    contact_stiffness: float
    body_type: int
    collision_margin: float
    
@define(frozen=True)
class Shape(abc.ABC):
    rgba: Tuple = field(converter=tuple)
    ghost: bool
    visual_offset_xyz_xyzw: Tuple = field(default=(0,0,0,0,0,0,1.), converter=tuple)
    col_offset_xyz_xyzw: Tuple = field(default=(0,0,0,0,0,0,1.), converter=tuple)

    def get_viz_query(self) -> dict:
        return dict(
            visualFramePosition=self.visual_offset.trans,
            visualFrameOrientation=self.visual_offset.rot.as_quat(),
            rgbaColor=self.rgba
        )
    
    def get_col_query(self):
        return dict(
            collisionFramePosition=self.col_offset.trans,
            collisionFrameOrientation=self.col_offset.rot.as_quat(),
        )
    
    @property
    def visual_offset(self): return SE3.from_xyz_xyzw(self.visual_offset_xyz_xyzw)
    @property
    def col_offset(self): return SE3.from_xyz_xyzw(self.col_offset_xyz_xyzw)

@define(frozen=True, kw_only=True)
class SphereShape(Shape):
    radius: float

    def get_viz_query(self):
        query = super().get_viz_query()
        query.update(
            shapeType=p.GEOM_SPHERE,
            radius=self.radius)
        return query
    
    def get_col_query(self):
        query = super().get_col_query()
        query.update(
            shapeType=p.GEOM_SPHERE,
            radius=self.radius)
        return query

@define(frozen=True, kw_only=True)
class CylinderShape(Shape):
    radius: float
    length: float

    def get_viz_query(self):
        query = super().get_viz_query()
        query.update(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            length=self.length,
        )
        return query
    
    def get_col_query(self):
        query = super().get_col_query()
        query.update(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            height=self.length,
        )
        return query
    
@define(frozen=True, kw_only=True)
class BoxShape(Shape):
    half_extents: Tuple = field(converter=tuple)

    def get_viz_query(self):
        query = super().get_viz_query()
        query.update(
            shapeType=p.GEOM_BOX,
            halfExtents=self.half_extents,
        )
        return query
    
    def get_col_query(self):
        query = super().get_col_query()
        query.update(
            shapeType=p.GEOM_BOX,
            halfExtents=self.half_extents,
        )
        return query

@define(frozen=True, kw_only=True)
class MeshShape(Shape):
    visual_mesh_path: str
    col_mesh_path: str
    scale: float

    def get_viz_query(self):
        query = super().get_viz_query()
        query.update(
            shapeType=p.GEOM_MESH,
            fileName=self.visual_mesh_path,
            meshScale=np.ones(3)*self.scale,
        )
        return query
    
    def get_col_query(self):
        query = super().get_col_query()
        query.update(
            shapeType=p.GEOM_MESH,
            fileName=self.col_mesh_path,
            meshScale=np.ones(3)*self.scale,
        )
        return query