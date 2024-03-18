from attrs import define, field
from numpy.typing import ArrayLike
import abc
from .world import World, Body, Shape
from .pose import SE3
from .data import *

    
@define
class Sphere(Body):
    # radius: float
    # mass: float = 0.5
    # body_type: str = "sphere"
    # rgba: ArrayLike = (1, 0, 0, 1)
    # ghost: bool = False
    
    @classmethod
    def create(
        cls,
        name:str,
        world:World,
        radius:float,
        mass:float=0.5,
        rgba:Tuple[float]=(1, 0, 0, 1),
        ghost:bool=False
    ):
        offset = tuple(SE3.identity().as_xyz_xyzw())
        shape = SphereShape(rgba, ghost, offset, offset, radius=radius)
        vis_id, col_id = world.get_shape_id(shape)
        uid = world.createMultiBody(
            baseVisualShapeIndex=vis_id,
            baseCollisionShapeIndex=col_id,
            baseMass=mass)
        cls(world, uid, "sphere", name, mass, ghost)
        
    def __attrs_post_init__(self):
        
        super().__attrs_post_init__()
    
    
    # def __init__(
    #     self, 
    #     name:str, 
    #     world:World, 
    #     radius:float, 
    #     mass:float=0.5,
    #     rgba:ArrayLike=(1,0,0,1), 
    #     ghost:bool=False
    # ):
    #     offset = SE3.identity().as_xyz_xyzw()
    #     shape = SphereShape(rgba, ghost, offset, offset, radius=radius)
    #     vis_id, col_id = world.get_shape_id(shape)
    #     uid = self.world.createMultiBody(
    #         baseVisualShapeIndex=vis_id,
    #         baseCollisionShapeIndex=col_id,
    #         baseMass=mass)
    #     super().__init__(name, uid, world, "sphere")
        
        
    @classmethod
    def get_shape(cls, radius, rgba=(1,0,0,1), ghost=False, offset: SE3 = None):
        offset = SE3.identity() if offset is None else offset
        offset = tuple(offset.as_xyz_xyzw())
        return SphereShape(rgba, ghost, offset, offset, radius=radius)