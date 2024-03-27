from __future__ import annotations
from attrs import define, field
from numpy.typing import ArrayLike
import abc
from .world import World, AbstractBody
from .body import Geometry, URDF
from .pose import SE3
from .data import *
from .utils import generate_temp_urdf
from pathlib import Path
import trimesh
from icecream import ic

    

@define(repr=False)
class Sphere(Geometry):
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
        shape = SphereShape(rgba=rgba, ghost=ghost, radius=radius)
        vis_id, col_id = world.get_shape_id(shape)
        return cls.make_geometry_body(name, world, vis_id, col_id, mass, shape)
        

@define(repr=False)
class Cylinder(Geometry):
    @classmethod
    def create(
        cls,
        name:str,
        world:World,
        radius:float,
        length:float,
        mass:float=0.5,
        rgba:Tuple[float]=(1, 0, 0, 1),
        offset:SE3 = SE3.identity(), 
        ghost:bool=False
    ):
        shape = CylinderShape(
            rgba=rgba, ghost=ghost, radius=radius, length=length,
            visual_offset_xyz_xyzw=offset.as_xyz_xyzw(),
            col_offset_xyz_xyzw=offset.as_xyz_xyzw()
        )
        vis_id, col_id = world.get_shape_id(shape)
        return cls.make_geometry_body(name, world, vis_id, col_id, mass, shape)
  
@define(repr=False)
class Box(Geometry):
    @classmethod
    def create(
        cls,
        name:str,
        world:World,
        half_extents:ArrayLike,
        mass:float=0.5,
        rgba:Tuple[float]=(1, 0, 0, 1),
        ghost:bool=False
    ):
        shape = BoxShape(rgba=rgba, ghost=ghost, half_extents=half_extents)
        vis_id, col_id = world.get_shape_id(shape)
        return cls.make_geometry_body(name, world, vis_id, col_id, mass, shape)

@define(repr=False)
class Mesh(Geometry):
    @classmethod
    def create(
        cls,
        name:str,
        world:World,
        visual_mesh_path:str|Path|None,
        col_mesh_path:str|Path|None = None,
        centering_type:str|None = "bb", # bb(bounding box center), centroid, None
        scale:float = 1.,
        mass:float=0.5,
        rgba:Tuple[float]=(1, 0, 0, 1)
    ):
        """We assume that visual mesh is in the same coordinate as collision mesh"""
        center = Mesh.get_center(visual_mesh_path, centering_type)
        offset = SE3(trans=-center)
        ghost = True if col_mesh_path is None else False
        shape = MeshShape(
            visual_mesh_path=visual_mesh_path,
            col_mesh_path=col_mesh_path,
            visual_offset_xyz_xyzw=offset.as_xyz_xyzw(),
            col_offset_xyz_xyzw=offset.as_xyz_xyzw(),
            scale=scale,
            rgba=rgba,
            ghost=ghost)
        vis_id, col_id = world.get_shape_id(shape)
        return cls.make_geometry_body(name, world, vis_id, col_id, mass, shape)
    
    @staticmethod
    def get_center(mesh_path, centering_type:str|None = "bb"):
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(True)
        if centering_type == "bb":
            center = mesh.bounding_box.primitive.center
        elif centering_type == "centroid":
            center = mesh.centroid
        elif centering_type is None:
            center = np.zeros(3)
        return center
    
    def as_trimesh(self, is_col:bool=False):
        """ if is_col is True, return the collision mesh, else return the visual mesh"""
        shape: MeshShape = self.shape
        mesh_path = shape.visual_mesh_path
        if is_col:
            assert shape.col_mesh_path is not None, "No collision mesh"
            mesh_path = shape.col_mesh_path
        return trimesh.load(mesh_path)
    
    @classmethod
    def from_trimesh(
        cls, 
        name, 
        world:World, 
        mesh:trimesh.Trimesh,
        col_mesh:trimesh.Trimesh|None=None,
        mass:float=0.5, 
        rgba:ArrayLike=[1,0,0,1]
    ):
        import tempfile
        tempdir = tempfile.TemporaryDirectory()
        if col_mesh is None: 
            col_mesh = mesh

        mesh_path = Path(tempdir.name) / "mesh.obj"
        col_mesh_path = Path(tempdir.name) / "col_mesh.obj"
        mesh.export(mesh_path, "obj")
        col_mesh.export(col_mesh_path, "obj")
        obj = cls.create(
            name, world, 
            str(mesh_path), str(col_mesh_path), 
            mass=mass, rgba=rgba)
        tempdir.cleanup()
        return obj

# @define
# class Frame(BodyContainer):
#     @classmethod
#     def create(
#         cls, name, world, 
#         pose=SE3.identity(), radius=0.004, length=0.04):
#         raise NotImplementedError
#         viz_offsets = [
#             SE3.from_xyz_xyzw([length/2,0,0, 0, 0.7071, 0, 0.7071]),
#             SE3.from_xyz_xyzw([0,length/2,0,-0.7071, 0, 0, 0.7071]),
#             SE3.from_xyz_xyzw([0,0,length/2,0, 0, 0,1]),
#         ]
#         axes_names = "xyz"
#         axes = []
#         rgb = np.eye(3)
#         for i, axis_name in enumerate(axes_names):
#             axes += [Cylinder.create(
#                 name+axis_name, world,
#                 radius, length, ghost=True, 
#                 rgba=tuple([*rgb[i],1.]),
#                 offset=viz_offsets[i]
#             )]
#         frame = cls.from_bodies(axes)
#         frame.set_pose(pose)
#         return frame