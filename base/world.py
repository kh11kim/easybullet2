from __future__ import annotations
from attrs import define, field

import numpy as np
import pybullet as p
from icecream import ic
from pybullet_utils.bullet_client import BulletClient
from .pose import SE3, SO3
from .utils import HideOutput
from typing import *
import abc
import time
from .data import *


class World(BulletClient):
    worlds: Dict[str, World] = dict()
    gui_world_exists = False
    
    @classmethod
    def __new__(cls, *args, **kwargs):
        """This prevents to create two visualizers"""
        gui = True
        if len(args) > 1:
            gui = args[1]
        elif 'gui' in kwargs:
            gui = kwargs['gui'] 
            
        if 'gui' in cls.worlds and gui:
            print("You can't create two visualizers")
            print('Load the existing world')
            return cls.worlds["gui"]
        else:
            world = super().__new__(cls)
            
        if gui:
            print('create gui world')
            cls.worlds['gui'] = world
        else:
            print('create no gui world')
            if 'no_gui' not in cls.worlds:
                cls.worlds['no_gui'] = []
            cls.worlds['no_gui'].append(world)
            
        return world
    
    def __init__(
        self, 
        gui=True, 
        dt=0.001, 
        vis_delay=0.05, 
    ):
        self.gui = gui
        self.dt = dt
        self.vis_delay = vis_delay #visualization delay
        self.t = 0.
        self.bodies:Dict[str, AbstractBody] = dict()
        self.shapes:Dict[str, Shape] = dict()
        self.debug_items: Dict[str, int] = dict()
        
        if gui == True:
            if self.gui_world_exists: return 
            else: self.gui_world_exists = True
        connection_mode = p.GUI if gui else p.DIRECT
        with HideOutput():
            super().__init__(connection_mode=connection_mode)
        if self.gui:
            self.pause_button_uid = p.addUserDebugParameter("turn off loop",1,0,1)
        
        self.set_gravity()

    def get_body(self, name):
        if name in self.bodies:
            return self.bodies[name]
        return None
            
    def set_gravity(self, force_z=-9.81):
        self.setGravity(0, 0, -9.81)
        
    def step(self, no_dynamics=False):
        if no_dynamics:
            self.performCollisionDetection()
        else:
            self.stepSimulation()
            
        if self.vis_delay != 0. and int(self.t / self.dt) % (1/self.vis_delay) == 0:
            time.sleep(self.vis_delay/3) # add delay for visualize
        self.t += self.dt
        
    def show(self):
        """Start infinite loop to visualize for macos"""
        num_quit = p.readUserDebugParameter(self.pause_button_uid)
        
        polling_rate = 100
        while True:
            self.step(no_dynamics=True)
            time.sleep(1/polling_rate)
            quit = p.readUserDebugParameter(self.pause_button_uid)
            if quit >= num_quit+1: break

    def get_shape_id(self, shape:Shape):
        if shape in self.shapes:
            return self.shapes[shape]
        
        viz_id = self.createVisualShape(**shape.get_viz_query())
        col_id = -1 if shape.ghost \
            else self.createCollisionShape(**shape.get_col_query())
        
        self.shapes[shape] = (viz_id, col_id)
        return self.shapes[shape]
    
    def make_geometry_body(
        self, name:str, vis_id:int, col_id:int, 
        mass:float, body_cls:Type[AbstractBody], shape:Shape):
        if name in self.bodies:
            if body_cls is not type(self.bodies[name]):
                raise ValueError(f"Body name already exists with different type!")
            ic("Body name already exists. Return the existing one")
            return self.bodies[name]

        
        uid = self.createMultiBody(
            baseVisualShapeIndex=vis_id,
            baseCollisionShapeIndex=col_id,
            baseMass=mass)
        body = body_cls(
            world=self, uid=uid, 
            name=name, mass=mass, ghost=shape.ghost, shape=shape)
        self.bodies[name] = body
        return body
    
    def remove_body(self, body: str | AbstractBody):
        if isinstance(body, str):
            body = self.bodies[body]
        self.removeBody(body.uid)
        del self.bodies[body.name]
    
    def get_distance_info(
        self, 
        body1:AbstractBody, 
        body2:AbstractBody, 
        link1:int=None, 
        link2:int=None,
        tol:float=0.,
    ):
        """world.step(no_dynamics=True) should be called before using"""
        
        link_index1 = -1
        
        if (link1 is None) and (link2 is None):
            link1, link2 = -1, -1
        elif (link1 is not None) and (link2 is not None):
            pass
        elif link1 is not None:
            link1 = -1
        else:
            raise NotImplementedError("?")
        results = self.getClosestPoints(
            bodyA=body1.uid, bodyB=body2.uid, 
            linkIndexA=link1, linkIndexB=link2, distance=tol)
        return [DistanceInfo(*info) for info in results]


@define
class AbstractBody(abc.ABC):
    world: World
    uid: int
    name: str
    mass: float
    ghost:bool
    
    def set_pose(self, pose: SE3):
        self.world.resetBasePositionAndOrientation(
            self.uid, pose.trans, pose.rot.as_quat())
    
    def get_pose(self):
        pos, orn = self.world.getBasePositionAndOrientation(self.uid)
        return SE3(SO3.from_quat(orn), pos)

    def is_collision_with(self, other_body:AbstractBody):
        distance_info = self.get_distance_info(self, other_body)
        return any(distance_info)
    
    def is_in_collision(self):
        for body in self.world.bodies.values():
            if body is self: continue
            if self.is_collision_with(body): return True
        return False

    def get_dynamics_info(self, link_idx=-1):
        return DynamicsInfo(*self.world.getDynamicsInfo(self.uid, link_idx))
    
    def set_dynamics_info(self, input_dict, link_idx=-1):
        """input_dict should contain key and value of the changeDynamics()
        mass, lateralFriction, restitution, rollingFriction, spinningFriction ..."""
        self.world.changeDynamics(
            bodyUniqueId=self.uid,
            linkIndex=link_idx,
            **input_dict)
    
    def get_aabb(self):
        lower, upper = self.world.getAABB(self.uid, linkIndex=-1)
        lower, upper = np.array(lower), np.array(upper)
        return lower, upper
        

    
        
if __name__ == "__main__":
    world = World()
    world.set_gravity()
    world.show()