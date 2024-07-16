from __future__ import annotations
from attrs import define, field

import numpy as np
import pybullet as p
from icecream import ic
from pybullet_utils.bullet_client import BulletClient
from contextlib import contextmanager
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
        z_gravity=-9.81,
        dt=0.005,
        solver_iter=150,
        realtime=True,
        realtime_factor=0.5, 
    ):
        if hasattr(self, "_init"): return #preventing multiple initialization

        self.gui = gui
        self.dt = dt
        self.solver_iter = solver_iter
        self.realtime = realtime
        self.realtime_factor = realtime_factor
        self.gravity = z_gravity

        if gui == True:
            if self.gui_world_exists: return 
            else: self.gui_world_exists = True
        connection_mode = p.GUI if gui else p.DIRECT
        with HideOutput():
            super().__init__(connection_mode=connection_mode)
        if self.gui:
            self.pause_button_uid = p.addUserDebugParameter("turn off loop",1,0,1)
        
        self.set_gravity()
        self.watch_workspace()
        self._init = True

    @contextmanager
    def no_rendering(self):
        self.configureDebugVisualizer(
            p.COV_ENABLE_RENDERING, 0)
        yield
        self.configureDebugVisualizer(
            p.COV_ENABLE_RENDERING, 1)

    def watch_workspace(
        self, 
        target_pos=[0,0,0], 
        distance=1.0, 
        cam_yaw=45, 
        cam_pitch=-35
    ):
        self.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=cam_yaw,
            cameraPitch=cam_pitch,
            cameraTargetPosition=target_pos)

    def get_body(self, name):
        if name in self.bodies:
            return self.bodies[name]
        return None
    
    def set_gravity(self, force_z=-9.81):
        self.setGravity(0, 0, force_z)
    
    def reset(self):
        self.resetSimulation()
        self.setPhysicsEngineParameter(
            fixedTimeStep=self.dt,
            numSolverIterations=self.solver_iter
        )
        self.t = 0.
        self.bodies:Dict[str, AbstractBody] = dict()
        self.shapes:Dict[str, Shape] = dict()
        self.debug_items: Dict[str, int] = dict()
        self.constr: Dict[str, int] = dict()
        self.set_gravity(self.gravity)

    def step(self, no_dynamics=False):
        if no_dynamics:
            self.performCollisionDetection()
        else:
            self.stepSimulation()
        
        # add delay for realtime visualization
        if self.gui and self.realtime:
            time.sleep(self.dt*self.realtime_factor)    
        # if self.vis_delay != 0. and int(self.t / self.dt) % (1/self.vis_delay) == 0:
        #     time.sleep(self.vis_delay/3) 
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

    def remove_body(self, body: str | AbstractBody):
        if isinstance(body, BodyContainer):
            for b in body.bodies:
                self.removeBody(b.uid) # delete all contained bodies
        else: 
            if isinstance(body, str):
                body = self.bodies[body]
            self.removeBody(body.uid)
        del self.bodies[body.name]
    
    def remove_all_debugitems(self):
        self.removeAllUserDebugItems()
    
    def remove_debug_item(self, name):
        self.removeUserDebugItem(self.debug_items[name])

    def add_debug_line(self, p_from, p_to, color=[1,0,0], name=None):
        """ if name is not None, debug uid is tracked in self.debug_items"""
        uid = self.addUserDebugLine(
            lineFromXYZ=p_from, 
            lineToXYZ=p_to, 
            lineColorRGB=color
        )

        if name is not None:
            if name in self.debug_items:
                self.remove_debug_item(name)
            self.debug_items[name] = uid
    
    def draw_workspace(self, workspace_size, workspace_center_bottom=np.zeros(3)):
        half_ws_size = workspace_size/2
        point_from = np.array([
            [half_ws_size, half_ws_size, 0],
            [-half_ws_size, half_ws_size, 0],
            [-half_ws_size, -half_ws_size, 0],
            [half_ws_size, -half_ws_size, 0],
            [half_ws_size, half_ws_size, workspace_size],
            [-half_ws_size, half_ws_size, workspace_size],
            [-half_ws_size, -half_ws_size, workspace_size],
            [half_ws_size, -half_ws_size, workspace_size],
        ])
        point_to = np.array([
            [half_ws_size, half_ws_size, workspace_size],
            [-half_ws_size, half_ws_size, workspace_size],
            [-half_ws_size, -half_ws_size, workspace_size],
            [half_ws_size, -half_ws_size, workspace_size],
            [-half_ws_size, half_ws_size, workspace_size],
            [-half_ws_size, -half_ws_size, workspace_size],
            [half_ws_size, -half_ws_size, workspace_size],
            [half_ws_size, half_ws_size, workspace_size],
        ])
        point_from += workspace_center_bottom
        point_to += workspace_center_bottom
        for p1, p2 in zip(point_from, point_to):
            self.add_debug_line(
                p1, p2, color=[0.5, 0.5, 0.5]
            )
    
    def save_state(self): return self.saveState()
    def restore_state(self, state_uid): return self.restoreState(state_uid)

    def get_distance_info(
        self, 
        body1:AbstractBody, 
        body2:AbstractBody, 
        link1:int=None, 
        link2:int=None,
        tol:float=0.,
    ):
        kwargs = dict()
        kwargs["bodyA"] = body1.uid
        kwargs["bodyB"] = body2.uid
        if link1 is not None: kwargs['linkIndexA'] = link1
        if link2 is not None: kwargs['linkIndexB'] = link2
        kwargs["distance"] = tol
        results = self.getClosestPoints(**kwargs)
        return [DistanceInfo(*info) for info in results]
    
    def get_contact_info(
        self,
        body1:AbstractBody, 
        body2:AbstractBody = None, 
        link1:int=None, 
        link2:int=None,
    ):
        """world.step(no_dynamics=True) should be called before using"""
        kwargs = dict()
        kwargs["bodyA"] = body1.uid
        if body2 is not None: kwargs["bodyB"] = body2.uid
        if link1 is not None: kwargs['linkIndexA'] = link1
        if link2 is not None: kwargs['linkIndexB'] = link2
        results = self.getContactPoints(**kwargs)
        return [ContactInfo(*info) for info in results]
    
    def wait_to_stablize(self, wait_time=0.1, timeout=5.0, tol=0.01, polling_period=100): # polling_dt= 0.1, 
        #timesteps = int(timeout / polling_dt)
        # initial wait
        #polling_steps = int(polling_dt / self.dt)
        for i in range(int(wait_time/self.dt)): 
            self.step()

        t = 0.
        while t < timeout:
            for _ in range(polling_period): # t += polling_dt
                self.step()
            t += self.dt * polling_period

            chk_all_bodies_rest = all([
                np.linalg.norm(body.get_velocity()) < tol
                for body in self.bodies.values()
            ])
            
            if chk_all_bodies_rest:
                break
        return
    def create_constraint(self, name, constr_info:ConstraintInfo):
        if name in self.constr:
            # constraint exists already
            return
        if constr_info.parent_link is None:
            constr_info.parent_link = -1
        if constr_info.child_body is None:
            constr_info.child_body = -1
        if constr_info.child_link is None:
            constr_info.child_link = -1
        
        self.constr[name] = self.createConstraint(
            parentBodyUniqueId=constr_info.parent_body,
            parentLinkIndex=constr_info.parent_link,
            childBodyUniqueId=constr_info.child_body,
            childLinkIndex=constr_info.child_link,
            jointType=constr_info.constr_type,
            jointAxis=constr_info.joint_axis,
            parentFramePosition=constr_info.parent_frame_pos,
            parentFrameOrientation=constr_info.parent_frame_orn,
            childFramePosition=constr_info.child_frame_pos,
            childFrameOrientation=constr_info.child_frame_orn,
        )

    def change_constraint(self, name, **kwargs):
        self.changeConstraint(self.constr[name], **kwargs)

    def remove_constraint(self, name):
        self.removeConstraint(self.constr[name])
        del self.constr[name]
        
    def make_fixed_constraint(self):
        raise NotImplementedError("haha")



@define(repr=False)
class AbstractBody(abc.ABC):
    world: World
    uid: int
    name: str
    mass: float
    
    def set_pose(self, pose: SE3):
        self.world.resetBasePositionAndOrientation(
            self.uid, pose.trans, pose.rot.as_quat())
    
    def get_pose(self):
        pos, orn = self.world.getBasePositionAndOrientation(self.uid)
        return SE3(SO3.from_quat(orn), pos)

    def get_velocity(self):
        linear, angular = self.world.getBaseVelocity(self.uid)
        return linear, angular
    
    def is_collision_with(self, other_body:AbstractBody):
        distance_info = self.world.get_distance_info(self, other_body)
        return any(distance_info)
    
    def is_in_collision(self):
        for other in self.world.bodies.values():
            if other is self: continue
            
            if isinstance(other, BodyContainer):
                for _other in other.bodies:
                    if self is _other: continue
                    if self.is_collision_with(_other):
                        return True
            
            else:
                if self.is_collision_with(other):
                    return True
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
    
    def change_color(self, rgba, link_idx=-1):
        self.world.changeVisualShape(
            bodyUniqueId=self.uid, linkIndex=link_idx, rgbaColor=rgba)

    def __repr__(self):
        return f"{self.__class__.__name__}:{self.name}"
        
@define
class BodyContainer:
    """Body container"""
    name: str
    world: World
    bodies: List[AbstractBody]
    relative_poses: List[AbstractBody]
    #pose: SE3 = field(factory=lambda : SE3.identity())

    def __attrs_post_init__(self):
        for body in self.bodies:
            del self.world.bodies[body.name]
        self.world.bodies[self.name] = self

    @classmethod
    def from_bodies(cls, name:str, bodies:List[AbstractBody]):
        """ The first body will be the reference body"""
        world = bodies[0].world
        rel_poses = [body.get_pose() for body in bodies]
        ref_pose = rel_poses[0]
        rel_poses = [ref_pose.inverse()@pose for pose in rel_poses]
        return cls(name, world, bodies, rel_poses)
    
    def get_pose(self):
        return self.bodies[0].get_pose()
    
    def set_pose(self, pose:SE3):
        #self.pose = pose
        poses = [pose@rel_pose for rel_pose in self.relative_poses]
        for pose, body in zip(poses, self.bodies):
            body.set_pose(pose)
    
    def get_velocity(self):
        return self.bodies[0].get_velocity()
    
    def is_in_collision(self):
        for other in self.world.bodies.values():
            if self is other: continue
            
            for body in self.bodies:
                if body.is_in_collision(): 
                    return True    
        return False
        
if __name__ == "__main__":
    world = World()
    world.set_gravity()
    world.show()
    