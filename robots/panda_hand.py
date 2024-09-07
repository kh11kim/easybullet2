from attrs import define, field
from ..base import *
from ..assets import PANDA_HAND_URDF

@define
class PandaHand(BodyContainer):
    max_width = 0.08
    z_offset = 0.105

    @property
    def hand_body(self)->URDF: return self.bodies[0]
    @property 
    def is_swept_vol(self): return len(self.bodies) == 2
    @property
    def swept_vol(self): 
        assert self.is_swept_vol
        return self.bodies[1]
    @property
    def tcp_sphere(self): return self.bodies[2]
    @property
    def T_tcp_base(self): return SE3(trans=[0,0,-self.z_offset])
    @property
    def T_base(self): return super().get_pose()

    def __attrs_post_init__(self):
        self.reset()
        super().__attrs_post_init__()

    def is_grasp_candidate(self, target_obj:AbstractBody):
        assert self.is_swept_vol
        self.world.step(no_dynamics=True)
        is_col_gripper = self.hand_body.is_collision_with(target_obj, ignore_fixed_base=False)
        is_in_swept_vol = self.swept_vol.is_collision_with(target_obj)
        return is_in_swept_vol and not is_col_gripper

    def is_grasped(self, target_obj:AbstractBody, min_depth=0.01):
        base_col = any(self.world.get_distance_info(self.hand_body, target_obj, -1, -1))
        col = self.hand_body.is_collision_with(target_obj)
        sth_in_gripper = self.get_width() > min_depth
        return not base_col and col and sth_in_gripper
    
    def set_pose(self, pose:SE3):
        base_pose = pose @ self.T_tcp_base
        super().set_pose(base_pose)
    
    def get_pose(self):
        base_pose = super().get_pose()
        return base_pose @ self.T_tcp_base.inverse()
    
    def get_width(self):
        return np.sum(self.hand_body.get_joint_angles())
    
    def reset(self, pose=SE3(), width=None):
        if width is None: 
            width = self.max_width
        self.set_pose(pose)
        self.hand_body.set_joint_angles([width/2,  width/2])

    def grasp(self, kinematics=False, duration=2., force=50):
        q_target = np.zeros(2)
        if kinematics:
            self.hand_body.set_joint_angles(q_target)
            return
        timesteps = int(duration / self.world.dt)
        if self.is_swept_vol:
            self.swept_vol.set_pose(SE3(trans=[0,0,-10]))
        
        for _ in range(timesteps):
            self.hand_body.max_torque = [force] * 2
            self.hand_body.set_ctrl_target_joint_angles(q_target)
            self.world.step()

    def open(self, kinematics=False, duration=1., force=50):
        q_target = np.ones(2) * self.max_width / 2
        if kinematics:
            self.hand_body.set_joint_angles(q_target)
            return
        timesteps = int(duration / self.world.dt)
        if self.is_swept_vol:
            self.swept_vol.set_pose(SE3(trans=[0,0,-10]))
        
        for _ in range(timesteps):
            self.hand_body.max_torque = [force] * 2
            self.hand_body.set_ctrl_target_joint_angles(q_target)
            self.world.step()

    @classmethod
    def create(cls, name:str, world:World, fixed:bool, grasping_vol:bool=True):
        if name in world.bodies:
            ic("Body name already exists.")
            return world.bodies[name]
        
        bodies = []
        hand = URDF.create(
            name=name, 
            world=world, 
            path=PANDA_HAND_URDF, 
            fixed=fixed, 
            ghost=False)
        bodies.append(hand)
        if grasping_vol:
            box_half_extents = [0.0085, 0.04, 0.0085]
            swept_vol = Box.create(
                name=f"{name}_swept_vol", 
                world=world, 
                half_extents=box_half_extents, 
                rgba=(0, 1, 0, 0.4))
            swept_vol.set_pose(SE3(trans=[0,0,0.105]))
            bodies.append(swept_vol)

        finger_constr_info = ConstraintInfo(
            hand.uid, 0,
            hand.uid, 1,
            p.JOINT_GEAR, [1,0,0],
            (0.,0.,0.), (0.,0.,0.,1.),
            (0.,0.,0.), (0.,0.,0.,1.)
        )
        world.create_constraint("finger_constr", finger_constr_info)
        world.change_constraint("finger_constr", **{"gearRatio":-1, "erp":0.1, "maxForce":50})
        
        return cls.from_bodies(name, bodies)


#@define
class PandaHandUtil:
    def __init__(self, world:World):
        self.world = world
        self.hand = None
        self.invisible_pose = SE3(trans=[0,0,2])
    
    def update_tcp_constraint(self, pose):
        T_body = pose @ self.hand.T_tcp_base @ self.hand.hand_body.T_com
        self.world.change_constraint(
            "hand_pose_constr",
            jointChildPivot=T_body.trans,
            jointChildFrameOrientation=T_body.rot.as_quat(),
            maxForce=1000,
        )

    def invisible(self):
        assert self.hand is not None
        if self.hand is None:
            self.hand.reset(self.invisible_pose)
    
    def is_collision(self, pose):
        assert self.hand is not None
        hand: PandaHand = self.hand
        hand.reset(pose)
        return hand.is_in_collision()


    def reset(self, pose=SE3(), width=None, finger_constraint=True):
        if self.hand is None:
            self.hand = PandaHand.create(
                "hand", 
                self.world, 
                fixed=False,
                grasping_vol=False
            )
        if width is None:
            width = self.hand.max_width
        self.hand.reset(pose, width)

        # if finger_constraint:
        pose_constr_info = ConstraintInfo(
            self.hand.hand_body.uid, -1,
            -1, -1,
            p.JOINT_FIXED, [0,0,0],
            (0.,0.,0.), (0.,0.,0.,1.),
            self.hand.T_base.trans, self.hand.T_base.rot.as_quat()
        )
        self.world.create_constraint("hand_pose_constr", pose_constr_info)
        self.update_tcp_constraint(pose)

    def move_xyz(self, target_pose:SE3, xyz_step=0.001, vel=0.1, abort_on_contact=True):
        curr_pose = self.hand.get_pose()
        self.update_tcp_constraint(curr_pose)
        
        diff = target_pose.trans - curr_pose.trans
        n_steps = int(np.linalg.norm(diff) / xyz_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            curr_pose.trans += dist_step
            self.update_tcp_constraint(curr_pose)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.hand.is_in_collision():
                return
        for _ in range(int(dur_step / self.world.dt)):
            self.world.step()
    
    def remove(self, remove_constraint=True):
        if remove_constraint:
            self.world.remove_constraint("hand_pose_constr")
            self.world.remove_constraint("finger_constr")
        if self.hand is not None:
            self.world.remove_body(self.hand)
            self.hand = None
    
    def execute_grasp(self, grasp_pose:SE3, allow_contact=False, return_grasp_obj=True, remove_gripper=True):
        pre_grasp_pose = grasp_pose @ SE3(trans=[0,0,-0.1])
        post_grasp_pose = SE3(trans=[0,0,0.2]) @ grasp_pose

        is_success = False
        grasped_obj = None
        try:
            self.reset(pre_grasp_pose)
            if self.hand.is_in_collision():
                raise ValueError
            self.move_xyz(grasp_pose)
            if self.hand.is_in_collision() and not allow_contact:
                raise ValueError
            
            self.hand.grasp(duration=1.)
            self.move_xyz(post_grasp_pose, abort_on_contact=False)
            is_contact = self.hand.is_in_collision()
            is_width_not_zero = self.hand.get_width() > 0.05 * self.hand.max_width
            is_success = is_contact and is_width_not_zero
            # if is_success and remove:
            for body in self.world.bodies.values():
                if body == self.hand: continue
                if self.hand.hand_body.is_collision_with(body):
                    grasped_obj = body
                        
        except Exception as e:
            print(e)
            is_success = False
        finally:
            self.hand.open(duration=.5)
            if remove_gripper:
                self.remove()
            if return_grasp_obj:
                return grasped_obj
            else: return is_success

    def check_grasps_in_collision(self, grasp_poses:List[SE3]):
        self.reset(finger_constraint=False)
        labels = []
        for grasp_pose in grasp_poses:
            self.hand.reset(grasp_pose)
            labels.append(self.hand.is_in_collision())
        self.remove(remove_constraint=False)
        return np.array(labels)
