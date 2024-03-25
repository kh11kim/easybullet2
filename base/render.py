
import numpy as np
import pybullet as p
from attrs import define
from .world import World
from .pose import SE3, SO3
from numpy.typing import ArrayLike

@define
class CameraIntrinsic:
    width: float
    height: float
    fx: float
    fy: float
    cx: float
    cy: float
    near: float
    far: float

    def get_projection_matrix(self):
        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]])
    
    def get_projection_matrix_opengl(self):
        w, h = self.width, self.height
        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy
        near, far = self.near, self.far
        
        x_scale = 2 / w * fx
        y_scale = 2 / h * fy
        x_shift = 1 - 2 * cx / w
        y_shift = (2 * cy - h) / h
        return np.array([
            [x_scale, 0, x_shift, 0],
            [0, y_scale, y_shift, 0],
            [0, 0, (near+far)/(near-far), 2*near*far/(near-far)],
            [0, 0, -1, 0]
        ]).flatten(order="F")
    
    def depth_to_points(self, depth, eps=0.01):
        height, width = depth.shape
        X, Y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
        pixels = np.stack([X, Y]).reshape(2, -1)
        pixels_homo_coord = np.vstack([pixels, np.ones(pixels.shape[1])])
        P_inv = np.linalg.inv(self.get_projection_matrix())
        obj_pixels_norm = (P_inv @ pixels_homo_coord).T
        points_cam = np.einsum("ij,i->ij", obj_pixels_norm, depth.flatten())
        is_valid = (self.near + eps <= depth) & (depth <= self.far - eps)
        return points_cam[is_valid.flatten()]
    
    

@define
class Camera:
    world:World
    intrinsic:CameraIntrinsic
    
    @staticmethod
    def get_look_at_pose(eye_pos:ArrayLike, target_pos=np.zeros(3), up_vector=np.array([0.,0,1])):
        f = np.asarray(target_pos) - np.asarray(eye_pos)
        f /= np.linalg.norm(f)
        s = np.cross(f, up_vector)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        rot_mat = np.vstack([s, -u, f]).T
        t =np.asarray(eye_pos)
        return SE3(SO3.from_matrix(rot_mat), t)
    
    def render(self, cam_pose:SE3, render_mode="tiny"):
        """output: rgb, depth, seg"""
        cam_pose_opengl = cam_pose @ SE3(SO3.from_euler("xyz", [np.pi,0,0]))
        view_matrix = list(cam_pose_opengl.inverse().as_matrix().flatten("F"))
        proj_matrix = list(self.intrinsic.get_projection_matrix_opengl())
        
        #result: (width, height, rgb, depth, seg)
        renderer = p.ER_BULLET_HARDWARE_OPENGL if render_mode != "tiny" else p.ER_TINY_RENDERER
        result = self.world.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=renderer)
        w, h = self.intrinsic.width, self.intrinsic.height
        far, near = self.intrinsic.far, self.intrinsic.near
        rgb = np.reshape(result[2], (h, w, 4))[:,:,:3] * 1. / 255.
        depth = np.asarray(result[3]).reshape(h,-1)
        seg = np.asarray(result[4])
        depth = far * near / (far - (far - near) * depth)
        return rgb, depth, seg