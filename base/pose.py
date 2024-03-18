from __future__ import annotations
from attrs import define, field
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

@define
class SO3:
    _rot: Rotation = field(factory=lambda: Rotation([0,0,0,1.]))

    def __repr__(self) -> str: return f"{self.__class__.__name__}(xyzw={np.round(self.as_quat(), 5)})"

    def __matmul__(self, target: SO3 | np.ndarray) -> SO3 | np.ndarray: 
        if isinstance(target, SO3): return self.multiply(target)
        elif isinstance(target, np.ndarray): return self.apply(target)
        else: raise ValueError("target must be SO3 or np.ndarray")

    def multiply(self, other: SO3) -> SO3: return SO3(self._rot * other._rot)
    def apply(self, target: ArrayLike) -> np.ndarray: return self._rot.apply(target)

    @classmethod
    def random(cls) -> SO3: return cls(Rotation.random())
    @classmethod
    def identity(cls) -> SO3: return cls()
    @classmethod
    def from_quat(cls, xyzw: ArrayLike) -> SO3: return cls(Rotation.from_quat(xyzw))
    @classmethod
    def from_rotvec(cls, rotvec: ArrayLike) -> SO3: return cls(Rotation.from_rotvec(rotvec))
    @classmethod
    def from_wxyz(cls, wxyz): return cls(np.roll(wxyz, -1))
    @classmethod
    def from_euler(cls, seq: str, angles: ArrayLike, degrees=False): return cls(Rotation.from_euler(seq, angles, degrees=degrees))
    @classmethod
    def from_matrix(cls, mat:ArrayLike): return cls(Rotation.from_matrix(mat))

    def as_quat(self): return self._rot.as_quat()
    def as_rotvec(self): return self._rot.as_rotvec()
    def as_matrix(self): return self._rot.as_matrix()
    def as_euler(self, seq: str, degrees: bool=False): return self._rot.as_euler(seq, degrees=degrees)
    def as_wxyz(self): return np.roll(self._rot.as_quat(), 1)
    
    @classmethod
    def exp(cls, tangent): return cls(Rotation.from_rotvec(tangent))
    def log(self): return self._rot.as_rotvec()
    def inverse(self): return SO3(self._rot.inv())

    

@define
class SE3:
    rot: SO3 = field(factory=lambda: SO3())
    trans: ArrayLike = field(factory=lambda: np.array([0,0,0]))
    
    def __repr__(self) -> str:
        xyz: np.ndarray = np.round(self.trans, 5)
        xyzw: np.ndarray = np.round(self.rot.xyzw, 5)
        return f"{self.__class__.__name__}(xyzw={xyzw}, xyz={xyz})"
    
    @classmethod
    def from_matrix(cls, mat: ArrayLike) -> SE3:
        assert isinstance(mat, np.ndarray) and mat.shape == (4, 4)
        rot: SO3 = SO3.from_matrix(mat[:3,:3])
        trans: np.ndarray = mat[:3, -1]
        return cls(rot, trans)
    
    @classmethod
    def from_xyz_xyzw(cls, xyz_xyzw: ArrayLike) -> SE3:
        xyz: np.ndarray = np.array(xyz_xyzw[:3])
        xyzw: np.ndarray = np.array(xyz_xyzw[-4:])
        return cls(rot=SO3.from_quat(xyzw), trans=np.asarray(xyz))
    
    @classmethod
    def from_xyz_wxyz(cls, xyz_wxyz: ArrayLike) -> SE3:
        xyz: np.ndarray = xyz_wxyz[:3]
        rot: SO3 = SO3.from_wxyz(xyz_wxyz[-4:])
        return cls(rot=rot, trans=np.asarray(xyz))
    
    @classmethod
    def random(cls, trans_lower: ArrayLike, trans_upper: ArrayLike) -> SE3:
        rot: SO3 = SO3.random()
        trans: np.ndarray = np.random.uniform(trans_lower, trans_upper)
        return cls(rot, trans)
    
    @classmethod
    def identity(cls) -> SE3: return cls()
    
    def as_matrix(self) -> np.ndarray:
        return np.vstack(
            (np.c_[self.rot.as_matrix(), self.trans], [0.0, 0.0, 0.0, 1.0])
        )
        
    def as_xyz_xyzw(self) -> np.ndarray:
        return np.hstack([self.trans, self.rot.as_quat()])
    
    def as_xyz_wxyz(self) -> np.ndarray:
        return np.hstack([self.trans, self.rot.as_wxyz()])
    
    def multiply(self, other: SE3) -> SE3:
        rotation: SO3 = self.rot @ other.rot
        translation: np.ndarray = self.rot.apply(other.trans) + self.trans
        return SE3(rotation, translation)
    
    def apply(self, target: ArrayLike) -> ArrayLike:
        assert target.shape == (3,) or target.shape[1] == 3
        return self.rot.apply(target) + self.trans
    
    def inverse(self) -> SE3:
        rot: SO3 = self.rot.inverse()
        trans: np.ndarray = -rot.apply(self.trans)
        return SE3(rot, trans)
    
    
if __name__ == "__main__":
    rot = SO3()
    rot2 = SO3.random()
    arr = np.array([1,0,0])
    print(rot@arr)