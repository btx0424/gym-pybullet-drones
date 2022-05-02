import numpy as np

def xyz2rpy(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz / (np.linalg.norm(xyz, axis=-1, keepdims=True)+1e-5) # xyz is of shape (*, 3)
    x, y, z = xyz.T
    return np.stack([
        np.zeros_like(x), 
        np.arcsin(z), 
        np.arctan2(y, x)]).T

def rpy2xyz(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy.T # rpy is of shape (*, 3)
    xyz = np.stack([
        np.cos(yaw)*np.cos(pitch),
        np.sin(yaw)*np.cos(pitch),
        np.sin(pitch)]).T
    return xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    