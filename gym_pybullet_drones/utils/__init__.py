import numpy as np

MAX_RPY = np.array([np.pi, np.pi/2, np.pi])

def xyz2rpy(xyz: np.ndarray, normalize=False) -> np.ndarray:
    xyz = xyz / (np.linalg.norm(xyz, axis=-1, keepdims=True)+1e-6) # xyz is of shape (*, 3)
    x, y, z = xyz.T
    rpy = np.stack([
        np.zeros_like(x), 
        np.arcsin(z), 
        np.arctan2(y, x)]).T
    if normalize: rpy /= MAX_RPY
    return rpy

def rpy2xyz(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy.T # rpy is of shape (*, 3)
    xyz = np.stack([
        np.cos(yaw)*np.cos(pitch),
        np.sin(yaw)*np.cos(pitch),
        np.sin(pitch)]).T
    return xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    