import numpy as np

from coordutils import mavlink_euler_to_dcm
from utils import to_homo
import config as cfg

def cam_homo_to_iner_xyz_matrix(pos):
    # from drone to inertial frame
    DCM = mavlink_euler_to_dcm(pos['roll'], pos['pitch'], -pos['yaw'])
    iDCM = np.linalg.inv(DCM)

    # sequential transforms from uv plane to xyz plane, from camera to drone, from drone to inertial frames
    CIM = iDCM.dot(
        cfg.C2B.dot(
            cfg.iCaM
        )
    )
    return CIM

def cam_to_iner_homo_matrix(pos):
    # convert inertial frame xyz to uv plane
    CIH = cfg.CaM.dot(
        cfg.iC2B.dot(
            cam_homo_to_iner_xyz_matrix(pos)
        )
    )
    return CIH/CIH[2,2]

def uvs2iner(uvs, pos):
    CIM = cam_homo_to_iner_xyz_matrix(pos)
    iner_xyz = CIM.dot(to_homo(uvs).T)
    iner_xyz = iner_xyz[:2]/iner_xyz[2] * pos['altitude'] 
    
    return iner_xyz.T

def kps2uvs(kps):
    uvs = [[pt.pt[0], pt.pt[1]] for pt in kps]
    
    return np.asarray(uvs)

def kps2iner(kps, pos):
    uvs = kps2uvs(kps)

    return uvs2iner(uvs, pos)

def get_dst_size_HoM(img, HoM):
    h, w = img.shape[:2]
    src_corners = np.asarray([[0,0,1], [w,0,1], [0,h,1], [w,h,1]]).T
    dst_corners = HoM.dot(src_corners)
    dst_corners = dst_corners[:2]/dst_corners[2]
    dst_rb = np.max(dst_corners, axis=1) 
    dst_lt = np.min(dst_corners, axis=1)
    dst_wh = dst_rb - dst_lt

    TrM = np.eye(3)
    TrM[0,2] = -dst_lt[0]
    TrM[1,2] = -dst_lt[1]
    return dst_wh.astype(np.int32).tolist(), TrM.dot(HoM)
