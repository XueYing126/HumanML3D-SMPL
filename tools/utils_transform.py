'''
# This code is based on https://github.com/eth-siplab/AvatarPoser
'''
import torch
from human_body_prior.tools.rotation_tools import aa2matrot, matrot2aa


def matrot2sixd(pose_matrot):
    '''
    :param pose_matrot: Nx3x3
    :return: pose_6d: Nx6
    '''
    pose_6d = torch.cat([pose_matrot[:,:3,0], pose_matrot[:,:3,1]], dim=1)
    return pose_6d


def aa2sixd(pose_aa):
    '''
    :param pose_aa Nx3
    :return: pose_6d: Nx6
    '''
    pose_matrot = aa2matrot(pose_aa)
    pose_6d = matrot2sixd(pose_matrot)
    return pose_6d

def sixd2matrot(pose_6d):
    '''
    :param pose_6d: Nx6
    :return: pose_matrot: Nx3x3
    '''
    rot_vec_1 = pose_6d[:,:3]
    rot_vec_2 = pose_6d[:,3:6]
    rot_vec_3 = torch.cross(rot_vec_1, rot_vec_2)
    pose_matrot = torch.stack([rot_vec_1,rot_vec_2,rot_vec_3],dim=-1)
    return pose_matrot

def sixd2aa(pose_6d, batch = False):
    '''
    :param pose_6d: Nx6
    :return: pose_aa: Nx3
    '''
    if batch:
        B,J,C = pose_6d.shape
        pose_6d = pose_6d.reshape(-1,6)
    pose_matrot = sixd2matrot(pose_6d)
    pose_aa = matrot2aa(pose_matrot)
    if batch:
        pose_aa = pose_aa.reshape(B,J,3)
    return pose_aa