'''
credit to joints2smpl
https://github.com/wangsen1312/joints2smpl

Use SMPLify to process humanact12 dataset and obtain SMPL parameters

Input folder: './pose_data/humanact12'
Output folder: './humanact12/
'''

import os
import numpy as np
from tqdm import tqdm
import torch
from joints2smpl.src import config
import smplx
import h5py
from joints2smpl.src.smplify import SMPLify3D
import random

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

num_joints = 22 
joint_category = "AMASS"
num_smplify_iters = 150
fix_foot = False

def Joints2SMPL(file_name, data_dir, save_dir):
    input_joints = np.load(os.path.join(data_dir, file_name))

    input_joints = input_joints[:, :, [0, 2, 1]] # amass stands on x, y

    '''XY at origin'''
    input_joints[..., [0, 1]] -= input_joints[0, 0, [0, 1]]

    '''Put on Floor'''
    floor_height = input_joints[:, :, 2].min()
    input_joints[:, :, 2] -= floor_height


    batch_size = input_joints.shape[0]

    smplmodel = smplx.create(config.SMPL_MODEL_DIR,
                            model_type="smpl", gender="neutral", ext="pkl",
                            batch_size=batch_size).to(device)

    # ## --- load the mean pose as original ----
    smpl_mean_file = config.SMPL_MEAN_FILE

    file = h5py.File(smpl_mean_file, 'r')
    init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(batch_size, 1).float().to(device)
    init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(batch_size, 1).float().to(device)
    cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(device)

    # # #-------------initialize SMPLify
    smplify = SMPLify3D(smplxmodel=smplmodel,
                        batch_size=batch_size,
                        joints_category=joint_category,
                        num_iters=num_smplify_iters,
                        device=device)


    keypoints_3d = torch.Tensor(input_joints).to(device).float()

    pred_betas = init_mean_shape
    pred_pose = init_mean_pose
    pred_cam_t = cam_trans_zero

    if joint_category == "AMASS":
        confidence_input = torch.ones(num_joints)
        # make sure the foot and ankle
        if fix_foot == True:
            confidence_input[7] = 1.5
            confidence_input[8] = 1.5
            confidence_input[10] = 1.5
            confidence_input[11] = 1.5
    else:
        print("Such category not settle down!")


    new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
    new_opt_cam_t, new_opt_joint_loss = smplify(
        pred_pose.detach(),
        pred_betas.detach(),
        pred_cam_t.detach(),
        keypoints_3d,
        conf_3d=confidence_input.to(device),
        # seq_ind=idx
    )

    poses = new_opt_pose.detach().cpu().numpy()
    betas = new_opt_betas.mean(axis=0).detach().cpu().numpy()
    trans = keypoints_3d[:, 0].detach().cpu().numpy()

    input_joints = input_joints[:, :, [0, 2, 1]] # jts stands on x, z
    input_joints[..., 0] *= -1
    param = {
        'bdata_poses': poses,
        'bdata_trans': trans,
        'betas': betas,
        'gender': 'male',
        'jtr' : input_joints,
    }

    np.save(os.path.join(save_dir, file_name), param)


if __name__ == "__main__":
    data_dir = './pose_data/humanact12/humanact12'
    save_dir = './humanact12/humanact12/'

    os.makedirs(save_dir, exist_ok=True)

    file_list =  os.listdir(data_dir)
    random.shuffle(file_list)
    for file_name in tqdm(file_list):
        if os.path.exists(os.path.join(save_dir, file_name)):
            print(f'{os.path.join(save_dir, file_name)} already exists')
            continue
        Joints2SMPL(file_name, data_dir, save_dir)
