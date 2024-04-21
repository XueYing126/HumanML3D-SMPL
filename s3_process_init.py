'''
Regulate the motion to start from origin
Add 6D rotation representation, inferred joints from 6d

Input  folder: './joints', 
Output folder: './HumanML3D/smpl'
'''
import os
import numpy as np
from tqdm import tqdm
from tools import utils_transform
from human_body_prior.body_model.body_model import BodyModel
import torch

# Choose the device to run the body model on.
comp_device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

bm_path = './body_models/smplh/neutral/model.npz'
bm = BodyModel(bm_fname=bm_path, num_betas=10).to(comp_device)

def process(file_name, data_dir, save_dir):
    data = np.load(os.path.join(data_dir, file_name),allow_pickle=True).item() # 'bdata_poses', 'bdata_trans', 'betas', 'gender', 'jtr'
    poses = data['bdata_poses']
    trans = data['bdata_trans']

    '''Move the first pose to origin'''
    trans[:, [0, 1]] -= trans[0, [0, 1]] # the global position x, y 

    # '''All initially face Y+/Z+'''
    # init_root_rot = poses[0, [0, 1, 2]] # the global root orientation

    '''from axis-angle to 6D'''
    pose_aa = torch.Tensor(poses).reshape(-1,3)
    pose_6d = utils_transform.aa2sixd(pose_aa).reshape(poses.shape[0],-1).numpy() # [fn, 312]

    '''from 6D to axis-angle'''
    rot_6d =  torch.Tensor(pose_6d).reshape(-1, 6)
    rot_aa_b = utils_transform.sixd2aa(rot_6d).reshape(poses.shape[0], -1).numpy()

    body_parms = {
            'root_orient': torch.Tensor(rot_aa_b[:, :3]).to(comp_device),  # controls the global root orientation
            'pose_body': torch.Tensor(rot_aa_b[:, 3:66]).to(comp_device),
            'trans': torch.Tensor(trans).to(comp_device),               # controls the global body position
    }
    
    body_world = bm(**body_parms)
    jtr = body_world.Jtr.detach().cpu().numpy()
    # exchange y z, human stands on xy -> xz plane
    jtr = jtr[:, :22, [0, 2, 1]]


    new_data={
        'bdata_poses': poses, 
        'bdata_trans':trans, 
        'betas':data['betas'], 
        'gender':data['gender'],
        'pose_6d': pose_6d,
        'jtr': jtr,
    }
    np.save(os.path.join(save_dir, file_name), new_data)


if __name__ == "__main__":
    data_dir = './joints/'
    save_dir = './HumanML3D/smpl/'

    os.makedirs(save_dir, exist_ok=True)

    file_list =  os.listdir(data_dir)
    for file_name in tqdm(file_list):
        process(file_name, data_dir, save_dir)