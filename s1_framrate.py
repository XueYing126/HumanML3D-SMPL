'''
Downsample AMASS data to 20 fps
Input  folder: './amass_data', 
Output folder: './pose_data'
jtr only for alignment visualization
'''
import os
import torch
import numpy as np
from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel

# Choose the device to run the body model on.
comp_device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

male_bm_path = './body_models/smplh/male/model.npz'
male_dmpl_path = './body_models/dmpls/male/model.npz'
female_bm_path = './body_models/smplh/female/model.npz'
female_dmpl_path = './body_models/dmpls/female/model.npz'

num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(comp_device)
female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path).to(comp_device)

# expect 20fps
ex_fps = 20


def amass_to_pose(src_path, save_path):
        bdata = np.load(src_path, allow_pickle=True)
        # ['trans':(frame_num, 3), 'gender':array('female', dtype='<U6'), 'mocap_framerate':array(120.), 'betas':(16,), 'dmpls':(frame_num, 8), 'poses':(frame_num, 156)]
        try:
            fps = bdata['mocap_framerate']
        except:
            return 0
        
        if str(bdata['gender']) == 'male':
            bm = male_bm
        else:
            bm = female_bm

        stride = int(round(fps / ex_fps))
        if 'SSM' in src_path:
            stride = 6
        num_betas = 10

        bdata_poses = bdata['poses'][::stride,...]
        bdata_trans = bdata['trans'][::stride,...]

        body_parms = {
            'root_orient': torch.Tensor(bdata_poses[:, :3]).to(comp_device),  # controls the global root orientation
            'pose_body': torch.Tensor(bdata_poses[:, 3:66]).to(comp_device),
            'pose_hand': torch.Tensor(bdata_poses[:, 66:]).to(comp_device),
            'trans': torch.Tensor(bdata_trans).to(comp_device),               # controls the global body position
            'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=len(bdata_trans), axis=0)).to(comp_device),
        }
        
        body_world = bm(**body_parms)
        jtr = body_world.Jtr.detach().cpu().numpy()
        
        # exchange y z, human stands on xy -> xz plane
        jtr = jtr[:, :, [0, 2, 1]]
        
        save_data={
            'bdata_poses': bdata_poses,
            'bdata_trans': bdata_trans,
            'betas': bdata['betas'],
            'gender': bdata['gender'],
            'jtr':jtr,
        }

        np.save(save_path, save_data)
        return fps


if __name__ == '__main__':
   
    # create pose_data/ to save processed data
    paths = [] # 14055 all files 
    folders = [] # 475 all folders
    dataset_names = [] # 18 datasets
    for root, dirs, files in os.walk('./amass_data'):
        folders.append(root)
        for name in files:
            dataset_name = root.split('/')[2]
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)
            paths.append(os.path.join(root, name))

    save_root = './pose_data'
    save_folders = [folder.replace('./amass_data', './pose_data') for folder in folders]
    for folder in save_folders:
        os.makedirs(folder, exist_ok=True)
    group_path = [[path for path in paths if name in path] for name in dataset_names]

    
    all_count = sum([len(paths) for paths in group_path]) #14055
    cur_count = 0
    
    for paths in group_path:
        dataset_name = paths[0].split('/')[2]
        pbar = tqdm(paths)
        pbar.set_description('Processing: %s'%dataset_name)
        for path in pbar:
            save_path = path.replace('./amass_data', './pose_data').replace('.npz', '.npy')
            fps = amass_to_pose(path, save_path)
            
        cur_count += len(paths)
        print('Processed / All (fps %d): %d/%d'% (fps, cur_count, all_count) )