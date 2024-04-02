'''
Regulate the motion to start from origin

Input  folder: './joints', 
Output folder: './HumanML3D/smpl'
'''
import os
import numpy as np
from tqdm import tqdm
import os

def process(file_name, data_dir, save_dir):
    data = np.load(os.path.join(data_dir, file_name),allow_pickle=True).item() # 'bdata_poses', 'bdata_trans', 'betas', 'gender', 'jtr'
    poses = data['bdata_poses']
    trans = data['bdata_trans']

    '''Move the first pose to origin'''
    trans[:, [0, 1]] -= trans[0, [0, 1]] # the global body position x, y 

    # '''All initially face Y+/Z+'''
    # init_root_rot = poses[0, [0, 1, 2]] # the global root orientation

    new_data={
        'bdata_poses': poses, 
        'bdata_trans':trans, 
        'betas':data['betas'], 
        'gender':data['gender'],
    }
    np.save(os.path.join(save_dir, file_name), new_data)


if __name__ == "__main__":
    data_dir = './joints/'
    save_dir = './HumanML3D/smpl/'

    os.makedirs(save_dir, exist_ok=True)

    file_list =  os.listdir(data_dir)
    for file_name in tqdm(file_list):
        process(file_name, data_dir, save_dir)