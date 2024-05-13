'''
Segment the motion to specified time periods
Mirror the motion
Relocate with new names: 000001.npy (align with text)

Input  folder: './pose_data', 
Output folder: './joints'
'''
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin

def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

def swap_left_right_pose(data):
    data = data.copy() # (num_frame, num_joint*3) num_joint: 52 (humanact12 24 -> 52)
    data = data.reshape(data.shape[0], -1, 3)

    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = range(22, 37)
    right_hand_chain = range(37, 52)

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    
    # mirror joint rot (x, y, z) -> (x, -y, -z)
    data[..., [1,2]] *=-1
    return data.reshape(data.shape[0], -1)

if __name__ == '__main__':
    save_dir = './joints'
    os.makedirs(save_dir, exist_ok=True)

    index_path = './index.csv'
    index_file = pd.read_csv(index_path)
    total_amount = index_file.shape[0]
    fps = 20

    for i in tqdm(range(total_amount)):
        source_path = index_file.loc[i]['source_path']
        new_name = index_file.loc[i]['new_name']
        start_frame = index_file.loc[i]['start_frame']
        end_frame = index_file.loc[i]['end_frame']

        if not os.path.exists(source_path):
            print(f'can not find {source_path}')
            continue
        
        data = np.load(source_path, allow_pickle=True).item()
        bdata_poses = data['bdata_poses']
        bdata_trans = data['bdata_trans']
        betas = data['betas']
        jtr = data['jtr']
        
        if bdata_poses.shape[1] < 156 and 'humanact12' in source_path: # humanact12, concatenate zeros columns for hands pose, smpl -> smpl-h
            zeros_matrix = np.zeros((bdata_poses.shape[0], 156 - bdata_poses.shape[1]))
            bdata_poses = np.concatenate((bdata_poses, zeros_matrix), axis=1)
        
        if 'humanact12' not in source_path:
            if 'Eyes_Japan_Dataset' in source_path or 'MPI_HDM05' in source_path:
                bdata_poses = bdata_poses[3*fps:]
                bdata_trans = bdata_trans[3*fps:]
                jtr = jtr[3*fps:]
            if 'TotalCapture' in source_path or 'MPI_Limits' in source_path:
                bdata_poses = bdata_poses[1*fps:]
                bdata_trans = bdata_trans[1*fps:]
                jtr = jtr[1*fps:]
            if 'Transitions_mocap' in source_path:
                bdata_poses = bdata_poses[int(0.5*fps):]
                bdata_trans = bdata_trans[int(0.5*fps):]
                jtr = jtr[int(0.5*fps):]

            bdata_poses = bdata_poses[start_frame:end_frame]
            bdata_trans = bdata_trans[start_frame:end_frame]
            jtr = jtr[start_frame:end_frame]
            
            jtr[..., 0] *= -1
        
        jtr_m = swap_left_right(jtr)
        bdata_poses_m = swap_left_right_pose(bdata_poses)

        # mirror translation in x axis
        bdata_trans_m = bdata_trans.copy()
        bdata_trans_m[..., 0] *= -1
        

        new_data = {
            'bdata_poses': bdata_poses,
            'bdata_trans': bdata_trans,
            'betas': betas,
            'gender': data['gender'],
            'jtr':jtr,
        }
        new_data_m = {
            'bdata_poses': bdata_poses_m,
            'bdata_trans': bdata_trans_m,
            'betas': betas,
            'gender': data['gender'],
            'jtr': jtr_m,
        }

        np.save(pjoin(save_dir, new_name), new_data)
        np.save(pjoin(save_dir, 'M'+new_name), new_data_m)