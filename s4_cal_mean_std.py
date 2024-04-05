'''
Compute Mean, Std

Input folder: './HumanML3D/smpl'
Output folder: './HumanML3D/
'''
import os
import numpy as np
   
if __name__ == "__main__":
    data_dir = './HumanML3D/smpl/'
    save_dir = './HumanML3D/'

    file_list = os.listdir(data_dir)
    poses6d_list = []

    for file in file_list:
        data = np.load(os.path.join(data_dir, file),allow_pickle=True).item()
        if data['pose_6d'].shape[1] < 312: #humanact12
            zeros_to_append = np.zeros((data['pose_6d'].shape[0], 312 - data['pose_6d'].shape[1]))
            data['pose_6d'] = np.concatenate((data['pose_6d'], zeros_to_append), axis=1)
        poses6d_list.append(data['pose_6d'])

    poses6d = np.concatenate(poses6d_list, axis=0)

    poses6d_mean = poses6d.mean(axis=0)
    poses6d_std = poses6d.std(axis=0)

    np.save(os.path.join(save_dir, 'Mean_6d.npy'), poses6d_mean)
    np.save(os.path.join(save_dir, 'Std_6d.npy'), poses6d_std)