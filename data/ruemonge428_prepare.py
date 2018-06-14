"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import argparse
import urllib.request
import shutil
import numpy as np
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='ruemonge428', required=False)
args = parser.parse_args()
data_dir = args.data_dir

pcl_gt_train_path = os.path.join(data_dir, 'pcl_gt_train.ply')
pcl_gt_test_path = os.path.join(data_dir, 'pcl_gt_test.ply')
pcl_all_path = os.path.join(data_dir, 'pcl.ply')
pcl_height_path = os.path.join(data_dir, 'pcl_height.mat')
save_pcl_train_path = os.path.join(data_dir, 'pcl_train.ply')
save_pcl_test_path = os.path.join(data_dir, 'pcl_test.ply')

# download mat file with height values
mat_url = 'http://maxwell.cs.umass.edu/splatnet-data/pcl_height.mat'
print('Downloading file of point height ... ', end='', flush=True)
with urllib.request.urlopen(mat_url) as response, open(pcl_height_path, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)
print('done!')

print('Preparing final data files ... ', end='', flush=True)
pcl_gt_train = np.loadtxt(pcl_gt_train_path, skiprows=10)
pcl_gt_test = np.loadtxt(pcl_gt_test_path, skiprows=10)
pcl_all = np.concatenate((np.loadtxt(pcl_all_path, skiprows=13), 
                          loadmat(pcl_height_path)['height']), axis=1)

cmap = [[   0,    0,    0],
        [ 255,  255,    0],
        [ 128,  255,  255],
        [ 128,    0,  255],
        [ 255,    0,    0],
        [ 255,  128,    0],
        [   0,  255,    0],
        [   0,    0,  255]]

pcl_label_train = np.zeros(pcl_all.shape[0])
pcl_label_test = np.zeros(pcl_all.shape[0])
for i, c in enumerate(cmap):
    pcl_label_train[np.where(np.all(pcl_gt_train[:, 3:] == c, axis=1))[0]] = i
    pcl_label_test[np.where(np.all(pcl_gt_test[:, 3:] == c, axis=1))[0]] = i
    
ind_train = np.where(pcl_label_train > 0)[0]
ind_test = np.where(pcl_label_test > 0)[0]
pcl_label = pcl_label_train + pcl_label_test
pcl_all = np.concatenate((pcl_all, pcl_label.reshape(-1, 1)), axis=1)
pcl_train = pcl_all[ind_train, :]
pcl_test = pcl_all[ind_test, :]

header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar diffuse_red
property uchar diffuse_green
property uchar diffuse_blue
property float height
property uchar label
end_header'''.format(pcl_train.shape[0])
fmt = '%.6f %.6f %.6f %.6f %.6f %.6f %d %d %d %.8f %d'
np.savetxt(save_pcl_train_path, pcl_train, fmt=fmt, header=header, comments='')

header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar diffuse_red
property uchar diffuse_green
property uchar diffuse_blue
property float height
property uchar label
end_header'''.format(pcl_test.shape[0])
fmt = '%.6f %.6f %.6f %.6f %.6f %.6f %d %d %d %.8f %d'
np.savetxt(save_pcl_test_path, pcl_test, fmt=fmt, header=header, comments='')

print('done!')

