"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--save_dir', default='shapenet_ericyi_ply', required=False)
args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir

SHAPE_CATEGORIES = ('03642806', '02958343', '04225987', '03001627', '03261776', '03790512', '03797390', '02691156',
                    '03948459', '03636649', '02773838', '02954340', '04099429', '03467517', '04379243', '03624134')
off = [28,  8, 44, 12, 16, 30, 36,  0, 38, 24,  4,  6, 41, 19, 47, 22]
off_map = dict(zip(SHAPE_CATEGORIES, off))

CMAP = ((255, 255, 0),
        (128, 255, 255),
        (128, 0, 255),
        (255, 0, 0),
        (255, 128, 0),
        (0, 255, 0),
        (0, 0, 255))

header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property uchar label
end_header'''

fmt = '%.6f %.6f %.6f %.6f %.6f %.6f %d %d %d %d'

for subset in ('val', 'test', 'train'):
    sample_list_path = os.path.join(data_dir, 'train_test_split', 'shuffled_{}_file_list.json'.format(subset))
    os.makedirs(os.path.join(save_dir, subset), exist_ok=True)
    sample_list = json.load(open(sample_list_path))
    print('processing {} {} samples ... '.format(len(sample_list), subset), flush=True, end='')
    for sample in sample_list:
        category, sample_id = str(sample).split('/')[1:]
        os.makedirs(os.path.join(save_dir, subset, category), exist_ok=True)
        data = np.loadtxt(os.path.join(data_dir, category, '{}.txt'.format(sample_id)))
        data[:, -1] -= off_map[category]
        data = np.vstack([np.concatenate((d[:6], CMAP[int(d[-1])], [int(d[-1])])) for d in data])
        np.savetxt(os.path.join(save_dir, subset, category, '{}.ply'.format(sample_id)), data,
                   fmt=fmt,
                   header=header.format(len(data)),
                   comments='')
    print('done!', flush=True)

