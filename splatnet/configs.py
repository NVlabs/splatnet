"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import sys

# project root
ROOT_DIR = os.path.abspath(os.path.join(__file__, '..', '..'))

# required for custom layers
sys.path.append(os.path.join(ROOT_DIR, 'splatnet'))
sys.path.append(os.path.join(ROOT_DIR, 'splatnet', 'dataset'))

# modify these if you put data in non-default locations
FACADE_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'ruemonge428')
SHAPENET3D_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'shapenet_ericyi_ply')
SHAPENET2D3D_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'shapenet_2d3d_h5')

# facade global variables

FACADE_CATEGORIES = ('wall', 'sky', 'balcony', 'window', 'door', 'shop', 'roof')

FACADE_CMAP = ((255, 255, 0),
               (128, 255, 255),
               (128, 0, 255),
               (255, 0, 0),
               (255, 128, 0),
               (0, 255, 0),
               (0, 0, 255))

# shapenet global variables

SN_CATEGORIES = ('03642806', '02958343', '04225987', '03001627', '03261776', '03790512', '03797390', '02691156',
                 '03948459', '03636649', '02773838', '02954340', '04099429', '03467517', '04379243', '03624134')

SN_CATEGORY_NAMES = ('laptop', 'car', 'skateboard', 'chair', 'earphone', 'motorbike', 'mug', 'airplane',
                     'pistol', 'lamp', 'bag', 'cap', 'rocket', 'guitar', 'table', 'knife')

SN_NUM_PART_CATEGORIES = (2, 4, 3, 4, 3, 6, 2, 4,
                          3, 4, 2, 2, 3, 3, 3, 2)

SN_CMAP = ((255, 255, 0),
           (128, 255, 255),
           (128, 0, 255),
           (255, 0, 0),
           (255, 128, 0),
           (0, 255, 0),
           (0, 0, 255))

