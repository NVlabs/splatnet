"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import numpy as np
from numpy.linalg import eig
import caffe
from splatnet.utils import rotate_3d
from splatnet.configs import FACADE_DATA_DIR


def ordered_points(subset, dims='x_y_z_nx_ny_nz_r_g_b_h,l', order_dim='z', val_ratio=0.0, root=FACADE_DATA_DIR):
    pcl_train_path = os.path.join(root, 'pcl_train.ply')
    pcl_test_path = os.path.join(root, 'pcl_test.ply')

    # data files have 11 columns: (x, y, z, nx, ny, nz, r, g, b, height, label)
    feat_dict = dict(zip('x_y_z_nx_ny_nz_r_g_b_h_l'.split('_'), range(11)))
    feat_idxs, feat_scales = [], []
    for g in dims.split(','):
        feat_idxs.append([])
        feat_scales.append([])
        for f in g.split('_'):
            if f.find('*') >= 0:
                feat_idxs[-1].append(feat_dict[f[:f.find('*')]])
                feat_scales[-1].append(float(f[f.find('*') + 1:]))
            elif f.find('/') >= 0:
                feat_idxs[-1].append(feat_dict[f[:f.find('/')]])
                feat_scales[-1].append(1.0 / float(f[f.find('/') + 1:]))
            else:
                feat_idxs[-1].append(feat_dict[f])
                feat_scales[-1].append(1.0)

    if subset == 'train':
        pcl_data = np.loadtxt(pcl_train_path, skiprows=15)
        num_val = int(val_ratio * len(pcl_data))
        order_idx = np.argsort(pcl_data[:, 2])[num_val:]
        pcl_data = pcl_data[order_idx, :]
    elif subset == 'val':
        pcl_data = np.loadtxt(pcl_train_path, skiprows=15)
        num_val = int(val_ratio * len(pcl_data))
        order_idx = np.argsort(pcl_data[:, 2])[:num_val]
        pcl_data = pcl_data[order_idx, :]
    elif subset == 'test':
        pcl_data = np.loadtxt(pcl_test_path, skiprows=15)
    else:
        raise ValueError('Unknown subset: ' + subset)

    order_dim = feat_dict[order_dim]
    if subset not in {'train', 'val'} or order_dim != 2:
        order_idx = np.argsort(pcl_data[:, order_dim])
        pcl_data = pcl_data[order_idx, :]

    return tuple([pcl_data[:, idx] * sc for (idx, sc) in zip(feat_idxs, feat_scales)])


def points(subset, dims='x_y_z_nx_ny_nz_r_g_b_h,l', shuffle=False, val_ratio=0.0, root=FACADE_DATA_DIR):
    pcl_train_path = os.path.join(root, 'pcl_train.ply')
    pcl_test_path = os.path.join(root, 'pcl_test.ply')

    # data files have 11 columns: (x, y, z, nx, ny, nz, r, g, b, height, label)
    feat_dict = dict(zip('x_y_z_nx_ny_nz_r_g_b_h_l'.split('_'), range(11)))
    feat_idxs, feat_scales = [], []
    for g in dims.split(','):
        feat_idxs.append([])
        feat_scales.append([])
        for f in g.split('_'):
            if f.find('*') >= 0:
                feat_idxs[-1].append(feat_dict[f[:f.find('*')]])
                feat_scales[-1].append(float(f[f.find('*') + 1:]))
            elif f.find('/') >= 0:
                feat_idxs[-1].append(feat_dict[f[:f.find('/')]])
                feat_scales[-1].append(1.0 / float(f[f.find('/') + 1:]))
            else:
                feat_idxs[-1].append(feat_dict[f])
                feat_scales[-1].append(1.0)

    order_dim = 2
    if subset == 'train':
        pcl_data = np.loadtxt(pcl_train_path, skiprows=15)
        num_val = int(val_ratio * len(pcl_data))
        order_idx = np.sort(np.argsort(pcl_data[:, order_dim])[num_val:])
        pcl_data = pcl_data[order_idx, :]
    elif subset == 'val':
        pcl_data = np.loadtxt(pcl_train_path, skiprows=15)
        num_val = int(val_ratio * len(pcl_data))
        order_idx = np.sort(np.argsort(pcl_data[:, order_dim])[:num_val])
        pcl_data = pcl_data[order_idx, :]
    elif subset == 'test':
        pcl_data = np.loadtxt(pcl_test_path, skiprows=15)
    else:
        raise ValueError('Unknown subset: ' + subset)

    if shuffle:
        order_idx = np.random.permutation(len(pcl_data))
        pcl_data = pcl_data[order_idx]

    return tuple([pcl_data[:, idx] * sc for (idx, sc) in zip(feat_idxs, feat_scales)])


class InputFacade(caffe.Layer):
    def _restart(self):
        self.data[...] = self.data_copy
        num_points = len(self.data)
        points_per_batch = len(self.data)
        if self.mode == 'random':
            idx = np.random.permutation(num_points)
            self.data, self.label = self.data[idx], self.label[idx]
            self.idx = 0
        elif self.mode == 'ordered':
            # pick a random starting index as a form of data augmentation
            # self.idx = np.random.randint(0, num_points - points_per_batch + 1)
            self.idx = np.random.randint(0, min(points_per_batch, num_points - points_per_batch + 1))

    def setup(self, bottom, top):
        params = dict(mode='ordered', batch_size=1, sample_size=-1,
                      jitter_color=0.5, jitter_h=0.001, jitter_rotation=True,
                      subset='train', val_ratio=0.0, feat_dims='nx_ny_nz_r_g_b_h',
                      root=FACADE_DATA_DIR)
        params.update(eval(self.param_str))
        self.mode = params['mode']
        self.batch_size = params['batch_size']
        self.sample_size = params['sample_size']

        feat_dims = []
        for f in params['feat_dims'].split('_'):
            if f.find('*') >= 0:
                feat_dims.append((f[:f.find('*')], float(f[f.find('*')+1:])))
            elif f.find('/') >= 0:
                feat_dims.append((f[:f.find('/')], 1.0 / float(f[f.find('/') + 1:])))
            else:
                feat_dims.append((f, 1.0))
        self.raw_dims = []
        for feat_group in [['x', 'y', 'z'], ['nx', 'ny', 'nz'], ['r', 'g', 'b'], ['h']]:
            if np.any([f in feat_group for f, _ in feat_dims]):
                self.raw_dims.extend(feat_group)
        self.feat_scales = [(self.raw_dims.index(f), s) for f, s in feat_dims]

        self.data, self.label = ordered_points(params['subset'],
                                               dims='_'.join(self.raw_dims) + ',l',
                                               val_ratio=params['val_ratio'],
                                               root=params['root'])
        self.data_copy = self.data.copy()
        self.label -= 1  # label starts from 0
        self.top_names = ['data', 'label']
        self.top_channels = [len(self.feat_scales), 1]

        if self.sample_size == -1:
            self.sample_size = self.data.shape[0]

        if len(self.data) < self.sample_size * self.batch_size:
            raise Exception('Too few samples ({}). Is batch size too large?'.format(len(self.data)))

        if len(top) != len(self.top_names):
            raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                            (len(self.top_names), len(top)))

        # prepare for jittering
        max_points = 100000
        part_idx = np.random.permutation(len(self.data))[:min(len(self.data), max_points)]
        if 'r' in self.raw_dims and params['jitter_color'] != 0:
            feat_idx = self.raw_dims.index('r')
            eigw, eigv = eig(np.cov(self.data[np.ix_(part_idx, range(feat_idx, feat_idx + 3))].T))
            self.jitter_color = params['jitter_color'] * eigv * np.sqrt(eigw)
        else:
            self.jitter_color = None
        if 'h' in self.raw_dims and params['jitter_h'] != 0:
            feat_idx = self.raw_dims.index('h')
            std = np.std(self.data[part_idx, feat_idx])
            self.jitter_h = params['jitter_h'] * std
        else:
            self.jitter_h = None
        self.jitter_rotation = params['jitter_rotation']

        self._restart()

    def reshape(self, bottom, top):
        for top_index, name in enumerate(self.top_names):
            shape = (self.batch_size, self.top_channels[top_index], 1, self.sample_size)
            top[top_index].reshape(*shape)

    def forward(self, bottom, top):
        points_per_batch = self.sample_size * self.batch_size
        data, label = self.data[self.idx:self.idx+points_per_batch].reshape(self.batch_size, self.sample_size, -1), \
                      self.label[self.idx:self.idx+points_per_batch]

        # jittering TODO: improve performance (e.g. move some computation to done at once for all samples?)
        for i in range(self.batch_size):
            if self.jitter_color is not None:
                feat_idx = self.raw_dims.index('r')
                data[i, :, feat_idx:feat_idx+3] += np.random.randn(3).dot(self.jitter_color.T)
                # clipping to [0, 255]
                data[i, :, feat_idx:feat_idx+3] = np.maximum(0.0, np.minimum(255.0, data[i, :, feat_idx:feat_idx+3]))
            if self.jitter_h is not None:
                feat_idx = self.raw_dims.index('h')
                data[i, :, feat_idx] += np.random.randn(self.sample_size) * self.jitter_h
            if self.jitter_rotation:
                rotations = (('z', np.random.rand() * np.pi / 8 - np.pi / 16),
                             ('x', np.random.rand() * np.pi / 8 - np.pi / 16),
                             ('y', np.random.rand() * np.pi * 2))
                if 'x' in self.raw_dims:
                    feat_idx = self.raw_dims.index('x')
                    center = (np.mean(data[i, :, feat_idx]),
                              np.max(data[i, :, feat_idx + 1]),
                              np.mean(data[i, :, feat_idx + 2]))
                    data[i, :, feat_idx:feat_idx + 3] = rotate_3d(data[i, :, feat_idx:feat_idx + 3], rotations, center)
                if 'nx' in self.raw_dims:
                    feat_idx = self.raw_dims.index('nx')
                    data[i, :, feat_idx:feat_idx + 3] = rotate_3d(data[i, :, feat_idx:feat_idx + 3], rotations)

        # slicing and scaling
        idxs, scs = [v[0] for v in self.feat_scales], [v[1] for v in self.feat_scales]
        data = data[:, :, idxs] * np.array(scs)

        top[0].data[...] = data.reshape(self.batch_size, self.sample_size, -1, 1).transpose(0, 2, 3, 1)
        top[1].data[...] = label.reshape(self.batch_size, self.sample_size, -1, 1).transpose(0, 2, 3, 1)

        self.idx += points_per_batch
        if self.idx + points_per_batch > len(self.data):
            self._restart()

    def backward(self, top, propagate_down, bottom):
        pass

