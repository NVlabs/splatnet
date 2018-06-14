"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import time
import tempfile
import numpy as np
from numpy import sin, cos


def get_prototxt(net_proto, save_path=None):
    if save_path:
        f = open(save_path, mode='w+')
    else:
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net_proto))
    f.close()

    return f.name


def parse_channel_scale(channel_scale_str, channel_str=False, scale_str=False):
    channels, scales = [], []
    for f in channel_scale_str.split('_'):
        if f.find('*') >= 0:
            channels.append(f[:f.find('*')] if channel_str else int(f[:f.find('*')]))
            scales.append(f[f.find('*'):] if scale_str else float(f[f.find('*') + 1:]))
        elif f.find('/') >= 0:
            channels.append(f[:f.find('/')] if channel_str else int(f[:f.find('/')]))
            scales.append(f[f.find('/'):] if scale_str else (1.0 / float(f[f.find('/') + 1:])))
        else:
            channels.append(f if channel_str else int(f))
            scales.append('' if scale_str else 1.0)
    return channels, scales


def map_channel_scale(channel_scale_str, refs):
    channels, scales = parse_channel_scale(channel_scale_str, channel_str=True, scale_str=True)
    return '_'.join([str(f) + s for f, s in zip([np.where([i == v for v in refs])[0][0] for i in channels], scales)])


def rotate_3d(xyz, rotations, center=(0, 0, 0)):
    """
    Apply rotations to 3d points.
    :param xyz: N x 3 ndarray
    :param rotations: a list of rotations, each as (axis, angle)
    :param center: optionally, rotate around a non-origin center
    :return: rotated N x 3 ndarray (a copy -- original xyz is untouched)
    """
    rot = np.eye(3)
    xyz = xyz.copy()

    # translate center to origin
    if np.any(center):
        xyz -= center

    # rotation matrix
    for axis, theta in rotations:
        if axis in {'x', 'X'}:
            rot_axis = np.array([[1, 0, 0],
                                 [0, cos(theta), -sin(theta)],
                                 [0, sin(theta), cos(theta)]])
        elif axis in {'y', 'Y'}:
            rot_axis = np.array([[cos(theta), 0, sin(theta)],
                                 [0, 1, 0],
                                 [-sin(theta), 0, cos(theta)]])
        elif axis in {'z', 'Z'}:
            rot_axis = np.array([[cos(theta), -sin(theta), 0],
                                 [sin(theta), cos(theta), 0],
                                 [0, 0, 1]])
        else:
            raise ValueError('Unknown axis: ' + axis)
        rot = rot.dot(rot_axis)

    # apply rotation to row vectors
    if len(rotations) > 0:
        xyz = xyz.dot(rot.T)

    # translate back to origin
    if np.any(center):
        xyz += center

    return xyz


def modify_blob_shape(net_path, tops, blob_size):
    """
    TODO: make this more generic
    Modify .prototxt network specs
    :return: path to a temporary network definition with modified blob size
    """
    from caffe.proto import caffe_pb2
    import google.protobuf.text_format as txtf
    import tempfile

    net = caffe_pb2.NetParameter()

    with open(net_path) as f:
        txtf.Merge(f.read(), net)

    for top in tops:
        for i in blob_size:
            net.layer[[l.name for l in net.layer].index(top)].input_param.shape[0].dim[i] = blob_size[i]

    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(net))
    f.close()
    return f.name


class TimedBlock:
    """
    Context manager that times the execution of a block of code.
    """
    def __init__(self, msg='', verbose=False):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        if self.verbose and self.msg:
            print('{} ...'.format(self.msg), end='', flush=True)
        self.tic = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        toc = time.time()
        if self.verbose and self.msg:
            print(' done! ({:.2f} secs)'.format(toc - self.tic))


def seg_scores(pred_list, gt_list, nclasses=-1):

    eps = 0.0001

    labels = np.unique(np.concatenate([np.unique(gt) for gt in gt_list]))
    if nclasses != -1:
        assert len(labels) == nclasses

    acc, avg_class_acc, avg_class_iou = [], [], []
    for pred, gt in zip(pred_list, gt_list):
        pred, gt = np.array(pred), np.array(gt)
        assert np.all([(v in labels) for v in np.unique(pred)])

        tp, fp, fn = [], [], []
        for l in labels:
            tp.append(sum((gt == l) * (pred == l)))
            fp.append(sum((gt != l) * (pred == l)))
            fn.append(sum((gt == l) * (pred != l)))

        class_acc = [(tp[i] + eps) / (tp[i] + fn[i] + eps) for i in range(len(labels))]
        class_iou = [(tp[i] + eps) / (tp[i] + fn[i] + fp[i] + eps) for i in range(len(labels))]

        acc.append(sum(gt == pred) / len(gt))
        avg_class_acc.append(sum(class_acc) / len(labels))
        avg_class_iou.append(sum(class_iou) / len(labels))

    return acc, avg_class_acc, avg_class_iou

