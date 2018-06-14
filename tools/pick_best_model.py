"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import sys
import glob
import shutil
import subprocess
import numpy as np


def parse_and_plot(path, skip_train=100, skip_test=10, caffe_root=None):

    if not caffe_root:
        import caffe
        caffe_root = os.path.join(os.path.dirname(caffe.__file__), '..', '..')

    parser_path = os.path.join(caffe_root, 'tools/extra/parse_log.py')
    subprocess.run('python2 {} {} {}'.format(parser_path, path, os.path.dirname(path)).split(' '))

    stats_train = np.loadtxt(path + '.train', delimiter=',', skiprows=1)
    stats_test = np.loadtxt(path + '.test', delimiter=',', skiprows=1)

    return stats_train, stats_test


def copy_best_model(exp_dir, categories, pick_rule='best_loss'):
    if os.path.exists(os.path.join(exp_dir, 'train.log')):
        raise NotImplementedError()

    for category in categories:
        log = os.path.join(exp_dir, category + '.log')
        snapshot_prefix = category
        iters_avail = [int(os.path.split(v)[1].split('_')[-1][:-11]) for v in
                       glob.glob(os.path.join(exp_dir, snapshot_prefix + '_iter_*.caffemodel'))]
        if pick_rule == 'last':
            snap_iter = max(iters_avail)
        elif pick_rule == 'best_acc':
            _, stats_val = parse_and_plot(log, skip_train=0, skip_test=0)
            iter_acc = dict(zip(stats_val[:, 0], stats_val[:, -2]))
            snap_iter = iters_avail[np.argmax([iter_acc[v] for v in iters_avail])]
        elif pick_rule == 'best_loss':
            _, stats_val = parse_and_plot(log, skip_train=0, skip_test=0)
            iter_loss = dict(zip(stats_val[:, 0], stats_val[:, -1]))
            snap_iter = iters_avail[np.argmin([iter_loss[v] for v in iters_avail])]
        elif pick_rule.isdigit():
            snap_iter = int(pick_rule)
        else:
            raise ValueError('Unknown snapshot rule: {}'.format(pick_rule))
        weights = os.path.join(exp_dir, '{}_iter_{}.caffemodel'.format(snapshot_prefix, snap_iter))
        target = os.path.join(exp_dir, '{}.caffemodel'.format(snapshot_prefix))
        print('copying {} to {} ...'.format(weights, target), end='', flush=True)
        shutil.copyfile(weights, target)
        print(' done!', flush=True)


SHAPE_CATEGORY_NAMES = ('laptop', 'car', 'skateboard', 'chair', 'earphone', 'motorbike', 'mug', 'airplane',
                        'pistol', 'lamp', 'bag', 'cap', 'rocket', 'guitar', 'table', 'knife')


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: {} '.format(os.path.basename(sys.argv[0])) + '<exp_dir> <pick_rule> [<category>]')
    else:
        categories = SHAPE_CATEGORY_NAMES if len(sys.argv) < 4 else (sys.argv[3],)
        copy_best_model(sys.argv[1], categories, sys.argv[2])

