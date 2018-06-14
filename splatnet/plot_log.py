"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt


def parse_and_plot(path, subplot_size=5, n_col=3, skip_train=100, skip_test=10,
                   caffe_root=None, show_wo_save=False):

    if not caffe_root:
        import caffe
        caffe_root = os.path.join(os.path.dirname(caffe.__file__), '..', '..')

    parser_path = os.path.join(caffe_root, 'tools/extra/parse_log.py')
    subprocess.run('python2 {} {} {}'.format(parser_path, path, os.path.dirname(path)).split(' '))

    with open(path + '.train') as f:
        l = f.readline()
    labels_train = l.strip().split(',')[2:]
    with open(path + '.test') as f:
        l = f.readline()
    labels_test = l.strip().split(',')[2:]

    stats_train = np.loadtxt(path + '.train', delimiter=',', skiprows=1)
    stats_test = np.loadtxt(path + '.test', delimiter=',', skiprows=1)

    n_row = (len(labels_train) - 1) // n_col + 1, (len(labels_test) - 1) // n_col + 1

    if not show_wo_save:
        plt.switch_backend('agg')

    plt.figure(figsize=(n_col * subplot_size, sum(n_row) * subplot_size))
    for i, l in enumerate(labels_train):
        plt.subplot(sum(n_row), n_col, i + 1)
        plt.plot(stats_train[skip_train:, 0], stats_train[skip_train:, i + 2])
        plt.title('train_' + l), plt.grid('on')
    for i, l in enumerate(labels_test):
        plt.subplot(sum(n_row), n_col, n_row[0] * n_col + i + 1)
        plt.plot(stats_test[skip_test:, 0], stats_test[skip_test:, i + 2])
        plt.title('test_' + l), plt.grid('on')

    if show_wo_save:
        plt.show()
    else:
        plt.savefig(path + '.png')

    return stats_train, stats_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse caffe training log and plot stats',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path')
    parser.add_argument('--subplot_size', default=5, type=int, help='width and height of each subplot')
    parser.add_argument('--num_column', default=3, type=int, help='number of columns')
    parser.add_argument('--skip_train', default=100, type=int, help='skip first such training iterations')
    parser.add_argument('--skip_test', default=10, type=int, help='skip first such testing iterations')
    parser.add_argument('--show_wo_save', action='store_true', help='show figure instead of saving it')
    parser.add_argument('--caffe_root', default=None, type=str, help='path to caffe installation directory')

    args = parser.parse_args()

    parse_and_plot(args.path, args.subplot_size, args.num_column, args.skip_train, args.skip_test,
                   args.caffe_root, args.show_wo_save)

