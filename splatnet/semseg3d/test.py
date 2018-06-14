"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import glob
import argparse
import time
import numpy as np
import caffe
import splatnet.configs
from splatnet.utils import modify_blob_shape
from splatnet import plot_log
import splatnet.configs


EVAL_SCRIPT_PATH = os.path.join(splatnet.configs.ROOT_DIR, 'splatnet', 'semseg3d', 'eval_seg.py')


def extract_feat_scene(network_path, weights_path, feed, out_names, batch_size=1, sample_size=-1):
    net = caffe.Net(network_path, weights_path, caffe.TEST)
    net_bs, _, _, net_ss = net.blobs[list(feed.keys())[0]].data.shape

    npt = list(feed.values())[0].shape[-1]
    if sample_size == -1:
        sample_size = npt
    elif sample_size == 0:
        sample_size = net_ss
    else:
        assert sample_size * batch_size <= npt

    if net_bs != batch_size or net_ss != sample_size:
        network_path = modify_blob_shape(network_path, feed.keys(), {0: batch_size, 3: sample_size})
        net = caffe.Net(network_path, weights_path, caffe.TEST)

    if type(out_names) == str:
        out_names = (out_names,)
        single_target = True
    else:
        single_target = False

    outs = {v: [] for v in out_names}
    pts_per_batch = batch_size * sample_size
    for b in range(int(np.ceil(npt / pts_per_batch))):
        b_end = min(pts_per_batch * (b + 1), npt)
        b_slice = slice(b_end - pts_per_batch, b_end)
        bs = min(pts_per_batch, npt - pts_per_batch * b)

        for in_key in feed:
            net.blobs[in_key].data[...] \
                = feed[in_key][:, :, :, b_slice].reshape(-1, batch_size, sample_size, 1).transpose(1, 0, 3, 2)
        net.forward()
        for out_key in out_names:
            out_sz = net.blobs[out_key].data.shape
            out = net.blobs[out_key].data.transpose(1, 2, 0, 3).reshape(1, out_sz[1], out_sz[2], -1)[:, :, :, -bs:]
            outs[out_key].append(out.copy())

    result = {v: np.concatenate(outs[v], axis=3) for v in out_names}
    if single_target:
        result = result[out_names[0]]

    return result


def semseg_test(dataset, network, weights, input_dims='nx_ny_nz_r_g_b_h', sample_size=-1,
                dataset_params=None, save_dir='', save_prefix='', use_cpu=False):
    """
    Testing trained semantic segmentation network
    :param dataset: choices: 'facade', 'stanford3d'
    :param network: path to a .prototxt file
    :param weights: path to a .caffemodel file
    :param input_dims: feat dims and scales
    :param sample_size: -1 -- use all points in a single sample, 0 -- use the size in network
    :param dataset_params: a dict with optional dataset parameters
    :param save_dir: default ''
    :param save_prefix: default ''
    :param use_cpu: default False
    :return:
    """

    if use_cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)

    # dataset specific: data, xyz, cmap
    if dataset == 'facade':
        from splatnet.dataset import dataset_facade
        dataset_params_new = {} if not dataset_params else dataset_params
        dataset_params = dict(subset='test', val_ratio=0.0)  # default values
        dataset_params.update(dataset_params_new)
        for v in {'val_ratio'}:
            if v in dataset_params:
                dataset_params[v] = float(dataset_params[v])
        data, xyz, norms = dataset_facade.points(dims=input_dims+',x_y_z,nx_ny_nz', **dataset_params)
        cmap = splatnet.configs.FACADE_CMAP
    elif dataset == 'stanford3d':
        norms = None
        pass  # TODO set cmap, data, xyz
    else:
        raise ValueError('Unsupported dataset: {}'.format(dataset))

    tic = time.time()
    prob = extract_feat_scene(network, weights,
                              feed=dict(data=data.transpose().reshape(1, -1, 1, len(data))),
                              out_names='prob',
                              sample_size=sample_size)
    elapsed = time.time() - tic

    pred = prob.argmax(axis=1).squeeze()

    if norms is None:
        out = np.array([np.concatenate((x, cmap[int(c)]), axis=0) for (x, c) in zip(xyz, pred)])
        header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar diffuse_red
property uchar diffuse_green
property uchar diffuse_blue
end_header'''.format(len(data))
        fmt = '%.6f %.6f %.6f %d %d %d'
    else:
        out = np.array([np.concatenate((x, n, cmap[int(c)]), axis=0) for (x, n, c) in zip(xyz, norms, pred)])
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
end_header'''.format(len(data))
        fmt = '%.6f %.6f %.6f %.6f %.6f %.6f %d %d %d'

    save_path = os.path.join(save_dir, '{}pred_{}.ply'.format(save_prefix, dataset_params['subset']))
    np.savetxt(save_path, out, fmt=fmt, header=header, comments='')

    return save_path, elapsed, len(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    group = parser.add_argument_group('testing options')
    group.add_argument('dataset')
    group.add_argument('--dataset_params', nargs='+', help='dataset-specific parameters (key value pairs)')
    group.add_argument('--input', default='nx_ny_nz_r_g_b_h', help='features to use as input')
    group.add_argument('--cpu', action='store_true', help='use cpu')
    group.add_argument('--exp_dir', default=None, type=str, help='together with exp_prefix, set defaults to args below')
    group.add_argument('--exp_prefix', default='', type=str, help='together with exp_dir, set defaults to args below')
    group.add_argument('--network', default=None, type=str, help='a .prototxt file')
    group.add_argument('--weights', default=None, type=str, help='a .caffemodel file')
    group.add_argument('--sample_size', default=-1, type=int, help='testing sample size')
    group.add_argument('--log', default=None, type=str, help='a .log file with training logs')
    group.add_argument('--log_eval', default=None, type=str, help='path to write evaluation logs')
    group.add_argument('--save_dir', default=None, type=str, help='together with save_prefix, a place for predictions')
    group.add_argument('--save_prefix', default=None, type=str, help='together with save_dir, a place for predictions')

    group = parser.add_argument_group('evaluation options')
    group.add_argument('--gt', default=None, type=str, help='path to ground-truth')
    group.add_argument('--gt_rgb', action='store_true', help='turn this on if gt is encoded with rgb values')
    group.add_argument('--gt_column', type=int, default=11, help='(starting) column of label in gt (1-index)')

    parser = argparse.ArgumentParser(description='Testing trained semantic segmentation network',
                                     parents=[parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    network, weights = args.network, args.weights
    save_dir, save_prefix = args.save_dir, args.save_prefix
    log_train = args.log
    log_eval = args.log_eval

    if args.exp_dir is not None:
        if args.exp_prefix:
            exp_prefix = args.exp_prefix + '_'
            snapshot_prefix = exp_prefix
        else:
            exp_prefix = ''
            snapshot_prefix = 'snapshot_'
        if network is None:
            network = os.path.join(args.exp_dir, exp_prefix + 'net_deploy.prototxt')
        if weights is None:
            last_iter = max([int(os.path.split(v)[1].split('_')[-1][:-11]) for v in
                             glob.glob(os.path.join(args.exp_dir, snapshot_prefix + 'iter_*.caffemodel'))])
            weights = os.path.join(args.exp_dir, '{}iter_{}.caffemodel'.format(snapshot_prefix, last_iter))
        if log_train is None:
            log_train = os.path.join(args.exp_dir, exp_prefix + 'train.log')
            if not os.path.exists(log_train):
                log_train = ''
        if save_dir is None:
            save_dir = args.exp_dir
        if save_prefix is None:
            save_prefix = exp_prefix
        if log_eval is None:
            log_eval = os.path.join(args.exp_dir, exp_prefix + 'test.log')

    if not args.dataset_params:
        args.dataset_params = {}
    else:
        args.dataset_params = dict(zip(args.dataset_params[::2], args.dataset_params[1::2]))

    pred_path, elapsed, num_pts = semseg_test(args.dataset, network, weights, args.input, args.sample_size,
                                              args.dataset_params, save_dir, save_prefix, args.cpu)

    if log_eval:
        with open(log_eval, 'a') as f:
            f.write('Predictions saved to {}.\n'.format(pred_path))
            f.write('{} points evaluated in {:.2f} secs.\n'.format(num_pts, elapsed))

    if args.gt is not None:
        import subprocess
        subprocess.run('python {} --dataset {} {}'
                       '{} --pred_rgb --pred_column 7 '
                       '{} {}--gt_column {}'.format(EVAL_SCRIPT_PATH,
                                                    args.dataset,
                                                    '--log {} '.format(log_eval) if log_eval else '',
                                                    os.path.abspath(pred_path),
                                                    os.path.abspath(args.gt),
                                                    '--gt_rgb ' if args.gt_rgb else '',
                                                    args.gt_column).split(' '))

    if log_train:
        plot_log.parse_and_plot(log_train)

