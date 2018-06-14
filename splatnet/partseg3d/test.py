"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import argparse
import glob
import time
import numpy as np
import caffe

import splatnet.configs
from splatnet import plot_log
from splatnet.utils import modify_blob_shape, seg_scores


def extract_feat_shapes(network_path, weights_path, feed, out_names, batch_size=64, sample_size=3000):
    if sample_size == -1:
        assert batch_size == 1

    ori_sample_sizes = [len(sample) for sample in list(feed.values())[0]]
    nsamples = len(ori_sample_sizes)

    batch_size = nsamples if nsamples < batch_size else batch_size

    net = caffe.Net(network_path, weights_path, caffe.TEST)
    net_bs, _, _, net_ss = net.blobs[list(feed.keys())[0]].data.shape

    if sample_size != -1 and (net_bs != batch_size or net_ss != sample_size):
        network_path = modify_blob_shape(network_path, feed.keys(), {0: batch_size, 3: sample_size})
        net = caffe.Net(network_path, weights_path, caffe.TEST)

    # pad samples to fixed length
    if sample_size != -1:
        for i in range(nsamples):
            k = ori_sample_sizes[i]
            idx = np.concatenate((np.tile(np.arange(k), (sample_size // k, )),
                                  np.random.permutation(k)[:(sample_size % k)]), axis=0)
            feed = feed.copy()
            for in_key in feed:
                feed[in_key][i] = feed[in_key][i][idx]

    outs = {v: [] for v in out_names}
    for b in range(int(np.ceil(nsamples / batch_size))):
        b_end = min(batch_size * (b + 1), nsamples)
        b_slice = slice(b_end - batch_size, b_end)
        bs = min(batch_size, nsamples - batch_size * b)

        if sample_size == -1:
            ss = list(feed.values())[0][b_end - 1].shape[0]
            if net_ss != ss or net_bs != 1:
                network_path = modify_blob_shape(network_path, feed.keys(), {0: 1, 3: ss})
                net = caffe.Net(network_path, weights_path, caffe.TEST)
                net_ss, net_bs = ss, 1
        else:
            ss = sample_size

        for in_key in feed:
            net.blobs[in_key].data[...] \
                = np.concatenate(feed[in_key][b_slice], axis=0).reshape(batch_size, ss, -1, 1).transpose(0, 2, 3, 1)
        net.forward()
        for out_key in out_names:
            out = net.blobs[out_key].data.transpose(0, 3, 1, 2)[-bs:].reshape(bs * ss, -1)
            outs[out_key].append(out.copy())

    if sample_size != -1:
        outs = {v: np.array_split(np.concatenate(outs[v], axis=0), nsamples) for v in out_names}
        # remove padding
        for v in out_names:
            outs[v] = [sample[:k] for (sample, k) in zip(outs[v], ori_sample_sizes)]

    return outs


def partseg_test(dataset, network, weights, input_dims='x_y_z', sample_size=3000, batch_size=64,
                 category='airplane', dataset_params=None,
                 save_dir='', skip_ply=False, use_cpu=False):
    """
    Testing trained segmentation network
    :param dataset: choices: 'shapenet'
    :param network: path to a .prototxt file
    :param weights: path to a .caffemodel file
    :param input_dims: network input featurse
    :param sample_size:
    :param batch_size:
    :param dataset_params: a dict with optional dataset parameters
    :param save_dir: default ''
    :param skip_ply: default False
    :param use_cpu: default False
    :return:
    """

    if use_cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)

    # dataset specific: data, xyz_norm_list, cmap, num_part_categories, category_offset
    if dataset == 'shapenet':
        import splatnet.dataset.dataset_shapenet as shapenet
        dataset_params_new = {} if not dataset_params else dataset_params
        dataset_params = dict(subset='test')  # default values
        dataset_params.update(dataset_params_new)

        data, _, part_label, names = shapenet.points_single_category(dims=input_dims, category=category, **dataset_params)
        xyz_norm_list, _, _, _ = shapenet.points_single_category(dims='x_y_z_nx_ny_nz', category=category, **dataset_params)
        cmap = splatnet.configs.SN_CMAP
        if category.startswith('0'):
            category_id = splatnet.configs.SN_CATEGORIES.index(category)
        else:
            category_id = splatnet.configs.SN_CATEGORY_NAMES.index(category)
        num_part_categories = splatnet.configs.SN_NUM_PART_CATEGORIES[category_id]
        category_offset = sum(splatnet.configs.SN_NUM_PART_CATEGORIES[:category_id])
    else:
        raise ValueError('Unsupported dataset: {}'.format(dataset))

    tic = time.time()
    probs = extract_feat_shapes(network, weights,
                                feed=dict(data=data),
                                out_names=('prob',),
                                sample_size=sample_size, batch_size=batch_size)['prob']
    elapsed = time.time() - tic

    if probs[0].shape[1] > num_part_categories:
        preds = [np.argmax(prob[:, category_offset:category_offset+num_part_categories], axis=1) for prob in probs]
    else:
        preds = [np.argmax(prob, axis=1) for prob in probs]

    acc, avgacc, avgiou = seg_scores(preds, part_label, nclasses=num_part_categories)

    if not skip_ply:
        os.makedirs(save_dir, exist_ok=True)
        for xyz_norm, pred, name in zip(xyz_norm_list, preds, names):
            out = np.array([np.concatenate((x, cmap[int(c)]), axis=0) for (x, c) in zip(xyz_norm, pred)])
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
end_header'''.format(len(pred))
            fmt = '%.6f %.6f %.6f %.6f %.6f %.6f %d %d %d'
            save_path = os.path.join(save_dir, '{}.ply'.format(name))
            np.savetxt(save_path, out, fmt=fmt, header=header, comments='')

    return acc, avgacc, avgiou, elapsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing trained part segmentation network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset')
    parser.add_argument('--dataset_params', nargs='+', help='dataset-specific parameters (key value pairs)')
    parser.add_argument('--categories', nargs='+', help='pick some categories, otherwise evaluate on all')
    parser.add_argument('--single_model', action='store_true', help='if True, use a single model across all categories')
    parser.add_argument('--input', default='x_y_z', help='features to use as input')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--skip_ply', action='store_true', help='if True, do not output .ply prediction results')
    parser.add_argument('--sample_size', default=3000, type=int, help='testing sample size')
    parser.add_argument('--batch_size', default=64, type=int, help='testing sample size')
    parser.add_argument('--snapshot', default=None, type=str, help='snapshot rule - last|best_acc|best_loss|ITER')
    parser.add_argument('--exp_dir', default=None, type=str, help='if present, assigns values to options below')
    parser.add_argument('--network', default=None, type=str, help='a .prototxt file')
    parser.add_argument('--weights', default=None, type=str, help='a .caffemodel file; overwrites \'--snapshot\'')
    parser.add_argument('--log', default=None, type=str, help='a .log file with training logs')
    parser.add_argument('--log_eval', default=None, type=str, help='path to write evaluation logs')
    parser.add_argument('--save_dir', default=None, type=str)

    args = parser.parse_args()
    
    if not args.categories:
        if args.dataset == 'shapenet':
            args.categories = splatnet.configs.SN_CATEGORY_NAMES
        else:
            raise ValueError('Unsupported dataset: {}'.format(args.dataset))

    dataset_params = dict(zip(args.dataset_params[::2], args.dataset_params[1::2])) if args.dataset_params else {}
    exp_dir = args.exp_dir
    save_dir = args.save_dir if args.save_dir else os.path.join(exp_dir, 'pred')
    log_eval = args.log_eval if args.log_eval else os.path.join(exp_dir, 'test.log')

    with open(log_eval, 'a') as f:
        f.write('category | #samples | acc | class-acc | iou | snapshot | predictions\n')
        f.write('(batch_size={}, sample_size={})\n'.format(args.batch_size, args.sample_size))

    tic = time.time()
    ious = []

    for category_cnt, category in enumerate(args.categories):
        if category_cnt == 0 or not args.single_model:
            if args.network:
                network = args.network
            else:
                network_prefix = 'net' if args.single_model else '{}_net'.format(category)
                network = os.path.join(exp_dir, '{}_deploy.prototxt'.format(network_prefix))
            if args.log:
                log = args.log
            else:
                log = os.path.join(exp_dir, 'train.log' if args.single_model else category+'.log')

            if args.weights:
                assert args.weights.endswith('.caffemodel')
                weights = args.weights
            else:
                snapshot_prefix = 'snapshot' if args.single_model else category
                if not args.snapshot:
                    weights = os.path.join(exp_dir, '{}.caffemodel'.format(snapshot_prefix))
                else:
                    iters_avail = [int(os.path.split(v)[1].split('_')[-1][:-11]) for v in
                                   glob.glob(os.path.join(exp_dir, snapshot_prefix + '_iter_*.caffemodel'))]
                    if args.snapshot == 'last':
                        snap_iter = max(iters_avail)
                    elif args.snapshot == 'best_acc':
                        _, stats_val = plot_log.parse_and_plot(log, skip_train=0, skip_test=0)
                        iter_acc = dict(zip(stats_val[:, 0], stats_val[:, -2]))
                        snap_iter = iters_avail[np.argmax([iter_acc[v] for v in iters_avail])]
                    elif args.snapshot == 'best_loss':
                        _, stats_val = plot_log.parse_and_plot(log, skip_train=0, skip_test=0)
                        iter_loss = dict(zip(stats_val[:, 0], stats_val[:, -1]))
                        snap_iter = iters_avail[np.argmin([iter_loss[v] for v in iters_avail])]
                    elif args.snapshot.isdigit():
                        snap_iter = int(args.snapshot)
                    else:
                        raise ValueError('Unknown snapshot rule: {}'.format(args.snapshot))
                    weights = os.path.join(exp_dir, '{}_iter_{}.caffemodel'.format(snapshot_prefix, snap_iter))

        acc, avgacc, avgiou, _ = partseg_test(args.dataset, network, weights, args.input,
                                              args.sample_size, args.batch_size, category, dataset_params,
                                              os.path.join(save_dir, category), args.skip_ply, args.cpu)
        ious.append((np.mean(avgiou), len(avgiou)))

        with open(log_eval, 'a') as f:
            f.write('{} {} {} {} {} {} {}\n'.format(category, len(avgiou),
                                                    np.mean(acc), np.mean(avgacc), np.mean(avgiou),
                                                    os.path.basename(weights),
                                                    '-' if args.skip_ply else os.path.join(save_dir, category)))

    elapsed = time.time() - tic

    with open(log_eval, 'a') as f:
        f.write('\nOverall weighted average mean IOU: {}\n'.format(
            sum([iou * cnt for (iou, cnt) in ious]) / sum([cnt for (_, cnt) in ious])))
        f.write('Time elapsed: {} seconds\n\n'.format(elapsed))

