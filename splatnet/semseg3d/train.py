"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import argparse
import caffe
import splatnet.semseg3d.models as models
from splatnet import create_solver
import splatnet.configs


def semseg_train(network, exp_dir, exp_prefix, args):

    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)

    if exp_prefix:
        snapshot_prefix = os.path.join(exp_dir, exp_prefix)
        exp_prefix = exp_prefix + '_'
    else:
        snapshot_prefix = os.path.join(exp_dir, 'snapshot')
        exp_prefix = ''

    if network == 'seq':
        batch_norm = True
        conv_weight_filler = 'xavier'
        network = models.semseg_seq(arch_str=args.arch,
                                    skip_str=args.skips,
                                    dataset=args.dataset,
                                    dataset_params=args.dataset_params,
                                    feat_dims_str=args.feat,
                                    lattice_dims_str=args.lattice,
                                    sample_size=args.sample_size,
                                    batch_size=args.batch_size,
                                    batchnorm=batch_norm,
                                    conv_weight_filler=conv_weight_filler,
                                    save_path=os.path.join(exp_dir, exp_prefix + 'net.prototxt'))

        models.semseg_seq(deploy=True,
                          arch_str=args.arch,
                          skip_str=args.skips,
                          dataset=args.dataset,
                          dataset_params=args.dataset_params,
                          feat_dims_str=args.feat,
                          lattice_dims_str=args.lattice,
                          sample_size=args.sample_size,
                          batchnorm=batch_norm,
                          save_path=os.path.join(exp_dir, exp_prefix + 'net_deploy.prototxt'))
    else:
        assert network.endswith('.prototxt'), 'Please provide a valid prototxt file'
        print('Using network defined at {}'.format(network))

    random_seed = 0
    debug_info = False
    solver = create_solver.standard_solver(network,
                                           network,
                                           snapshot_prefix,
                                           base_lr=args.base_lr,
                                           gamma=args.lr_decay,
                                           stepsize=args.stepsize,
                                           test_iter=args.test_iter,
                                           test_interval=args.test_interval,
                                           max_iter=args.num_iter,
                                           snapshot=args.snapshot_interval,
                                           solver_type=args.solver_type,
                                           weight_decay=args.weight_decay,
                                           iter_size=args.iter_size,
                                           debug_info=debug_info,
                                           random_seed=random_seed,
                                           save_path=os.path.join(exp_dir, exp_prefix + 'solver.prototxt'))
    solver = caffe.get_solver(solver)

    if args.init_model:
        if args.init_model.endswith('.solverstate'):
            solver.restore(args.init_model)
        elif args.init_model.endswith('.caffemodel'):
            solver.net.copy_from(args.init_model)
        else:
            raise ValueError('Invalid file: {}'.format(args.init_model))

    solver.solve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    group = parser.add_argument_group('network options')
    group.add_argument('--network', default='seq', help='network type (\'seq\') or path to a .prototxt file')
    group.add_argument('--arch', default='64_128_256_256', help='network architecture')
    group.add_argument('--dataset', default='facade', choices=('facade', 'stanford3d', 'stanford3dh5', 'shapenet'), help='dataset')
    group.add_argument('--dataset_params', nargs='+', help='dataset-specific parameters (key value pairs)')
    group.add_argument('--category', type=str, default=None, help='for shapenet tasks')
    group.add_argument('--skips', nargs='+', help='skip connections')
    group.add_argument('--sample_size', type=int, default=10000, help='number of points in a sample')
    group.add_argument('--batch_size', type=int, default=10, help='number of samples in a batch')
    group.add_argument('--feat', default='nx_ny_nz_r_g_b_h', help='features to use as input')
    group.add_argument('--lattice', nargs='+', help='bnn lattice features and scales')

    group = parser.add_argument_group('solver options')
    group.add_argument('--base_lr', type=float, default=0.01, help='starting learning rate')
    group.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay rate')
    group.add_argument('--stepsize', type=int, default=10000, help='learning rate decay interval')
    group.add_argument('--num_iter', type=int, default=10000, help='number of iterations to train')
    group.add_argument('--iter_size', type=int, default=1, help='number of mini-batches per iteration')
    group.add_argument('--test_iter', type=int, default=10, help='number of iterations to use at each testing phase')
    group.add_argument('--test_interval', type=int, default=100, help='test every such iterations')
    group.add_argument('--snapshot_interval', type=int, default=2000, help='snapshot every such iterations')
    group.add_argument('--solver_type', default='ADAM', help='optimizer type')
    group.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')

    parser = argparse.ArgumentParser(description='Semantic segmentation training',
                                     parents=[parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('exp_dir')
    parser.add_argument('--exp_prefix', default='', help='optional prefix for the experiment')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--init_model', default=None, help='a .caffemodel/.solverstate file to start/resume training')

    args = parser.parse_args()

    if not args.dataset_params:
        args.dataset_params = {}
    else:
        args.dataset_params = dict(zip(args.dataset_params[::2], args.dataset_params[1::2]))

    os.makedirs(args.exp_dir, exist_ok=True)

    semseg_train(args.network, args.exp_dir, args.exp_prefix, args)

