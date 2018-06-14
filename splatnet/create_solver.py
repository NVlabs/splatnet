"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import tempfile
from caffe.proto import caffe_pb2


def get_prototxt(solver_proto, save_path=None):
    if save_path:
        f = open(save_path, mode='w+')
    else:
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(str(solver_proto))
    f.close()

    return f.name


def standard_solver(train_net,
                    test_net,
                    prefix,
                    solver_type='SGD',
                    weight_decay=0.001,
                    base_lr=0.01,
                    gamma=0.1,
                    stepsize=100,
                    test_iter=100,
                    test_interval=1000,
                    max_iter=1e5,
                    iter_size=1,
                    snapshot=1000,
                    display=1,
                    random_seed=0,
                    debug_info=False,
                    create_prototxt=True,
                    save_path=None):

    solver = caffe_pb2.SolverParameter()
    solver.train_net = train_net
    solver.test_net.extend([test_net])

    solver.test_iter.extend([test_iter])
    solver.test_interval = test_interval

    solver.base_lr = base_lr
    solver.lr_policy = 'step'  # "fixed"
    solver.gamma = gamma
    solver.stepsize = stepsize

    solver.display = display
    solver.max_iter = max_iter
    solver.iter_size = iter_size
    solver.snapshot = snapshot
    solver.snapshot_prefix = prefix
    solver.random_seed = random_seed

    solver.solver_mode = caffe_pb2.SolverParameter.GPU
    if solver_type is 'SGD':
        solver.solver_type = caffe_pb2.SolverParameter.SGD
    elif solver_type is 'ADAM':
        solver.solver_type = caffe_pb2.SolverParameter.ADAM
    solver.momentum = 0.9
    solver.momentum2 = 0.999

    solver.weight_decay = weight_decay

    solver.debug_info = debug_info

    if create_prototxt:
        solver = get_prototxt(solver, save_path)

    return solver

