"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import caffe


class GlobalPooling(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        n, c, h, w = bottom[0].data.shape
        self.max_loc = bottom[0].data.reshape(n, c, h*w).argmax(axis=2)
        top[0].data[...] = bottom[0].data.max(axis=(2, 3), keepdims=True)

    def backward(self, top, propagate_down, bottom):
        n, c, h, w = top[0].diff.shape
        nn, cc = np.ix_(np.arange(n), np.arange(c))
        bottom[0].diff[...] = 0
        bottom[0].diff.reshape(n, c, -1)[nn, cc, self.max_loc] = top[0].diff.sum(axis=(2, 3))


class ProbRenorm(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        clipped = bottom[0].data * bottom[1].data
        self.sc = 1.0 / (np.sum(clipped, axis=1, keepdims=True) + 1e-10)
        top[0].data[...] = clipped * self.sc

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff * bottom[1].data * self.sc


class Permute(caffe.Layer):
    def setup(self, bottom, top):
        self.dims = [int(v) for v in self.param_str.split('_')]
        self.dims_ind = list(np.argsort(self.dims))

    def reshape(self, bottom, top):
        old_shape = bottom[0].data.shape
        new_shape = [old_shape[d] for d in self.dims]
        top[0].reshape(*new_shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data.transpose(*self.dims)

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff.transpose(*self.dims_ind)


class LossHelper(caffe.Layer):
    def setup(self, bottom, top):
        self.old_shape = bottom[0].data.shape

    def reshape(self, bottom, top):
        new_shape = (self.old_shape[0] * self.old_shape[3], self.old_shape[1], 1, 1)
        top[0].reshape(*new_shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data.transpose(0, 3, 1, 2).reshape(*top[0].data.shape)

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff.reshape(self.old_shape[0], self.old_shape[3], self.old_shape[1], 1
                                                  ).transpose(0, 2, 3, 1)


class LogLoss(caffe.Layer):
    def setup(self, bottom, top):
        self.n, self.c, _, self.s = bottom[0].data.shape
        self.inds = np.ix_(np.arange(self.n), np.arange(self.c), np.arange(1), np.arange(self.s))

    def reshape(self, bottom, top):
        top[0].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        self.valid = bottom[0].data[self.inds[0], bottom[1].data.astype(int), self.inds[2], self.inds[3]]
        top[0].data[:] = -np.mean(np.log(self.valid + 1e-10))

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[:] = 0.0
        bottom[0].diff[self.inds[0], bottom[1].data.astype(int), self.inds[2], self.inds[3]] = \
            -1.0 / ((self.valid + 1e-10) * (self.n * self.s))


class PickAndScale(caffe.Layer):
    def setup(self, bottom, top):
        self.nch_out = len(self.param_str.split('_'))
        self.dims = []
        for f in self.param_str.split('_'):
            if f.find('*') >= 0:
                self.dims.append((int(f[:f.find('*')]), float(f[f.find('*') + 1:])))
            elif f.find('/') >= 0:
                self.dims.append((int(f[:f.find('/')]), 1.0 / float(f[f.find('/') + 1:])))
            else:
                self.dims.append((int(f), 1.0))

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], self.nch_out, bottom[0].data.shape[2], bottom[0].data.shape[3])

    def forward(self, bottom, top):
        for i, (j, s) in enumerate(self.dims):
            top[0].data[:, i, :, :] = bottom[0].data[:, j, :, :] * s

    def backward(self, top, propagate_down, bottom):
        pass  # TODO NOT_YET_IMPLEMENTED

