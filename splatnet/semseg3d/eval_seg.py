"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
import splatnet.configs
from splatnet.utils import TimedBlock


def get_label(ply_path, column=3, from_rgb=True, cmap=None):
    with open(ply_path) as f:
        header_size = np.where([l.strip() == 'end_header' for l in f.readlines(1000)])[0][0] + 1
    ply_data = np.loadtxt(ply_path, skiprows=header_size)
    if from_rgb:
        return [np.where(np.prod(v == cmap, axis=1))[0][0] for v in ply_data[:, column:column+3]]
    else:
        return ply_data[:, column] - 1


def compute_scores(pred, gt):
    pred, gt = np.array(pred), np.array(gt)
    scores = dict()
    labels = np.unique(gt)
    assert np.all([(v in labels) for v in np.unique(pred)])

    TPs, FPs, FNs, Total = [], [], [], []
    for l in labels:
        TPs.append(sum((gt == l) * (pred == l)))
        FPs.append(sum((gt != l) * (pred == l)))
        FNs.append(sum((gt == l) * (pred != l)))
        Total.append(sum(gt == l))

    scores['accuracy'] = sum(gt == pred) / len(gt)
    scores['confusion'] = confusion_matrix(gt, pred)
    scores['class_accuracy'] = [TPs[i] / (TPs[i] + FNs[i]) for i in range(len(labels))]
    scores['avg_class_accuracy'] = sum(scores['class_accuracy']) / len(labels)
    scores['class_iou'] = [TPs[i] / (TPs[i] + FNs[i] + FPs[i]) for i in range(len(labels))]
    scores['avg_class_iou'] = sum(scores['class_iou']) / len(labels)
    scores['num_points'] = Total

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute evaluation metrics for segmentation results',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('pred', type=str, help='path to predictions')
    parser.add_argument('gt', type=str, help='path to ground-truth')
    parser.add_argument('--dataset', default='facade', choices=('facade', 'stanford3d'),
                        help='specify a dataset for the evaluation')
    parser.add_argument('--pred_rgb', action='store_true', help='turn this on if pred is encoded with rgb values')
    parser.add_argument('--gt_rgb', action='store_true', help='turn this on if gt is encoded with rgb values')
    parser.add_argument('--pred_column', type=int, default=4, help='(starting) column of label in pred (1-index)')
    parser.add_argument('--gt_column', type=int, default=4, help='(starting) column of label in gt (1-index)')
    parser.add_argument('--log', type=str, default=None, help='redirect output to log if specified')
    parser.add_argument('-q', '--quiet', action='store_true', help='silent intermediate information')
    args = parser.parse_args()

    if args.dataset == 'facade':
        classes = splatnet.configs.FACADE_CATEGORIES
        cmap = splatnet.configs.FACADE_CMAP

        with TimedBlock('Loading predictions from {}'.format(args.pred), not args.quiet):
            pred = get_label(args.pred, args.pred_column - 1, args.pred_rgb, cmap=cmap)

        with TimedBlock('Loading ground-truth from {}'.format(args.gt), not args.quiet):
            gt = get_label(args.gt, args.gt_column - 1, args.gt_rgb, cmap=cmap)

    elif args.dataset == 'stanford3d':
        raise NotImplementedError()
    else:
        raise ValueError('Dataset {} is not supported'.format(args.dataset))

    with TimedBlock('Computing scores', not args.quiet):
        scores = compute_scores(pred, gt)

    if not args.quiet:
        print('Evaluation done!')

    if args.log:
        sys.stdout = open(args.log, 'a')

    print('-------------------- Summary --------------------')
    print('   Overall accuracy: {:.4f}'.format(scores['accuracy']))
    print('Avg. class accuracy: {:.4f}'.format(scores['avg_class_accuracy']))
    print('                IoU: {:.4f}'.format(scores['avg_class_iou']))
    print('-------------------- Breakdown --------------------')
    print('  class      count(ratio) accuracy   IoU')
    total_points = sum(scores['num_points'])
    for i in range(len(classes)):
        print('{:10} {:7d}({:4.1f}%) {:.4f}   {:.4f}'.format(classes[i], scores['num_points'][i],
                                                             100 * scores['num_points'][i] / total_points,
                                                             scores['class_accuracy'][i], scores['class_iou'][i]))
    print('-------------------- Confusion --------------------')
    print('        {}'.format(' '.join(['{:>7}'.format(v) for v in classes])))
    for i, c in enumerate(classes):
        print('{:7} {}'.format(c, ' '.join(['{:7d}'.format(v) for v in scores['confusion'][i]])))

