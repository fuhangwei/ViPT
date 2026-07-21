import os
from os.path import isdir, join

import numpy as np


def load_sequence_list(seq_home, seq_file=None, video=None):
    if video:
        return [video]
    if seq_file:
        with open(seq_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    seqs = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
    seqs.sort()
    return seqs


def load_rgbt_sequence(seq_home, seq_name, dataset):
    seq_path = join(seq_home, seq_name)
    if dataset == 'RGBT234':
        rgb_dir = join(seq_path, 'visible')
        tir_dir = join(seq_path, 'infrared')
        rgb_imgs = sorted(join(rgb_dir, p) for p in os.listdir(rgb_dir) if os.path.splitext(p)[1].lower() == '.jpg')
        tir_imgs = sorted(join(tir_dir, p) for p in os.listdir(tir_dir) if os.path.splitext(p)[1].lower() == '.jpg')
        rgb_gt = np.loadtxt(join(seq_path, 'visible.txt'), delimiter=',')
        tir_gt = np.loadtxt(join(seq_path, 'infrared.txt'), delimiter=',')
    elif dataset == 'LasHeR':
        rgb_dir = join(seq_path, 'visible')
        tir_dir = join(seq_path, 'infrared')
        rgb_imgs = sorted(join(rgb_dir, p) for p in os.listdir(rgb_dir) if p.endswith('.jpg'))
        tir_imgs = sorted(join(tir_dir, p) for p in os.listdir(tir_dir) if p.endswith('.jpg'))
        rgb_gt = np.loadtxt(join(seq_path, 'visible.txt'), delimiter=',')
        tir_gt = np.loadtxt(join(seq_path, 'infrared.txt'), delimiter=',')
    elif dataset == 'GTOT':
        rgb_dir = join(seq_path, 'v')
        tir_dir = join(seq_path, 'i')
        rgb_imgs = sorted(join(rgb_dir, p) for p in os.listdir(rgb_dir) if os.path.splitext(p)[1].lower() == '.png')
        tir_imgs = sorted(join(tir_dir, p) for p in os.listdir(tir_dir) if os.path.splitext(p)[1].lower() == '.png')
        rgb_gt = np.loadtxt(join(seq_path, 'groundTruth_v.txt'), delimiter=' ')
        tir_gt = np.loadtxt(join(seq_path, 'groundTruth_i.txt'), delimiter=' ')
        rgb_gt = _poly_to_xywh(rgb_gt)
        tir_gt = _poly_to_xywh(tir_gt)
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    rgb_gt = _ensure_2d(rgb_gt)
    tir_gt = _ensure_2d(tir_gt)
    return rgb_imgs, tir_imgs, rgb_gt, tir_gt


def _ensure_2d(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, :4]


def _poly_to_xywh(gt):
    gt = _ensure_2d(gt)
    if gt.shape[1] < 4:
        return gt
    x_min = np.min(gt[:, [0, 2]], axis=1)[:, None]
    y_min = np.min(gt[:, [1, 3]], axis=1)[:, None]
    x_max = np.max(gt[:, [0, 2]], axis=1)[:, None]
    y_max = np.max(gt[:, [1, 3]], axis=1)[:, None]
    return np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
