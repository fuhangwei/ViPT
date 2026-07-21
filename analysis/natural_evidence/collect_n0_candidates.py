import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.natural_evidence.dataset_io import load_rgbt_sequence, load_sequence_list
from lib.test.tracker.vipt import ViPTTrack
import lib.test.parameter.vipt as vipt_params
from lib.train.dataset.depth_utils import get_x_frame


def parse_args():
    parser = argparse.ArgumentParser(description='Collect CUETrack N0 top-k candidates on natural RGB-T frames.')
    parser.add_argument('--dataset', required=True, choices=['RGBT234', 'LasHeR', 'GTOT'])
    parser.add_argument('--seq-home', required=True)
    parser.add_argument('--seq-file', default='')
    parser.add_argument('--video', default='')
    parser.add_argument('--script-name', default='vipt')
    parser.add_argument('--yaml-name', required=True)
    parser.add_argument('--epoch', default=60, type=int)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--max-frames', default=0, type=int)
    parser.add_argument('--branch-source', default='auto', choices=['auto', 'fusion-only', 'ablation-probe'])
    parser.add_argument('--allow-ablation-probes', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.script_name != 'vipt':
        raise ValueError('Only --script-name vipt is supported for N0 collection.')
    if args.branch_source == 'ablation-probe' and not args.allow_ablation_probes:
        raise ValueError('--branch-source ablation-probe requires --allow-ablation-probes.')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'candidates.jsonl'
    meta_path = out_dir / 'metadata.json'
    if args.skip_existing and out_path.exists():
        print(f'Skip existing {out_path}')
        return

    params = vipt_params.parameters(args.yaml_name, args.epoch)
    params.n0_collect = True
    params.n0_topk = args.topk
    params.n0_branch_source = args.branch_source

    seqs = load_sequence_list(args.seq_home, args.seq_file or None, args.video or None)
    metadata = {
        'dataset': args.dataset,
        'seq_home': args.seq_home,
        'seq_file': args.seq_file,
        'video': args.video,
        'yaml_name': args.yaml_name,
        'epoch': args.epoch,
        'topk': args.topk,
        'branch_source': args.branch_source,
        'allow_ablation_probes': args.allow_ablation_probes,
        'sequence_count': len(seqs),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    with out_path.open('w') as f:
        for seq in seqs:
            collect_sequence(args, params, seq, f)
    print(f'Wrote {out_path}')


def collect_sequence(args, params, seq_name, writer):
    rgb_imgs, tir_imgs, rgb_gt, tir_gt = load_rgbt_sequence(args.seq_home, seq_name, args.dataset)
    n = min(len(rgb_imgs), len(tir_imgs), len(rgb_gt))
    if args.max_frames > 0:
        n = min(n, args.max_frames)
    if n <= 1:
        return

    tracker = ViPTTrack(params)
    xt = getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb')
    first = get_x_frame(rgb_imgs[0], tir_imgs[0], dtype=xt)
    tracker.initialize(first, {'init_bbox': rgb_gt[0].astype(float).tolist()})

    for frame_idx in range(1, n):
        image = get_x_frame(rgb_imgs[frame_idx], tir_imgs[frame_idx], dtype=xt)
        outputs = tracker.track(image)
        n0 = outputs.get('n0')
        if n0 is None:
            raise RuntimeError('Tracker did not return N0 diagnostics. Check params.n0_collect.')
        branches = n0['branches']
        if args.branch_source == 'auto':
            missing = [name for name in ('rgb', 'tir') if not branches.get(name, {}).get('available', False)]
            if missing:
                raise RuntimeError(f'Missing real branch candidates for {missing}. Use fusion-only or explicit ablation-probe mode if intended.')
        if args.branch_source == 'fusion-only':
            branches = {'fusion': branches['fusion']}

        record = {
            'dataset': args.dataset,
            'sequence': seq_name,
            'frame_idx': frame_idx,
            'rgb_path': rgb_imgs[frame_idx],
            'tir_path': tir_imgs[frame_idx],
            'gt_xywh': rgb_gt[frame_idx].astype(float).tolist(),
            'tir_gt_xywh': tir_gt[frame_idx].astype(float).tolist() if frame_idx < len(tir_gt) else None,
            'tracker_state_before_xywh': n0.get('tracker_state_before_xywh'),
            'pred_xywh': outputs['target_bbox'],
            'best_score': outputs['best_score'],
            'branches': branches,
        }
        writer.write(json.dumps(record) + '\n')
        writer.flush()


if __name__ == '__main__':
    main()
