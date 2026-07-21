import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Compute CUETrack N0 diagnostics from collected candidates.')
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--topk', default=20, type=int)
    parser.add_argument('--evidence-iou', default=0.5, type=float)
    parser.add_argument('--agree-iou', default=0.3, type=float)
    parser.add_argument('--severe-iou', default=0.3, type=float)
    parser.add_argument('--omission-margin', default=0.2, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_jsonl(args.candidates)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_metrics = [compute_frame_metrics(r, args) for r in records if valid_record(r)]
    write_candidate_recall(frame_metrics, out_dir)
    write_state_distribution(frame_metrics, out_dir)
    write_omissions(frame_metrics, out_dir, args)
    write_oracle_gap(frame_metrics, out_dir)
    print(f'Wrote metrics to {out_dir}')


def load_jsonl(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def valid_record(record):
    gt = record.get('gt_xywh')
    return gt and len(gt) >= 4 and gt[2] > 0 and gt[3] > 0


def compute_frame_metrics(record, args):
    gt = record['gt_xywh']
    branches = {}
    for name, branch in record.get('branches', {}).items():
        if not branch.get('available', False):
            branches[name] = {'available': False}
            continue
        boxes = branch.get('topk_xywh', [])[:args.topk]
        scores = branch.get('topk_scores', [])[:args.topk]
        ious = [iou_xywh(box, gt) for box in boxes]
        best_idx = int(np.argmax(ious)) if ious else -1
        top1_iou = ious[0] if ious else 0.0
        branches[name] = {
            'available': True,
            'source': branch.get('source', ''),
            'boxes': boxes,
            'scores': scores,
            'ious': ious,
            'best_iou': max(ious) if ious else 0.0,
            'top1_iou': top1_iou,
            'best_rank': best_idx + 1 if best_idx >= 0 else None,
            'best_box': boxes[best_idx] if best_idx >= 0 else None,
            'best_score': scores[best_idx] if best_idx >= 0 and best_idx < len(scores) else None,
        }
    fusion = branches.get('fusion', {'available': False, 'best_iou': 0.0, 'top1_iou': 0.0})
    union_best_iou = 0.0
    for branch in branches.values():
        if branch.get('available'):
            union_best_iou = max(union_best_iou, branch['best_iou'])
    rgb = branches.get('rgb', {'available': False, 'best_iou': 0.0})
    tir = branches.get('tir', {'available': False, 'best_iou': 0.0})
    state = evidence_state(rgb, tir, args.evidence_iou, args.agree_iou)
    fusion_state = 'fusion_hit' if fusion.get('best_iou', 0.0) >= args.evidence_iou else 'fusion_miss'
    return {
        'record': record,
        'branches': branches,
        'state': state,
        'fusion_state': fusion_state,
        'fusion_best_iou': fusion.get('best_iou', 0.0),
        'fusion_top1_iou': fusion.get('top1_iou', 0.0),
        'union_best_iou': union_best_iou,
    }


def evidence_state(rgb, tir, evidence_iou, agree_iou):
    rgb_has = rgb.get('available', False) and rgb.get('best_iou', 0.0) >= evidence_iou
    tir_has = tir.get('available', False) and tir.get('best_iou', 0.0) >= evidence_iou
    if rgb_has and not tir_has:
        return 'rgb_only'
    if tir_has and not rgb_has:
        return 'tir_only'
    if not rgb_has and not tir_has:
        return 'none'
    best_rgb = rgb.get('best_box')
    best_tir = tir.get('best_box')
    if best_rgb is not None and best_tir is not None and iou_xywh(best_rgb, best_tir) >= agree_iou:
        return 'both_agree'
    return 'both_conflict'


def write_candidate_recall(metrics, out_dir):
    ks = [1, 5, 10, 20]
    thresholds = [0.3, 0.5, 0.7]
    branches = sorted({name for m in metrics for name in m['branches']})
    summary = {'num_frames': len(metrics), 'branches': {}}
    for branch in branches:
        summary['branches'][branch] = {}
        for k in ks:
            values = []
            for metric in metrics:
                b = metric['branches'].get(branch)
                if not b or not b.get('available'):
                    continue
                values.append(max(b['ious'][:k]) if b['ious'] else 0.0)
            summary['branches'][branch][f'recall@{k}'] = {
                f'iou>{thr}': safe_mean([v >= thr for v in values]) for thr in thresholds
            }
            summary['branches'][branch][f'recall@{k}']['num_frames'] = len(values)
    summary['union'] = {}
    for k in ks:
        values = []
        for metric in metrics:
            best = 0.0
            for b in metric['branches'].values():
                if b.get('available') and b.get('ious'):
                    best = max(best, max(b['ious'][:k]))
            values.append(best)
        summary['union'][f'recall@{k}'] = {f'iou>{thr}': safe_mean([v >= thr for v in values]) for thr in thresholds}
        summary['union'][f'recall@{k}']['num_frames'] = len(values)
    write_json(out_dir / 'candidate_recall_summary.json', summary)


def write_state_distribution(metrics, out_dir):
    states = Counter(m['state'] for m in metrics)
    fusion_states = Counter(m['fusion_state'] for m in metrics)
    joint = Counter(f"{m['state']}|{m['fusion_state']}" for m in metrics)
    by_sequence = defaultdict(Counter)
    for m in metrics:
        by_sequence[m['record']['sequence']][m['state']] += 1
        by_sequence[m['record']['sequence']][m['fusion_state']] += 1
    summary = {
        'num_frames': len(metrics),
        'rgb_tir_state_counts': dict(states),
        'fusion_state_counts': dict(fusion_states),
        'joint_state_counts': dict(joint),
        'by_sequence': {seq: dict(counts) for seq, counts in sorted(by_sequence.items())},
    }
    write_json(out_dir / 'evidence_state_distribution.json', summary)


def write_omissions(metrics, out_dir, args):
    path = out_dir / 'evidence_omission_events.csv'
    fields = ['dataset', 'sequence', 'frame_idx', 'rgb_path', 'tir_path', 'state', 'event_type',
              'rgb_best_iou', 'tir_best_iou', 'fusion_best_iou', 'fusion_top1_iou', 'union_best_iou',
              'best_source_branch', 'best_source_rank', 'source_best_score', 'gt_xywh', 'pred_xywh']
    rows = []
    for m in metrics:
        branches = m['branches']
        source = best_source(branches)
        source_iou = source[1].get('best_iou', 0.0) if source else 0.0
        fusion_iou = m['fusion_best_iou']
        source_has = source_iou >= args.evidence_iou
        fusion_has = fusion_iou >= args.evidence_iou
        event_types = []
        if source_has and not fusion_has:
            event_types.append(f"fusion_miss_with_{source[0]}_evidence")
        if source_has and source_iou >= args.evidence_iou and fusion_iou < args.severe_iou:
            event_types.append('severe_omission')
        if source_has and source_iou - fusion_iou >= args.omission_margin:
            event_types.append('margin_omission')
        if not event_types:
            continue
        r = m['record']
        rows.append({
            'dataset': r['dataset'],
            'sequence': r['sequence'],
            'frame_idx': r['frame_idx'],
            'rgb_path': r.get('rgb_path', ''),
            'tir_path': r.get('tir_path', ''),
            'state': m['state'],
            'event_type': ';'.join(event_types),
            'rgb_best_iou': branches.get('rgb', {}).get('best_iou'),
            'tir_best_iou': branches.get('tir', {}).get('best_iou'),
            'fusion_best_iou': fusion_iou,
            'fusion_top1_iou': m['fusion_top1_iou'],
            'union_best_iou': m['union_best_iou'],
            'best_source_branch': source[0] if source else '',
            'best_source_rank': source[1].get('best_rank') if source else '',
            'source_best_score': source[1].get('best_score') if source else '',
            'gt_xywh': json.dumps(r.get('gt_xywh')),
            'pred_xywh': json.dumps(r.get('pred_xywh')),
        })
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_oracle_gap(metrics, out_dir):
    fusion = [m['fusion_top1_iou'] for m in metrics]
    fusion_topk = [m['fusion_best_iou'] for m in metrics]
    oracle = [m['union_best_iou'] for m in metrics]
    summary = {
        'num_frames': len(metrics),
        'fusion_top1_success_auc': success_auc(fusion),
        'fusion_topk_oracle_success_auc': success_auc(fusion_topk),
        'union_oracle_success_auc': success_auc(oracle),
        'oracle_gap_auc': success_auc(oracle) - success_auc(fusion),
        'fusion_top1_success_0_5': safe_mean([v >= 0.5 for v in fusion]),
        'union_oracle_success_0_5': safe_mean([v >= 0.5 for v in oracle]),
        'recoverable_failure_rate': safe_mean([f < 0.5 and o >= 0.5 for f, o in zip(fusion, oracle)]),
    }
    write_json(out_dir / 'oracle_gap_summary.json', summary)


def best_source(branches):
    candidates = [(name, b) for name, b in branches.items() if name != 'fusion' and b.get('available')]
    if not candidates:
        candidates = [(name, b) for name, b in branches.items() if b.get('available')]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1].get('best_iou', 0.0))


def success_auc(values):
    if not values:
        return 0.0
    thresholds = np.linspace(0.0, 1.0, 21)
    return float(np.mean([np.mean([v >= t for v in values]) for t in thresholds]))


def iou_xywh(a, b):
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = [float(x) for x in a[:4]]
    bx, by, bw, bh = [float(x) for x in b[:4]]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def safe_mean(values):
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
