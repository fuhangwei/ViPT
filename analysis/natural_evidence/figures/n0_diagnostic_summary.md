# CUETrack N0 Diagnostic Summary — RGBT234 val47

## Dataset

- Frames: 21417
- Split: RGBT234 val47
- Candidate branches: Fusion / RGB / TIR, Top-20 each
- Evidence IoU threshold: 0.5

## Candidate Recall

| Branch | Recall@1 IoU≥0.5 | Recall@20 IoU≥0.5 | Recall@20 IoU≥0.7 |
|---|---:|---:|---:|
| Fusion | 0.7986 | 0.8630 | 0.7499 |
| RGB | 0.7883 | 0.8600 | 0.7445 |
| TIR | 0.6024 | 0.8154 | 0.5386 |
| Union | 0.8372 | 0.8981 | 0.8055 |

Union Top-20 improves over Fusion Top-20 by **0.0351** at IoU≥0.5.

## Evidence State Distribution

| State | Frames |
|---|---:|
| both_agree | 16834 |
| rgb_only | 1584 |
| tir_only | 629 |
| both_conflict | 1 |
| none | 2369 |


## Fusion Evidence Omission

| Metric | Value |
|---|---:|
| Omission rows | 958 |
| Severe omission | 302 |
| Margin omission | 759 |
| Fusion miss with RGB evidence | 265 |
| Fusion miss with TIR evidence | 487 |
| Sequences with omission | 35 |
| rgb_only/tir_only + fusion_miss | 569 |
| both_conflict + fusion_miss | 0 |

## Oracle Gap

| Metric | Value |
|---|---:|
| Fusion top1 success AUC | 0.6456 |
| Fusion topK oracle success AUC | 0.7163 |
| Union oracle success AUC | 0.7511 |
| Oracle gap AUC | 0.1055 |
| Fusion top1 success@0.5 | 0.7986 |
| Union oracle success@0.5 | 0.8981 |
| Recoverable failure rate | 0.0995 |

## N0 Decision

| Criterion | Threshold | Observed | Decision |
|---|---:|---:|---|
| Oracle gap AUC | ≥ 0.03 | 0.1055 | PASS |
| Recoverable failure rate | ≥ 0.03 | 0.0995 | PASS |
| Union Top-20 gain over Fusion Top-20 | > 0 | 0.0351 | PASS |
| Multi-sequence omission events | non-zero | 35 seqs | PASS |
| Unique evidence + fusion miss | non-zero | 569 | PASS |

Conclusion: N0 supports continuing CUETrack. The next controlled step should be E0/E1-style minimal candidate evidence preservation or candidate union, not the full architecture at once.
