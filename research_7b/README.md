# Frozen DINOv3-7B Mid20 research implementation

This directory contains the original training, feature extraction, evidence analysis and cross-source evaluation scripts. File versions are retained without rewriting or consolidating the historical experiments.

**Core algorithm:** [`train_mid20_edge_debiased_mil_v2.py`](train_mid20_edge_debiased_mil_v2.py): `TopKMILScorer`, `apply_dropout_masks`, `edge_mass_loss`, `checkpoint_score`, `train_one_epoch`, and `evaluate_payload`.

The full-SD1.4 sharded training runner is [`train_mid20_edge_debiased_mil_sharded_fullsd14_v1.py`](train_mid20_edge_debiased_mil_sharded_fullsd14_v1.py). Input tensors, data manifests and checkpoints are external.

For a small feature-payload experiment:

```bash
python research_7b/train_mid20_edge_debiased_mil_v2.py --feature_dir /path/to/features --output_dir /path/to/results
```

Run this command from the repository root. For method, experiment scope and caveats see [method documentation](../docs/METHOD.md) and [experiment map](../docs/EXPERIMENTS.md).
