# EviRoute method and code walkthrough

This document maps the existing research scripts to the algorithmic concepts; it is **not** a substitute for a line-by-line paper reproduction.

## 1. Frozen intermediate patch features

`research_7b/models/dinov3_models.py` contains DINO feature helpers. The main EviRoute research path uses **frozen DINOv3 ViT-7B, Mid20**, extracting 196 patch tokens (14 × 14, 4096 channels). The separate `edge_vitb/models/backbone.py` extracts Mid6 from frozen DINOv3 ViT-B/16, with 196 patch tokens × 768 channels.

In both cases a feature normalization stage uses checkpoint-specific `mu` and `std`. The smaller-backbone inference explicitly reproduces a historical float32 → float16 → float32 patch-cache round trip before feature normalization.

## 2. Patch-level scorer and image decision

A two-layer MLP scores each patch:

```text
token feature (D) → Linear(D, 256) → GELU → Dropout(0.1) → Linear(256, 1)
```

The clean image logit is the arithmetic mean of the top-10 patch scores. The fake probability is the sigmoid of that logit. This **Top-K multiple-instance learning (MIL)** formulation is visible in `pool_scores` of `research_7b/train_mid20_edge_debiased_mil_v2.py`.

The score map is a diagnostic *candidate evidence ranking*. Without a corresponding intervention experiment, attention/score visualization alone does not prove causality.

## 3. Region-aware score dropout (training only)

Implementation: `apply_dropout_masks` in `research_7b/train_mid20_edge_debiased_mil_v2.py`.

- Random masking is applied across all token scores.
- **B1** is the outermost ring of the 14 × 14 patch grid.
- **B2** is the outer two rings; **B2-only** excludes B1, preventing the two edge masks from double-counting the same positions.
- Masked scores receive a large negative value (`-1e4`) *before* Top-K pooling; an all-masked case is reset rather than producing an empty selection.

For the sharded training implementation, CLI defaults are `token_dropout=0.15`, `edge1_dropout=0.50`, `edge2_only_dropout=0.25`. **Do not apply these masks at inference.**

## 4. Fake-only evidence-mass penalty

Implementation: `edge_mass_loss` in the same training script.

For a *fake* sample, convert **clean (unmasked)** patch scores to a distribution over patches with a temperature-scaled softmax. Sum the distribution within B1 and/or B2. Penalize only mass exceeding the selected ceiling:

```text
L_B1 = λ_B1 × mean( ReLU(mass(B1) − ρ_B1)^2 )
L_B2 = λ_B2 × mean( ReLU(mass(B2) − ρ_B2)^2 )
L = BCEWithLogits(top10_mean(masked_scores), label) + L_B1 + L_B2
```

Real samples **do not** receive this edge-mass penalty. The full-SD1.4 sharded script defaults to `λ_B1=0.01, ρ_B1=0.40, λ_B2=0.0, ρ_B2=0.60`. These are source-code defaults for that script, not global constants of every experiment.

## 5. Performance-constrained checkpoint selection

Implementation: `checkpoint_score` in the source training script.

A validation AUC floor is checked first. When AUC passes the floor, the selection objective favors lower B1/B2 evidence concentration; otherwise an AUC shortfall penalty discourages a weak detector. This checkpoint-selection heuristic is part of the method—not a test-time postprocessing step.

**Important configuration difference:** The single-payload script defaults to a minimum validation AUC of `0.995`; the separate ViT-B configuration records `0.98182222`. They describe **different experimental settings** and must not be silently treated as identical.

## 6. Auditing evidence

The research folder contains complementary scripts for:
- Evidence-region faithfulness and counterfactual feature replacement
- Border-band and crop-boundary ablations
- Perturbation/crop/flip evidence stability
- Low-level characterization and source-wise cross-generator evaluation

See [the experiment map](EXPERIMENTS.md) for exact filenames.

## 7. Limitations

Token scores are **not ground-truth pixel masks**. Boundary-mass reduction alone cannot establish causal faithfulness. The repo provides code implementing separate evaluation protocols, but does not bundle benchmark outputs or checkpoints.
