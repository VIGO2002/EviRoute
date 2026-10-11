# EviRoute: Evidence Routing for AI-Generated Image Detection

**Research implementation by Chenwei He (何晨炜)**  
University of Electronic Science and Technology of China

**Manuscript:** EviRoute — submitted to IEEE ICASSP 2027 (**under review**).

EviRoute studies a practical weakness of image-forgery detectors: a high-confidence prediction does not necessarily mean that the **selected evidence tokens** are meaningful. In a frozen vision backbone, naïve patch scoring can over-select border tokens. EviRoute uses **region-aware token score dropout**, a **fake-only boundary-mass penalty**, and **performance-constrained checkpoint selection** to reduce that shortcut while retaining detection performance.

The repository contains two **distinct** tracks: the DINOv3-7B research implementation used to study evidence routing, and a separate DINOv3 ViT-B/16 clean-PyTorch inference adaptation. Their architectures, training conditions and checkpoints should not be interchanged.

## How it works

```text
Image ──► frozen DINOv3 backbone ──► intermediate patch tokens
                                               │
                                  feature normalization (μ, σ)
                                               │
                                     token scorer (MLP)
                                               │
                                196 patch suspicion scores
                                     │             │
                   TRAINING ONLY    │             │  CLEAN INFERENCE
           random token / boundary  │             └── top-10 mean
           score dropout             │                      │
           fake-only edge penalty    │                image logit
           BCE classification        │                      │
                     │               │                fake probability
            validation AUC + edge    │
            checkpoint selection     └── 14×14 evidence map
```

The train-time dropout is applied to **patch scores before Top-K pooling**; it is not a change to the frozen backbone's self-attention. The edge-mass penalty is computed from **unmasked scores for fake samples**. At inference, neither dropout nor edge penalties are applied.

## Explore the actual implementation

| Key mechanism | Source | What to look for |
|---|---|---|
| **Top-K multiple-instance scoring** | [`research_7b/train_mid20_topk_mil_v1.py`](research_7b/train_mid20_topk_mil_v1.py) | Baseline scorer, aggregation |
| **Region-aware score masking** | [`apply_dropout_masks`](research_7b/train_mid20_edge_debiased_mil_v2.py) | Random, outer-ring and second-ring token masking |
| **Fake-only border regularization** | [`edge_mass_loss`](research_7b/train_mid20_edge_debiased_mil_v2.py) | Softmax evidence mass constraint |
| **Performance-constrained selection** | [`checkpoint_score`](research_7b/train_mid20_edge_debiased_mil_v2.py) | AUC gate and border evidence trade-off |
| **Larger-scale sharded training** | [`train_mid20_edge_debiased_mil_sharded_fullsd14_v1.py`](research_7b/train_mid20_edge_debiased_mil_sharded_fullsd14_v1.py) | Full SD1.4 training and checkpoint protocol |
| **Full cross-source evaluation** | [`eval_fullsd14_genimage_aiginow_v1.py`](research_7b/eval_fullsd14_genimage_aiginow_v1.py) | Source-wise AUC/AP/ACC and Top-1/Top-10 edge metrics |
| **Clean smaller-backbone inference** | [`edge_vitb/inference/infer_pytorch.py`](edge_vitb/inference/infer_pytorch.py) | ViT-B Mid6 feature extraction and top-10 token predictions |

For the equations, token masks and B1/B2 definitions, see the [method walkthrough](docs/METHOD.md). For evidence audits, see the [experiment map](docs/EXPERIMENTS.md).

## Two implementations, two purposes

| | 7B research | ViT-B inference reference |
|---|---|---|
| Backbone | DINOv3 ViT-7B, frozen | DINOv3 ViT-B/16, frozen |
| Intermediate tokens | Mid20, 14×14 patches | Mid6, 14×14 patches |
| Feature width | 4096 | 768 |
| Scoring head | 4096 → 256 → 1 | 768 → 256 → 1 |
| Image pooling | Top-10 mean | Top-10 mean |
| Purpose | Training, evidence audits, cross-source evaluation | Clean PyTorch image inference |

The `edge_vitb/` track **is not a complete RKNN or RK3588 deployment package**. It documents the lighter PyTorch inference protocol; RKNN conversion, runtime assets and device integration are not included here.

## Quick start

Use a compatible CUDA/PyTorch installation for the large-backbone experiments. Python 3.10+ is recommended.

```bash
git clone https://github.com/VIGO2002/EviRoute.git
cd EviRoute
pip install -r requirements.txt
```

### 1. Research head on pre-extracted Mid20 features

```bash
python research_7b/train_mid20_edge_debiased_mil_v2.py \
    --feature_dir /path/to/mid20_features \
    --output_dir /path/to/results
```

The feature directory is expected to contain the historical `train.pt`, `val.pt` and `test.pt` payloads. Feature extraction is a **separate and resource-intensive stage**. For the sharded full-SD1.4 procedure, use the corresponding [training script](research_7b/train_mid20_edge_debiased_mil_sharded_fullsd14_v1.py) with its required `--train_shard_root`, `--eval_feature_dir` and `--output_dir` arguments. See [setup/provenance notes](docs/REPRODUCIBILITY.md).

### 2. Smaller-backbone clean inference (requires external weights)

```bash
python edge_vitb/inference/infer_pytorch.py /path/to/image.jpg \
    --checkpoint /path/to/eviroute_vitb_mid6.pt \
    --print_map
```

The ViT-B implementation loads a frozen `timm` backbone and an EviRoute scorer checkpoint containing model weights, feature normalization statistics and scorer metadata. **No checkpoint or dataset is bundled in this source repository.** Use checkpoints only from trusted sources because the historical loader deserializes Python objects.

## Repository structure

```text
EviRoute/
├── research_7b/              # Original DINOv3-7B research scripts
│   └── models/               # DINOv3 backbone feature helpers
├── edge_vitb/                # DINOv3 ViT-B clean PyTorch reference
│   ├── configs/
│   ├── models/
│   └── inference/
├── docs/                     # Method, experiments, provenance, source manifest
├── tests/                    # Static and optional CPU correctness checks
├── .github/workflows/        # Automated syntax / source-integrity checks
├── requirements.txt
└── .gitignore
```

## What is (and is not) released

The code includes the training, evaluation, diagnostic and inference scripts. The original source files are preserved when moved from the `EviRoute_GitHub_source_v2/` collection into these top-level folders; the [SHA-256 manifest](docs/SOURCE_MANIFEST.sha256) helps verify that provenance. We have **not** rerun the original experimental metrics or validated historical checkpoints against this reorganized source collection.

Datasets, model weights, extracted patch-feature caches, experimental result archives, and actual RKNN deployment binaries are **not part of this repository**. Consequently, this source collection alone does not reproduce any manuscript result without the external artifacts and compute.

## Research ownership and acknowledgments

**EviRoute research and implementation: Chenwei He (何晨炜).** The work relies on third-party libraries and pretrained DINOv3 backbones, including `timm` and PyTorch. Refer to [third-party notices](THIRD_PARTY_NOTICES.md) and the original upstream licenses and model-weight conditions.

**Research status:** submitted for review; do not interpret this repository as an accepted conference publication.
