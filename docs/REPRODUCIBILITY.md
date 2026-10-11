# Setup, provenance and reproducibility

## Archive provenance

The initial repository was uploaded as a collection under
`EviRoute_GitHub_source_v2/`. This cleanup lifts `research_7b/`
and `edge_vitb/` to the repository root **without modifying the
historical research-source file bytes**.

The [source manifest](SOURCE_MANIFEST.sha256) contains the original
SHA-256 checksums for those historical files. It has only been adjusted
to point the collection note at its new archival documentation path.

From the repository root, verify the retained files with:

```bash
sha256sum -c docs/SOURCE_MANIFEST.sha256
```

This proves source-file integrity relative to the original package,
**not correctness of the algorithms or reproducibility of results**.

## Software requirements

- Python 3.10+ (recommended; historical code may require adjustment)
- PyTorch, torchvision, NumPy, Pillow
- DINOv3-capable `timm` version for the ViT-B Mid6 example
- Pandas, scikit-learn, SciPy, OpenCV, Matplotlib, PyYAML, tqdm
- CUDA and appropriate GPU memory for original 7B feature extraction

`requirements.txt` lists libraries but intentionally does not promise
an exact historical CUDA/PyTorch lockfile.

## Data and artifacts

The research head expects external pre-extracted patch tensors and labels:
the single-payload trainer reads `train.pt`, `val.pt`, `test.pt`
from `--feature_dir`. The sharded trainer instead expects a shard root
and an evaluation-feature directory. These are not interchangeable inputs.

The ViT-B demo needs a checkpoint whose contents match
`edge_vitb/models/eviroute.py`:
`args`, `model_state_dict`, `mu`, `std`, `best_epoch`.
The script's default checkpoint path is illustrative: the file is not
distributed here. **Only load checkpoints from sources you trust**:
the historical loader uses `torch.load(..., weights_only=False)`.

## What has not been verified

- End-to-end original 7B model training and GPU/CUDA compatibility
- Numerical agreement with past experimental logs and checkpoints
- Evaluation metric matching against original preprocessed datasets
- RK3588/RKNN graph conversion, runtime, or real device performance

No test success or paper-metric reproduction is implied by a syntax check.

## Attribution and source limitations

The repository supplies the author's research scripts and a PyTorch
adaptation of a pretrained backbone. Underlying model and library
licenses may differ. Refer to [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)
and original upstream conditions.
