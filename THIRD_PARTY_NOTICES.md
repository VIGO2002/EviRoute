# Third-party components and model provenance

**Project author and research implementation:** Chenwei He (何晨炜).

EviRoute's evidence-routing training/evaluation scripts and the
separate lightweight inference reference depend on publicly available
third-party frameworks and pretrained representations.

- **DINOv3 backbone:** pretrained model family developed outside this project.
  Respect the original DINOv3 model-card and checkpoint distribution terms.
- **timm (PyTorch Image Models):** loaded by
  `research_7b/models/dinov3_models.py` and
  `edge_vitb/models/backbone.py`. Project:
  https://github.com/huggingface/pytorch-image-models
- **PyTorch / torchvision:** tensor operations, model modules and transforms:
  https://github.com/pytorch/pytorch
- Supporting numerical / imaging packages include NumPy, pandas,
  scikit-learn, SciPy, Pillow and OpenCV.

This file records dependencies and upstream provenance; it does not
purport to relicense those libraries or pretrained weights. Their
licensing and original copyrights remain with the respective authors.
No DINOv3 pretrained weights are redistributed by this repository.
