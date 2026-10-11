# ViT-B Mid6 clean inference reference

This is the separate frozen DINOv3 ViT-B/16 PyTorch inference implementation, **not** the original 7B training model or an RKNN deployment package.

Run from repository root:

```bash
python edge_vitb/inference/infer_pytorch.py /path/to/image.jpg --checkpoint /path/to/eviroute_vitb_mid6.pt --print_map
```

The checkpoint is external. It must supply scorer weights, feature normalization statistics and matching metadata. The original loader uses `torch.load(..., weights_only=False)`: load only trusted checkpoints.

The input is RGB bicubic-resized to 224×224 without channel mean/std normalization. DINOv3 ViT-B block 6 yields 196 patch tokens of width 768 after removing five prefix tokens. The head scores patches and uses Top-10 mean pooling. This reference alone does **not** demonstrate RK3588 deployment performance.
