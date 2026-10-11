from typing import Tuple, Any

import torch
import torch.nn as nn
import timm


def unwrap_tokens(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        if output.ndim != 3:
            raise RuntimeError(
                f"Expected [B,N,D], got {tuple(output.shape)}"
            )
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 3:
                return item

    if isinstance(output, dict):
        for key in ["x", "tokens", "last_hidden_state"]:
            value = output.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                return value

    raise RuntimeError(
        f"Cannot obtain token tensor from {type(output)}"
    )


class DinoV3ViTBMid6(nn.Module):
    """
    Exact clean implementation of the historical small-backbone
    EviRoute feature extraction protocol.

    Backbone:
        DINOv3 ViT-B/16
        timm: vit_base_patch16_dinov3

    Intermediate representation:
        raw output of blocks[6]
        remove 5 prefix tokens
        retain 14x14 = 196 patch tokens
        feature dimension = 768
    """

    TIMM_NAME = "vit_base_patch16_dinov3"

    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            self.TIMM_NAME,
            pretrained=pretrained,
            img_size=224,
        )

        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)

        self.mid_block = 6
        self.num_prefix_tokens = int(
            getattr(self.backbone, "num_prefix_tokens", 5)
        )

        if self.num_prefix_tokens != 5:
            raise RuntimeError(
                f"Expected 5 prefix tokens, "
                f"got {self.num_prefix_tokens}"
            )

        if len(self.backbone.blocks) != 12:
            raise RuntimeError(
                f"Expected 12 blocks, "
                f"got {len(self.backbone.blocks)}"
            )

        grid = self.backbone.patch_embed.grid_size
        grid = (int(grid[0]), int(grid[1]))

        if grid != (14, 14):
            raise RuntimeError(
                f"Expected 14x14 patch grid, got {grid}"
            )

        self.feature_dim = int(self.backbone.num_features)

        if self.feature_dim != 768:
            raise RuntimeError(
                f"Expected feature_dim=768, "
                f"got {self.feature_dim}"
            )

        # Historical experiment freezes the backbone.
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.backbone.eval()

    @torch.inference_mode()
    def forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if images.ndim != 4:
            raise ValueError(
                f"Expected [B,3,224,224], got {images.shape}"
            )

        if tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError(
                f"Expected [B,3,224,224], "
                f"got {tuple(images.shape)}"
            )

        activations = {}

        def hook_fn(module, inputs, output):
            activations["mid"] = output

        handle = (
            self.backbone.blocks[self.mid_block]
            .register_forward_hook(hook_fn)
        )

        try:
            self.backbone.forward_features(images)
        finally:
            handle.remove()

        if "mid" not in activations:
            raise RuntimeError(
                "Mid6 forward hook captured no activation"
            )

        full_tokens = unwrap_tokens(
            activations["mid"]
        )

        expected = self.num_prefix_tokens + 196

        if full_tokens.shape[1] < expected:
            raise RuntimeError(
                f"Expected at least {expected} tokens, "
                f"got {tuple(full_tokens.shape)}"
            )

        patches = full_tokens[
            :,
            self.num_prefix_tokens:
            self.num_prefix_tokens + 196,
            :,
        ]

        if tuple(patches.shape[1:]) != (196, 768):
            raise RuntimeError(
                f"Unexpected patch shape: "
                f"{tuple(patches.shape)}"
            )

        return full_tokens, patches
