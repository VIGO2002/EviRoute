from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn


class EviRouteScorer(nn.Module):
    def __init__(
        self,
        feat_dim=768,
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def score_patches(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.scorer(x).squeeze(-1)


class EviRouteInference(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        device="cpu",
    ):
        super().__init__()

        checkpoint_path = Path(checkpoint_path)

        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        args = ckpt["args"]

        if int(args["feat_dim"]) != 768:
            raise RuntimeError(
                f'Unexpected feat_dim={args["feat_dim"]}'
            )

        if int(args["hidden_dim"]) != 256:
            raise RuntimeError(
                f'Unexpected hidden_dim={args["hidden_dim"]}'
            )

        if int(args["topk"]) != 10:
            raise RuntimeError(
                f'Unexpected topk={args["topk"]}'
            )

        if str(args["pool"]) != "topk":
            raise RuntimeError(
                f'Unexpected pool={args["pool"]}'
            )

        self.topk = 10

        self.scorer = EviRouteScorer(
            feat_dim=768,
            hidden_dim=256,
            dropout=float(args["dropout"]),
        )

        self.scorer.load_state_dict(
            ckpt["model_state_dict"],
            strict=True,
        )

        self.scorer.eval()

        self.register_buffer(
            "mu",
            ckpt["mu"].float().view(1, 1, 768),
        )

        self.register_buffer(
            "std",
            ckpt["std"].float().view(1, 1, 768),
        )

        if torch.any(self.std < 1e-6):
            raise RuntimeError(
                "Checkpoint contains invalid std values"
            )

        self.best_epoch = int(
            ckpt["best_epoch"]
        )

        self.to(device)
        self.eval()

    def normalize_features(
        self,
        patches: torch.Tensor,
    ) -> torch.Tensor:
        # Exact historical evaluation formula.
        return (
            patches.float() - self.mu
        ) / self.std

    @torch.inference_mode()
    def forward(
        self,
        patches: torch.Tensor,
    ) -> Dict[str, Any]:

        normalized = self.normalize_features(
            patches
        )

        patch_scores = self.scorer.score_patches(
            normalized
        )

        top10_scores, top10_indices = torch.topk(
            patch_scores,
            k=self.topk,
            dim=1,
        )

        image_logit = top10_scores.mean(
            dim=1
        )

        fake_probability = torch.sigmoid(
            image_logit
        )

        prediction = (
            fake_probability >= 0.5
        ).long()

        coordinates = torch.stack(
            [
                top10_indices // 14,
                top10_indices % 14,
            ],
            dim=-1,
        )

        return {
            "normalized_features": normalized,
            "patch_scores": patch_scores,
            "top10_indices": top10_indices,
            "top10_scores": top10_scores,
            "top10_coordinates": coordinates,
            "image_logit": image_logit,
            "fake_probability": fake_probability,
            "prediction": prediction,
        }
