#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.backbone import DinoV3ViTBMid6
from models.eviroute import EviRouteInference
from inference.preprocess import preprocess_image


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom == 0:
        return float("nan")

    return float(np.dot(a, b) / denom)


def compare_with_legacy(
    current,
    golden_path,
):
    legacy = np.load(golden_path)

    print()
    print("=" * 72)
    print("LEGACY GOLDEN COMPARISON")
    print("=" * 72)

    current_fp16 = (
        current["legacy_patch_features"]
        .detach()
        .cpu()
        .numpy()
    )

    legacy_fp16 = legacy[
        "block6_patch_feature_fp16"
    ]

    print(
        "FP16 patch exact match :",
        bool(
            np.array_equal(
                current_fp16,
                legacy_fp16,
            )
        ),
    )

    for name, current_key, legacy_key in [
        (
            "normalized feature",
            "normalized_features",
            "normalized_feature",
        ),
        (
            "patch scores",
            "patch_scores",
            "patch_scores",
        ),
    ]:
        a = (
            current[current_key]
            .detach()
            .cpu()
            .numpy()
        )
        b = legacy[legacy_key]

        diff = np.abs(
            a.astype(np.float64)
            - b.astype(np.float64)
        )

        print()
        print(name)
        print(
            "  cosine    :",
            f"{cosine_similarity(a, b):.10f}"
        )
        print(
            "  mean abs  :",
            f"{diff.mean():.10e}"
        )
        print(
            "  max abs   :",
            f"{diff.max():.10e}"
        )

    current_idx = (
        current["top10_indices"][0]
        .detach()
        .cpu()
        .numpy()
    )

    legacy_idx = legacy["top10_indices"]

    print()
    print(
        "Top10 exact match       :",
        bool(
            np.array_equal(
                current_idx,
                legacy_idx,
            )
        ),
    )

    overlap = len(
        set(current_idx.tolist())
        & set(legacy_idx.tolist())
    )

    print(
        "Top10 set overlap       :",
        f"{overlap}/10"
    )

    curr_logit = float(
        current["image_logit"].item()
    )
    old_logit = float(
        legacy["image_logit"]
    )

    curr_prob = float(
        current["fake_probability"].item()
    )
    old_prob = float(
        legacy["fake_probability"]
    )

    print(
        "Logit abs error         :",
        f"{abs(curr_logit-old_logit):.10e}"
    )

    print(
        "Probability abs error   :",
        f"{abs(curr_prob-old_prob):.10e}"
    )


def save_reference(
    path,
    image_tensor,
    full_tokens,
    raw_patches,
    legacy_patches,
    result,
):
    np.savez_compressed(
        path,

        input_tensor=(
            image_tensor
            .detach()
            .cpu()
            .numpy()
        ),

        block6_full_tokens=(
            full_tokens
            .detach()
            .cpu()
            .float()
            .numpy()
        ),

        block6_patch_feature_raw=(
            raw_patches
            .detach()
            .cpu()
            .float()
            .numpy()
        ),

        # Historical feature files stored
        # patch tokens as float16.
        block6_patch_feature_legacy_fp16=(
            legacy_patches
            .detach()
            .cpu()
            .numpy()
        ),

        normalized_feature=(
            result["normalized_features"]
            .detach()
            .cpu()
            .numpy()
        ),

        patch_scores=(
            result["patch_scores"][0]
            .detach()
            .cpu()
            .numpy()
        ),

        top10_indices=(
            result["top10_indices"][0]
            .detach()
            .cpu()
            .numpy()
        ),

        top10_scores=(
            result["top10_scores"][0]
            .detach()
            .cpu()
            .numpy()
        ),

        top10_coordinates=(
            result["top10_coordinates"][0]
            .detach()
            .cpu()
            .numpy()
        ),

        image_logit=np.asarray(
            float(
                result["image_logit"].item()
            ),
            dtype=np.float32,
        ),

        fake_probability=np.asarray(
            float(
                result["fake_probability"].item()
            ),
            dtype=np.float32,
        ),

        prediction=np.asarray(
            int(
                result["prediction"].item()
            ),
            dtype=np.int64,
        ),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        type=str,
    )

    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT
            / "checkpoints"
            / "eviroute_vitb_mid6.pt"
        ),
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--save_npz",
        default=None,
    )

    parser.add_argument(
        "--legacy_golden",
        default=None,
    )

    parser.add_argument(
        "--print_map",
        action="store_true",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 72)
    print("EVIRoute CLEAN PYTORCH INFERENCE")
    print("=" * 72)

    print("image      :", args.image)
    print("device     :", device)
    print("checkpoint :", args.checkpoint)

    # -------------------------------------------------
    # Preprocess
    # -------------------------------------------------

    image_tensor = preprocess_image(
        args.image
    )

    print()
    print(
        "input      :",
        tuple(image_tensor.shape),
        image_tensor.dtype,
        f"[{image_tensor.min():.6f}, "
        f"{image_tensor.max():.6f}]",
    )

    image_device = image_tensor.to(
        device
    )

    # -------------------------------------------------
    # Backbone
    # -------------------------------------------------

    backbone = DinoV3ViTBMid6(
        pretrained=True
    ).to(device)

    backbone.eval()

    full_tokens, raw_patches = backbone(
        image_device
    )

    print(
        "full tokens:",
        tuple(full_tokens.shape),
        full_tokens.dtype,
    )

    print(
        "raw patches:",
        tuple(raw_patches.shape),
        raw_patches.dtype,
    )

    # -------------------------------------------------
    # Historical compatibility roundtrip
    # -------------------------------------------------
    #
    # Original extraction saved:
    # patches.detach().cpu().half()
    #
    # Historical evaluation then used:
    # patches.float()
    #
    # Therefore:
    # float32 -> float16 -> float32
    #
    # before mu/std normalization.
    # -------------------------------------------------

    legacy_patch_fp16 = (
        raw_patches
        .detach()
        .to(torch.float16)
    )

    scorer_input = (
        legacy_patch_fp16
        .to(torch.float32)
    )

    # -------------------------------------------------
    # EviRoute
    # -------------------------------------------------

    detector = EviRouteInference(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    result = detector(
        scorer_input
    )

    logit = float(
        result["image_logit"].item()
    )

    prob = float(
        result["fake_probability"].item()
    )

    pred = int(
        result["prediction"].item()
    )

    indices = (
        result["top10_indices"][0]
        .detach()
        .cpu()
        .tolist()
    )

    coords = (
        result["top10_coordinates"][0]
        .detach()
        .cpu()
        .tolist()
    )

    scores = (
        result["top10_scores"][0]
        .detach()
        .cpu()
        .tolist()
    )

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    print(
        "Prediction :",
        "FAKE" if pred == 1 else "REAL"
    )

    print(
        "Probability:",
        f"{prob:.9f}"
    )

    print(
        "Logit      :",
        f"{logit:.9f}"
    )

    print()
    print("Top-10 indices:")
    print(indices)

    print()
    print("Top-10 coordinates:")
    print(coords)

    print()
    print("Top-10 scores:")
    print(
        [round(x, 6) for x in scores]
    )

    if args.print_map:
        evidence = (
            result["patch_scores"][0]
            .detach()
            .cpu()
            .reshape(14, 14)
            .numpy()
        )

        print()
        print("Evidence map [14x14]:")
        np.set_printoptions(
            precision=3,
            suppress=True,
            linewidth=200,
        )
        print(evidence)

    current = {
        **result,
        "legacy_patch_features":
            legacy_patch_fp16,
    }

    if args.legacy_golden:
        compare_with_legacy(
            current,
            args.legacy_golden,
        )

    if args.save_npz:
        out = Path(args.save_npz)
        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_reference(
            out,
            image_tensor=image_tensor,
            full_tokens=full_tokens,
            raw_patches=raw_patches,
            legacy_patches=legacy_patch_fp16,
            result=result,
        )

        print()
        print("[Saved]")
        print(out)


if __name__ == "__main__":
    main()
