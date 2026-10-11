#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import functional as TF

from run_input_space_edge_diagnostic_v1 import (
    load_protocol_module,
    infer_gt,
    find_case_image,
    load_raw_image,
    infer_batch,
)

from run_coarse_input_causal_map_v1 import (
    GRID,
    perturb_block,
)


def safe_corr(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def topk_set(array, k=3):
    flat = np.asarray(array).reshape(-1)
    return set(np.argsort(flat)[-k:].tolist())


@torch.inference_mode()
def compute_causal_map(
    image,
    gt,
    mode,
    model,
    probe,
    fisher_abs_indices,
    real_idx,
    fake_idx,
    device,
    batch_size,
):
    variants = [image]

    for row in range(GRID):
        for col in range(GRID):
            variants.append(
                perturb_block(
                    image,
                    row,
                    col,
                    mode=mode,
                )
            )

    margins, fake_probs = infer_batch(
        variants,
        model,
        probe,
        fisher_abs_indices,
        real_idx,
        fake_idx,
        device,
        batch_size,
    )

    baseline_margin = float(margins[0])
    perturbed = margins[1:].reshape(GRID, GRID)

    class_sign = 1.0 if gt == "Fake" else -1.0
    support_drop = class_sign * (
        baseline_margin - perturbed
    )

    return support_drop, baseline_margin, float(fake_probs[0])


def summarize(df, group):
    sub = df if group == "All" else df[df["gt"] == group]

    if len(sub) == 0:
        return {"group": group, "n": 0}

    return {
        "group": group,
        "n": int(len(sub)),
        "mean_original_margin":
            float(sub["original_margin"].mean()),
        "mean_flip_margin":
            float(sub["flip_margin"].mean()),
        "mean_abs_margin_change":
            float(sub["margin_abs_change"].mean()),

        "prediction_consistency":
            float(sub["prediction_consistent"].mean()),

        "mean_content_aligned_corr":
            float(sub["content_aligned_corr"].mean()),

        "mean_absolute_position_corr":
            float(sub["absolute_position_corr"].mean()),

        "mean_equivariance_advantage":
            float(sub["equivariance_advantage"].mean()),

        "content_aligned_corr_higher_rate":
            float(
                (
                    sub["content_aligned_corr"]
                    > sub["absolute_position_corr"]
                ).mean()
            ),

        "top1_equivariant_rate":
            float(sub["top1_equivariant"].mean()),

        "mean_top3_equivariant_overlap":
            float(sub["top3_equivariant_overlap"].mean()),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_csv", required=True)
    ap.add_argument(
        "--cases_root",
        default="aigi_now_eval/probe_genimage/cases_unique",
    )
    ap.add_argument(
        "--protocol_file",
        default="repro_pack/fgts_core/test_system1_protocol.py",
    )
    ap.add_argument(
        "--ckpt_path",
        default="checkpoints/AIGCDetectionBenchmark/linear_probe.pth",
    )
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--num_fake", type=int, default=10)
    ap.add_argument("--num_real", type=int, default=10)
    ap.add_argument("--fake_margin_min", type=float, default=4.0)

    ap.add_argument(
        "--mode",
        choices=["local_mean", "blur"],
        default="local_mean",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    protocol = load_protocol_module(args.protocol_file)

    DinoV3Model = protocol.DinoV3Model
    LinearProbe = protocol.LinearProbe

    real_idx = int(getattr(protocol, "REAL_CLASS_IDX", 0))
    fake_idx = int(getattr(protocol, "FAKE_CLASS_IDX", 1))

    print("[*] Loading DINOv3...")
    model = DinoV3Model(
        model_name="dinov3_vit_7b",
        pool_type="patch_avg",
    ).to(args.device)
    model.eval()

    print("[*] Loading probe...")
    ckpt = torch.load(args.ckpt_path, map_location=args.device)

    fisher_abs_indices = [
        int(x) for x in ckpt["token_indices"]
    ]

    state = ckpt.get(
        "probe_state_dict",
        ckpt.get("model_state_dict"),
    )

    probe = LinearProbe(
        input_dim=state["fc.weight"].shape[1],
        num_classes=state["fc.weight"].shape[0],
    ).to(args.device)

    probe.load_state_dict(state, strict=True)
    probe.eval()

    df = pd.read_csv(args.input_csv)

    if "gt" not in df.columns:
        df["gt"] = df["case_name"].apply(infer_gt)

    fake_df = df[df["gt"] == "Fake"].copy()

    if "margin_full" in fake_df.columns:
        fake_df = fake_df[
            fake_df["margin_full"] > args.fake_margin_min
        ]

    real_df = df[df["gt"] == "Real"].copy()

    fake_sample = fake_df.sample(
        n=min(args.num_fake, len(fake_df)),
        random_state=args.seed,
    )
    real_sample = real_df.sample(
        n=min(args.num_real, len(real_df)),
        random_state=args.seed,
    )

    selected = pd.concat(
        [fake_sample, real_sample],
        ignore_index=True,
    ).sample(
        frac=1.0,
        random_state=args.seed,
    ).reset_index(drop=True)

    rows = []

    for i, row in selected.iterrows():
        case_name = str(row["case_name"])
        gt = str(row["gt"])

        try:
            image_path = find_case_image(
                args.cases_root,
                case_name,
            )

            original = load_raw_image(image_path)
            flipped = TF.hflip(original)

            original_map, original_margin, original_prob = (
                compute_causal_map(
                    original,
                    gt,
                    args.mode,
                    model,
                    probe,
                    fisher_abs_indices,
                    real_idx,
                    fake_idx,
                    args.device,
                    args.batch_size,
                )
            )

            flip_map, flip_margin, flip_prob = compute_causal_map(
                flipped,
                gt,
                args.mode,
                model,
                probe,
                fisher_abs_indices,
                real_idx,
                fake_idx,
                args.device,
                args.batch_size,
            )

            # 翻回原图坐标系
            aligned_flip_map = np.fliplr(flip_map)

            content_corr = safe_corr(
                original_map,
                aligned_flip_map,
            )

            absolute_corr = safe_corr(
                original_map,
                flip_map,
            )

            top1_original = int(np.argmax(original_map))
            top1_aligned = int(np.argmax(aligned_flip_map))

            top3_original = topk_set(original_map, 3)
            top3_aligned = topk_set(aligned_flip_map, 3)

            prediction_original = original_margin > 0
            prediction_flip = flip_margin > 0

            rows.append({
                "case_name": case_name,
                "gt": gt,
                "mode": args.mode,

                "original_fake_prob": original_prob,
                "flip_fake_prob": flip_prob,

                "original_margin": original_margin,
                "flip_margin": flip_margin,
                "margin_abs_change":
                    abs(original_margin - flip_margin),

                "prediction_consistent":
                    prediction_original == prediction_flip,

                "content_aligned_corr": content_corr,
                "absolute_position_corr": absolute_corr,

                "equivariance_advantage":
                    content_corr - absolute_corr,

                "top1_equivariant":
                    top1_original == top1_aligned,

                "top3_equivariant_overlap":
                    len(top3_original & top3_aligned) / 3.0,
            })

            print(
                f"[{i+1:02d}/{len(selected):02d}] "
                f"{gt:4s} "
                f"aligned={content_corr:+.3f} "
                f"absolute={absolute_corr:+.3f} "
                f"adv={content_corr-absolute_corr:+.3f}"
            )

        except Exception as exc:
            print(f"[!] failed {case_name}: {exc}")

    result = pd.DataFrame(rows)

    result_path = os.path.join(
        args.output_dir,
        "causal_flip_equivariance_results.csv",
    )

    result.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    summaries = [
        summarize(result, "Fake"),
        summarize(result, "Real"),
        summarize(result, "All"),
    ]

    summary_path = os.path.join(
        args.output_dir,
        "causal_flip_equivariance_summary.json",
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": args.mode,
                "groups": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n== summary ==")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))

    print("\nsaved:", result_path)
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
