#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFilter, ImageOps


SOURCE_ORDER = [
    "Nano",
    "flux-dev",
    "flux-kera",
    "flux-kontext",
    "flux-pro",
    "gpt4o",
    "jimeng",
    "keling",
    "minimax",
]

MODE_ORDER = ["pixel", "semantic"]

TRANSFORMS = [
    "clean",
    "flip",
    "jpeg_q75",
    "blur_r1",
    "resize_half",
]


try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BICUBIC = Image.BICUBIC


def parse_filter(value, allowed):
    if value is None or value.strip().lower() in {"", "all"}:
        return list(allowed)

    selected = [x.strip() for x in value.split(",") if x.strip()]
    invalid = [x for x in selected if x not in allowed]

    if invalid:
        raise ValueError(
            f"Invalid values: {invalid}; allowed={allowed}"
        )

    return selected


def load_clean_224(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize((224, 224), BICUBIC)
        return image.copy()


def create_variant(clean, transform):
    if transform == "clean":
        return clean.copy()

    if transform == "flip":
        return ImageOps.mirror(clean)

    if transform == "blur_r1":
        return clean.filter(
            ImageFilter.GaussianBlur(radius=1.0)
        )

    if transform == "resize_half":
        small = clean.resize((112, 112), BICUBIC)
        return small.resize((224, 224), BICUBIC)

    if transform == "jpeg_q75":
        # JPEG 由保存阶段处理
        return clean.copy()

    raise ValueError(transform)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_index",
        required=True,
    )
    parser.add_argument(
        "--output_root",
        required=True,
    )
    parser.add_argument(
        "--sources",
        default="all",
        help="Comma-separated source names or all",
    )
    parser.add_argument(
        "--modes",
        default="all",
        help="Comma-separated modes or all",
    )
    parser.add_argument(
        "--max_per_label",
        type=int,
        default=0,
        help="0 means use all samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    args = parser.parse_args()

    input_index = Path(args.input_index)
    output_root = Path(args.output_root)

    image_root = output_root / "images"
    split_root = output_root / "splits"

    image_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    selected_sources = parse_filter(
        args.sources,
        SOURCE_ORDER,
    )
    selected_modes = parse_filter(
        args.modes,
        MODE_ORDER,
    )

    df = pd.read_csv(input_index)

    required = [
        "source",
        "mode",
        "subset_idx",
        "orig_idx",
        "label",
        "path",
        "case_name",
    ]

    missing = [x for x in required if x not in df.columns]
    if missing:
        raise KeyError(
            f"Input index missing columns: {missing}; "
            f"existing={list(df.columns)}"
        )

    df = df[
        df["source"].isin(selected_sources)
        & df["mode"].isin(selected_modes)
    ].copy()

    if args.max_per_label > 0:
        sampled = []

        for (source, mode, label), group in df.groupby(
            ["source", "mode", "label"],
            sort=False,
        ):
            n = min(args.max_per_label, len(group))
            sampled.append(
                group.sample(
                    n=n,
                    random_state=(
                        args.seed
                        + int(label)
                        + sum(ord(c) for c in source + mode)
                    ),
                )
            )

        df = pd.concat(sampled, ignore_index=True)

    all_rows = []
    failed = []

    for source in selected_sources:
        for mode in selected_modes:
            group = df[
                (df["source"] == source)
                & (df["mode"] == mode)
            ].copy()

            if len(group) == 0:
                continue

            group = group.sort_values(
                ["label", "subset_idx"]
            ).reset_index(drop=True)

            split_rows = []

            for _, row in group.iterrows():
                original_path = Path(str(row["path"]))

                if not original_path.exists():
                    failed.append(str(original_path))
                    print("[Missing]", original_path)
                    continue

                clean = load_clean_224(original_path)

                subset_idx = int(row["subset_idx"])
                label = int(row["label"])
                label_name = "fake" if label == 1 else "real"

                sample_id = (
                    f"{source}__{mode}__"
                    f"{label_name}__{subset_idx:04d}"
                )

                sample_dir = (
                    image_root
                    / source
                    / mode
                    / sample_id
                )
                sample_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                for transform in TRANSFORMS:
                    variant = create_variant(clean, transform)

                    if transform == "jpeg_q75":
                        out_path = sample_dir / "jpeg_q75.jpg"
                        variant.save(
                            out_path,
                            format="JPEG",
                            quality=75,
                            subsampling=2,
                            optimize=False,
                        )
                    else:
                        out_path = sample_dir / f"{transform}.png"
                        variant.save(
                            out_path,
                            format="PNG",
                        )

                    transformed_case = (
                        f"{row['case_name']}"
                        f"__stability_{transform}"
                    )

                    out_row = {
                        # extractor 使用的字段
                        "path": str(out_path),
                        "label": label,
                        "case_name": transformed_case,
                        "source": source,
                        "split": "test",

                        # 后续配对和审计字段
                        "sample_id": sample_id,
                        "mode": mode,
                        "transform": transform,
                        "subset_idx": subset_idx,
                        "orig_idx": int(row["orig_idx"]),
                        "original_path": str(original_path),
                        "original_case_name": str(row["case_name"]),
                    }

                    split_rows.append(out_row)
                    all_rows.append(out_row)

            split_df = pd.DataFrame(split_rows)

            out_dir = split_root / source / mode
            out_dir.mkdir(parents=True, exist_ok=True)

            out_csv = out_dir / "stability.csv"
            split_df.to_csv(out_csv, index=False)

            print(
                f"[Saved] {source}/{mode}: "
                f"base={split_df['sample_id'].nunique()}, "
                f"rows={len(split_df)}"
            )
            print(" ", out_csv)

    all_df = pd.DataFrame(all_rows)

    manifest_path = output_root / "stability_manifest.csv"
    all_df.to_csv(manifest_path, index=False)

    print("\n========================================")
    print("[Saved manifest]", manifest_path)
    print("base images:", all_df["sample_id"].nunique())
    print("variant rows:", len(all_df))
    print("\nTransform counts:")
    print(all_df["transform"].value_counts())
    print("\nLabel counts by base image:")
    print(
        all_df[
            all_df["transform"] == "clean"
        ]["label"].value_counts().sort_index()
    )

    if failed:
        print("\nMissing files:", len(failed))
        for path in failed[:20]:
            print(path)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
