#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFilter, ImageOps


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
        small = clean.resize(
            (112, 112),
            BICUBIC,
        )
        return small.resize(
            (224, 224),
            BICUBIC,
        )

    if transform == "jpeg_q75":
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
        help="Comma-separated sources or all",
    )
    parser.add_argument(
        "--max_per_label",
        type=int,
        default=0,
        help="0 means all selected rows",
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

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns: {missing}; "
            f"existing={list(df.columns)}"
        )

    if args.sources.strip().lower() != "all":
        selected_sources = [
            x.strip()
            for x in args.sources.split(",")
            if x.strip()
        ]

        df = df[
            df["source"].isin(selected_sources)
        ].copy()

    if args.max_per_label > 0:
        sampled = []

        for (
            source,
            label,
        ), group in df.groupby(
            ["source", "label"],
            sort=False,
        ):
            sampled.append(
                group.sample(
                    n=min(
                        args.max_per_label,
                        len(group),
                    ),
                    random_state=(
                        args.seed
                        + int(label)
                        + sum(ord(c) for c in source)
                    ),
                )
            )

        df = pd.concat(
            sampled,
            ignore_index=True,
        )

    all_rows = []
    missing_paths = []

    for source, group in df.groupby(
        "source",
        sort=False,
    ):
        mode = "genimage"

        group = group.sort_values(
            ["label", "subset_idx"]
        ).reset_index(drop=True)

        source_rows = []

        for local_order, row in group.iterrows():
            original_path = Path(str(row["path"]))

            if not original_path.exists():
                print("[Missing]", original_path)
                missing_paths.append(str(original_path))
                continue

            clean = load_clean_224(original_path)

            label = int(row["label"])
            label_name = (
                "fake" if label == 1 else "real"
            )

            sample_id = (
                f"{source}__{label_name}__"
                f"{int(row['subset_idx']):04d}"
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
                variant = create_variant(
                    clean,
                    transform,
                )

                if transform == "jpeg_q75":
                    output_path = (
                        sample_dir
                        / "jpeg_q75.jpg"
                    )

                    variant.save(
                        output_path,
                        format="JPEG",
                        quality=75,
                        subsampling=2,
                        optimize=False,
                    )
                else:
                    output_path = (
                        sample_dir
                        / f"{transform}.png"
                    )

                    variant.save(
                        output_path,
                        format="PNG",
                    )

                output_row = {
                    "path": str(output_path),
                    "label": label,
                    "case_name": (
                        f"{row['case_name']}"
                        f"__stability_{transform}"
                    ),
                    "source": source,
                    "split": "test",

                    "sample_id": sample_id,
                    "mode": mode,
                    "transform": transform,
                    "subset_idx": int(
                        row["subset_idx"]
                    ),
                    "orig_idx": int(
                        row["orig_idx"]
                    ),
                    "original_path": str(
                        original_path
                    ),
                    "original_case_name": str(
                        row["case_name"]
                    ),
                }

                source_rows.append(output_row)
                all_rows.append(output_row)

        source_df = pd.DataFrame(source_rows)

        output_dir = (
            split_root
            / source
            / mode
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_csv = (
            output_dir
            / "stability.csv"
        )

        source_df.to_csv(
            output_csv,
            index=False,
        )

        print(
            f"[Saved] {source}: "
            f"base={source_df['sample_id'].nunique()}, "
            f"rows={len(source_df)}"
        )
        print(" ", output_csv)

    manifest = pd.DataFrame(all_rows)

    manifest_path = (
        output_root
        / "stability_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    print("\n========================================")
    print("[Saved]", manifest_path)
    print(
        "base images:",
        manifest["sample_id"].nunique(),
    )
    print("variant rows:", len(manifest))

    print("\nTransform counts:")
    print(manifest["transform"].value_counts())

    print("\nClean label counts:")
    print(
        manifest[
            manifest["transform"] == "clean"
        ]["label"].value_counts().sort_index()
    )

    if missing_paths:
        raise RuntimeError(
            f"Missing image paths: {len(missing_paths)}"
        )


if __name__ == "__main__":
    main()
