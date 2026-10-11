#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    paths = []
    if not os.path.isdir(folder):
        return paths

    for root, _, files in os.walk(folder):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(root, name))

    return sorted(paths)


def add_rows(rows, paths, label, generator, subset):
    source = f"{generator}__{subset}"

    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        rows.append({
            "path": os.path.abspath(p),
            "label": int(label),
            "split": "test",
            "source": source,
            "generator": generator,
            "subset": subset,
            "case_name": f"AIGINow__{source}__{label}__{stem}",
        })


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
        help="AIGI-Now root, e.g. /root/autodl-tmp/datasets/AIGI-Now/AIGI-Now",
    )
    ap.add_argument(
        "--out_csv",
        default="splits/aiginow_all_eval.csv",
    )
    ap.add_argument(
        "--subsets",
        default="pixel,semantic",
        help="comma-separated subsets: pixel,semantic",
    )
    ap.add_argument(
        "--max_per_class",
        type=int,
        default=0,
        help="0 means use all images",
    )

    args = ap.parse_args()

    subsets = [x.strip() for x in args.subsets.split(",") if x.strip()]
    rows = []

    generators = [
        x for x in sorted(os.listdir(args.root))
        if os.path.isdir(os.path.join(args.root, x))
    ]

    print("generators:", generators)

    for generator in generators:
        for subset in subsets:
            base = os.path.join(args.root, generator, subset)

            real_dir = os.path.join(base, "0_real")
            fake_dir = os.path.join(base, "1_fake")

            real_paths = list_images(real_dir)
            fake_paths = list_images(fake_dir)

            if args.max_per_class and args.max_per_class > 0:
                real_paths = real_paths[:args.max_per_class]
                fake_paths = fake_paths[:args.max_per_class]

            if len(real_paths) == 0 and len(fake_paths) == 0:
                print(f"[skip] {generator}/{subset}: no images")
                continue

            print(
                f"{generator}/{subset}: "
                f"real={len(real_paths)}, fake={len(fake_paths)}"
            )

            add_rows(rows, real_paths, 0, generator, subset)
            add_rows(rows, fake_paths, 1, generator, subset)

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print("\nSaved:", args.out_csv)
    print("Total:", len(df))

    if len(df) > 0:
        print("\nBy source/label:")
        print(df.groupby(["source", "label"]).size())


if __name__ == "__main__":
    main()
