#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random
import pandas as pd

IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def list_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(root, name))
    return sorted(paths)


def split_items(items, train_ratio, val_ratio):
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]

    return train, val, test


def add_rows(rows, paths, label, split, source):
    for p in paths:
        case_name = os.path.splitext(os.path.basename(p))[0]
        rows.append({
            "path": os.path.abspath(p),
            "label": int(label),
            "split": split,
            "source": source,
            "case_name": f"{source}__{case_name}",
        })


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default="datasets/progan/car",
    )
    ap.add_argument(
        "--real_folder",
        default="0_real",
    )
    ap.add_argument(
        "--fake_folder",
        default="1_fake_sd14",
    )
    ap.add_argument(
        "--out_csv",
        default="splits/allpatch_sd14_car_split.csv",
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
    )
    ap.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--max_per_class",
        type=int,
        default=0,
        help="0 means use all available images",
    )

    args = ap.parse_args()

    real_dir = os.path.join(args.root, args.real_folder)
    fake_dir = os.path.join(args.root, args.fake_folder)

    real_paths = list_images(real_dir)
    fake_paths = list_images(fake_dir)

    if len(real_paths) == 0:
        raise RuntimeError(f"No real images found in {real_dir}")
    if len(fake_paths) == 0:
        raise RuntimeError(f"No fake images found in {fake_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(real_paths)
    rng.shuffle(fake_paths)

    n = min(len(real_paths), len(fake_paths))

    if args.max_per_class and args.max_per_class > 0:
        n = min(n, args.max_per_class)

    real_paths = real_paths[:n]
    fake_paths = fake_paths[:n]

    real_train, real_val, real_test = split_items(
        real_paths,
        args.train_ratio,
        args.val_ratio,
    )
    fake_train, fake_val, fake_test = split_items(
        fake_paths,
        args.train_ratio,
        args.val_ratio,
    )

    rows = []

    add_rows(rows, real_train, 0, "train", "real")
    add_rows(rows, fake_train, 1, "train", "sd14")

    add_rows(rows, real_val, 0, "val", "real")
    add_rows(rows, fake_val, 1, "val", "sd14")

    add_rows(rows, real_test, 0, "test", "real")
    add_rows(rows, fake_test, 1, "test", "sd14")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print("Saved:", args.out_csv)
    print(df.groupby(["split", "label"]).size())
    print("\nSources:")
    print(df.groupby(["split", "source"]).size())


if __name__ == "__main__":
    main()
