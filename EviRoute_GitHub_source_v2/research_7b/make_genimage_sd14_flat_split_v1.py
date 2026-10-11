#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random
import re
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(root, name))
    return sorted(paths)


def infer_class_from_filename(path):
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]

    # ImageNet synset style, e.g. n01440764_xxx
    m = re.match(r"^(n\d{8})[_\-\.]", stem)
    if m:
        return m.group(1)

    # fallback: use the first token if it looks nontrivial
    parts = re.split(r"[_\-\.]", stem)
    if len(parts) >= 2 and len(parts[0]) > 2:
        return parts[0]

    return "unknown"


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
        stem = os.path.splitext(os.path.basename(p))[0]
        cls = infer_class_from_filename(p)
        rows.append({
            "path": os.path.abspath(p),
            "label": int(label),
            "split": split,
            "source": source,
            "class_name": cls,
            "case_name": f"GenImageSD14__{source}__{cls}__{stem}",
        })


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True)
    ap.add_argument("--real_dirname", default="nature")
    ap.add_argument("--fake_dirname", default="ai")
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--max_per_label",
        type=int,
        default=0,
        help="0 means use all images per label",
    )

    args = ap.parse_args()

    real_dir = os.path.join(args.root, args.real_dirname)
    fake_dir = os.path.join(args.root, args.fake_dirname)

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

    if args.max_per_label and args.max_per_label > 0:
        n = min(n, args.max_per_label)

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
    print("Total:", len(df))

    print("\nBy split/label:")
    print(df.groupby(["split", "label"]).size())

    print("\nBy split/source:")
    print(df.groupby(["split", "source"]).size())

    print("\nClass name examples:")
    print(df["class_name"].value_counts().head(20))

    print("\nNum unique inferred classes:", df["class_name"].nunique())


if __name__ == "__main__":
    main()
