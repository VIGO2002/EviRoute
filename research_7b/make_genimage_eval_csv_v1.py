#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import random
from glob import glob

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".PNG", ".JPG", ".JPEG"]


def find_files(d):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob(os.path.join(d, f"*{ext}")))
    return sorted(files)


def find_val_dirs(gen_root):
    """
    Find */val/ai and */val/nature under one generator root.
    """
    ai_dirs = []
    nature_dirs = []

    for root, dirs, files in os.walk(gen_root):
        norm = root.replace("\\", "/")
        if norm.endswith("/val/ai"):
            ai_dirs.append(root)
        if norm.endswith("/val/nature"):
            nature_dirs.append(root)

    if len(ai_dirs) == 0 or len(nature_dirs) == 0:
        raise RuntimeError(f"Cannot find val/ai or val/nature under {gen_root}")

    return sorted(ai_dirs)[0], sorted(nature_dirs)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/GenImage_extracted")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--n_per_class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    gens = [
        "ADM",
        "BigGAN",
        "Midjourney",
        "VQDM",
        "glide",
        "stable_diffusion_v_1_4",
        "stable_diffusion_v_1_5",
        "wukong",
    ]

    rng = random.Random(args.seed)
    rows = []

    for gen in gens:
        gen_root = os.path.join(args.root, gen)
        if not os.path.isdir(gen_root):
            print(f"[Skip] missing: {gen_root}")
            continue

        ai_dir, nature_dir = find_val_dirs(gen_root)

        ai_files = find_files(ai_dir)
        nature_files = find_files(nature_dir)

        print(f"[{gen}] ai={len(ai_files)} nature={len(nature_files)}")
        print(f"  ai_dir={ai_dir}")
        print(f"  nature_dir={nature_dir}")

        if len(ai_files) == 0 or len(nature_files) == 0:
            print(f"[Skip] empty files: {gen}")
            continue

        n_ai = min(args.n_per_class, len(ai_files))
        n_real = min(args.n_per_class, len(nature_files))

        ai_sel = rng.sample(ai_files, n_ai)
        real_sel = rng.sample(nature_files, n_real)

        for p in real_sel:
            rows.append({
                "path": p,
                "label": 0,
                "source": gen,
                "class_name": "nature",
                "split": "val",
            })

        for p in ai_sel:
            rows.append({
                "path": p,
                "label": 1,
                "source": gen,
                "class_name": "ai",
                "split": "val",
            })

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Saved] {args.output_csv}")
    print(f"[Total] {len(rows)} images")


if __name__ == "__main__":
    main()
