# build_aigi_now_manifest.py

import os
import csv
import argparse
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def safe_case_name(split: str, generator: str, task_type: str, gt_label: str, rel_path: Path) -> str:
    stem = rel_path.with_suffix("").as_posix()
    stem = stem.replace("/", "__").replace("\\", "__").replace(" ", "_")
    stem = "".join(ch if ch.isalnum() or ch in "._-=" else "_" for ch in stem)
    return f"AIGINow__{split}__{generator}__{task_type}__{gt_label}__{stem}"


def parse_aigi_now_path(rel_path: Path):
    """
    期望结构：
      generator/task_type/0_real/xxx
      generator/task_type/1_fake/xxx

    返回：
      generator, task_type, gt_label
    """
    parts = rel_path.parts
    if len(parts) < 4:
        return None, None, None

    generator = parts[0]
    task_type = parts[1]
    label_dir = parts[2].lower()

    if label_dir == "0_real":
        gt_label = "Real"
    elif label_dir == "1_fake":
        gt_label = "Fake"
    else:
        return None, None, None

    return generator, task_type, gt_label


def collect_images(root_dir: Path, split: str):
    rows = []
    skipped = 0

    for p in sorted(root_dir.rglob("*")):
        if not is_image_file(p):
            continue

        rel_path = p.relative_to(root_dir)
        generator, task_type, gt_label = parse_aigi_now_path(rel_path)

        if generator is None:
            skipped += 1
            continue

        case_name = safe_case_name(
            split=split,
            generator=generator,
            task_type=task_type,
            gt_label=gt_label,
            rel_path=rel_path,
        )

        rows.append({
            "case_name": case_name,
            "dataset": "AIGI-Now",
            "split": split,
            "gt_label": gt_label,
            "image_path": str(p.resolve()),
            "relative_path": rel_path.as_posix(),

            # 与 Tiny-GenImage 保持兼容：source_type 作为主要分组字段
            "source_type": generator,

            # AIGI-Now 额外字段
            "generator": generator,
            "task_type": task_type,
        })

    if skipped > 0:
        print(f"[!] skipped images due to unmatched path structure: {skipped}")

    return rows


def write_csv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "case_name",
        "dataset",
        "split",
        "gt_label",
        "image_path",
        "relative_path",
        "source_type",
        "generator",
        "task_type",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build manifest for AIGI-Now")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Inner AIGI-Now root, e.g. /root/autodl-tmp/datasets/AIGI-Now/AIGI-Now"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name written into manifest"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="aigi_now_eval/manifest.csv"
    )
    parser.add_argument(
        "--max_per_group",
        type=int,
        default=0,
        help="Optional debug limit per generator/task_type/label. 0 means no limit."
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"[!] root_dir not found: {root_dir}")

    rows = collect_images(root_dir=root_dir, split=args.split)

    rows = sorted(rows, key=lambda x: (
        x["generator"],
        x["task_type"],
        x["gt_label"],
        x["relative_path"]
    ))

    if args.max_per_group > 0:
        limited = []
        counts = {}
        for r in rows:
            key = (r["generator"], r["task_type"], r["gt_label"])
            counts[key] = counts.get(key, 0)
            if counts[key] < args.max_per_group:
                limited.append(r)
                counts[key] += 1
        rows = limited

    write_csv(args.output_csv, rows)

    n_real = sum(1 for r in rows if r["gt_label"] == "Real")
    n_fake = sum(1 for r in rows if r["gt_label"] == "Fake")

    gen_counts = {}
    task_counts = {}
    gen_task_counts = {}

    for r in rows:
        gen_counts[r["generator"]] = gen_counts.get(r["generator"], 0) + 1
        task_counts[r["task_type"]] = task_counts.get(r["task_type"], 0) + 1
        key = f"{r['generator']}::{r['task_type']}"
        gen_task_counts[key] = gen_task_counts.get(key, 0) + 1

    print("[+] AIGI-Now manifest built")
    print(f"[*] root_dir    : {root_dir}")
    print(f"[*] split       : {args.split}")
    print(f"[*] total       : {len(rows)}")
    print(f"[*] real        : {n_real}")
    print(f"[*] fake        : {n_fake}")
    print(f"[*] output_csv  : {args.output_csv}")

    print("\n[*] generator counts:")
    for k, v in sorted(gen_counts.items(), key=lambda x: x[0]):
        print(f"    {k}: {v}")

    print("\n[*] task_type counts:")
    for k, v in sorted(task_counts.items(), key=lambda x: x[0]):
        print(f"    {k}: {v}")

    print("\n[*] generator/task_type counts:")
    for k, v in sorted(gen_task_counts.items(), key=lambda x: x[0]):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()