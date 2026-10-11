#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json
import re
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


GRID = 14
TOPK = 10

TRAIN_SOURCE = "stable_diffusion_v_1_4"

PER_SAMPLE = Path(
    "eval/ablation_v2b_components_seed42/"
    "genimage_full/genimage_ablation_per_sample.csv.gz"
)

FEATURE_ROOT = Path(
    "features/genimage_full_mid20_shards"
)

A0_CKPT = Path(
    "runs/ablation_v2b_components_seed42/"
    "A0_samecode_naive_auc/"
    "mid20_edge_debiased_mil_best.pt"
)

A4_CKPT = Path(
    "runs/mid20_edgemask_edgepen_v2b_lite_genimage_smoke2k/"
    "mid20_edge_debiased_mil_best.pt"
)


class TopKMILScorer(nn.Module):
    def __init__(
        self,
        feat_dim=4096,
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

    def forward(self, x):
        return self.scorer(x).squeeze(-1)


def load_model(path):
    ckpt = torch.load(
        path,
        map_location="cpu",
    )

    a = ckpt["args"]

    model = TopKMILScorer(
        feat_dim=int(
            a.get("feat_dim", 4096)
        ),
        hidden_dim=int(
            a.get("hidden_dim", 256)
        ),
        dropout=float(
            a.get("dropout", 0.1)
        ),
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    return (
        model,
        ckpt["mu"].float(),
        ckpt["std"].float(),
    )


@torch.no_grad()
def infer_top10(
    feat,
    model,
    mu,
    std,
):
    """
    Exact formal-evaluation behavior.

    Important:
    - Image logit follows the prediction path:
        torch.topk(...).values.mean()
    - Evidence ordering follows the formal spatial audit:
        np.argsort(-scores)[:10]

    These are normally identical. Using NumPy argsort for evidence
    indices also reproduces the exact ordering convention used by
    edge_metrics_for_group() in the formal evaluation script.
    """

    # Exact formal-evaluation normalization
    x = (
        feat.float()
        - mu.view(1, -1)
    ) / std.view(1, -1)

    scores = model(x)  # [196]

    # --------------------------------------------------------
    # Prediction path:
    # formal Top-k MIL logit
    # --------------------------------------------------------

    pool_vals = torch.topk(
        scores,
        k=TOPK,
        largest=True,
        sorted=True,
    ).values

    image_logit = float(
        pool_vals.mean()
    )

    # --------------------------------------------------------
    # Evidence ordering:
    # match formal edge_metrics_for_group exactly
    # --------------------------------------------------------

    scores_np = (
        scores.detach()
        .cpu()
        .numpy()
    )

    evidence_idx = np.argsort(
        -scores_np
    )[:TOPK]

    evidence_scores = (
        scores_np[
            evidence_idx
        ]
    )

    return {
        "logit":
            image_logit,

        "indices":
            evidence_idx.astype(int),

        "scores":
            evidence_scores,
    }


def edge_fractions(indices):
    rows = indices // GRID
    cols = indices % GRID

    e1 = (
        (rows == 0)
        | (rows == GRID - 1)
        | (cols == 0)
        | (cols == GRID - 1)
    )

    e2 = (
        (rows <= 1)
        | (rows >= GRID - 2)
        | (cols <= 1)
        | (cols >= GRID - 2)
    )

    return (
        float(e1.mean()),
        float(e2.mean()),
    )


def safe(x):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(x),
    )


def make_model_input_view(img):
    """
    Reproduce the exact spatial preprocessing used to extract
    the frozen Mid20 features:

        PIL RGB
        -> resize((224, 224), Image.BICUBIC)

    No crop or padding is used.
    """
    return (
        img.convert("RGB")
        .resize(
            (224, 224),
            Image.Resampling.BICUBIC,
        )
    )


def overlay(
    image,
    indices,
):
    img = image.convert("RGBA")

    layer = Image.new(
        "RGBA",
        img.size,
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(layer)

    w, h = img.size

    for rank, idx in enumerate(
        indices,
        start=1,
    ):
        row = int(idx) // GRID
        col = int(idx) % GRID

        x0 = round(
            col * w / GRID
        )
        x1 = round(
            (col + 1) * w / GRID
        )

        y0 = round(
            row * h / GRID
        )
        y1 = round(
            (row + 1) * h / GRID
        )

        alpha = (
            105
            if rank == 1
            else 55
        )

        width = (
            4
            if rank == 1
            else 2
        )

        draw.rectangle(
            [x0, y0, x1, y1],
            fill=(255, 0, 0, alpha),
            outline=(255, 0, 0, 230),
            width=width,
        )

    return Image.alpha_composite(
        img,
        layer,
    ).convert("RGB")


def triptych(
    original,
    naive,
    evi,
):
    display_size = 448

    panels = []

    for img in [
        original,
        naive,
        evi,
    ]:
        panels.append(
            img.resize(
                (display_size, display_size),
                Image.Resampling.BICUBIC,
            )
        )

    out = Image.new(
        "RGB",
        (
            display_size * 3,
            display_size,
        ),
        "white",
    )

    out.paste(
        panels[0],
        (0, 0),
    )

    out.paste(
        panels[1],
        (display_size, 0),
    )

    out.paste(
        panels[2],
        (display_size * 2, 0),
    )

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help=(
            "One or more explicit generator sources. "
            "If omitted, automatic source selection is used."
        ),
    )

    ap.add_argument(
        "--n-per-source",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    ap.add_argument(
        "--out",
        default=(
            "paper_assets/"
            "figure1_overlay_candidates"
        ),
    )

    args = ap.parse_args()

    df = pd.read_csv(
        PER_SAMPLE
    )

    # --------------------------------------------------------
    # Fixed reproducible candidate protocol
    # --------------------------------------------------------

    x = df[
        (df["label"] == 1)
        & (
            df["source"]
            != TRAIN_SOURCE
        )
        & (df["A0_prob"] >= 0.5)
        & (df["A4_prob"] >= 0.5)
    ].copy()

    counts = (
        x.groupby("source")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print("=" * 100)
    print("COMMON-CORRECT UNSEEN FAKE COUNTS")
    print("=" * 100)
    print(counts.to_string())

    if args.sources is None:
        valid = counts[
            counts >= args.n_per_source
        ]

        sources = (
            valid.index[:2]
            .tolist()
        )
    else:
        sources = args.sources

    print(
        "\nSelected sources:",
        sources,
    )

    parts = []

    for source in sources:
        q = x[
            x["source"] == source
        ].copy()

        q = q.sample(
            n=args.n_per_source,
            random_state=args.seed,
            replace=False,
        )

        # Preserve random-sample order explicitly.
        q["candidate_rank"] = np.arange(
            1,
            len(q) + 1,
        )

        parts.append(q)

    selected = pd.concat(
        parts,
        ignore_index=True,
    )

    # Memory-safe processing order:
    # group samples from the same ~3.2GB shard together.
    # candidate_rank is preserved as the original random-sample rank.
    selected = selected.sort_values(
        ["source", "shard", "local_idx"]
    ).reset_index(drop=True)

    a0_model, a0_mu, a0_std = (
        load_model(A0_CKPT)
    )

    a4_model, a4_mu, a4_std = (
        load_model(A4_CKPT)
    )

    out_root = Path(args.out)

    if out_root.exists():
        # Do not mix v1 partial output with v2.
        import shutil
        shutil.rmtree(out_root)

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep only ONE feature shard in RAM at a time.
    # A single val.pt is ~3.2 GB, so caching multiple shards
    # can trigger the Linux OOM killer.
    current_pt = None
    current_payload = None

    records = []

    for _, r in selected.iterrows():
        source = str(
            r["source"]
        )

        shard = str(
            r["shard"]
        )

        idx = int(
            r["local_idx"]
        )

        rank = int(
            r["candidate_rank"]
        )

        pt = (
            FEATURE_ROOT
            / source
            / shard
            / "val.pt"
        )

        if current_pt != pt:

            # Release the previous ~3.2 GB shard before loading
            # the next one.
            if current_payload is not None:
                del current_payload
                current_payload = None

            gc.collect()

            print(
                "\nLoading:",
                pt,
            )

            current_payload = torch.load(
                pt,
                map_location="cpu",
            )

            current_pt = pt

        payload = current_payload

        # ----------------------------------------------------
        # Strict indexing checks
        # ----------------------------------------------------

        assert (
            str(
                payload["sources"][idx]
            )
            == source
        )

        assert (
            int(
                payload["labels"][idx]
            )
            == 1
        )

        feat = (
            payload[
                "patch_features"
            ][idx]
        )

        assert tuple(
            feat.shape
        ) == (196, 4096)

        path = Path(
            payload["paths"][idx]
        )

        assert path.exists(), path

        # ----------------------------------------------------
        # Exact formal inference
        # ----------------------------------------------------

        a0 = infer_top10(
            feat,
            a0_model,
            a0_mu,
            a0_std,
        )

        a4 = infer_top10(
            feat,
            a4_model,
            a4_mu,
            a4_std,
        )

        err0 = abs(
            a0["logit"]
            - float(r["A0_logit"])
        )

        err4 = abs(
            a4["logit"]
            - float(r["A4_logit"])
        )

        if err0 >= 5e-4:
            raise RuntimeError(
                f"A0 logit mismatch: "
                f"{source}/{shard}/{idx}, "
                f"err={err0}"
            )

        if err4 >= 5e-4:
            raise RuntimeError(
                f"A4 logit mismatch: "
                f"{source}/{shard}/{idx}, "
                f"err={err4}"
            )

        # top1 should reproduce frozen evidence index
        if int(
            a0["indices"][0]
        ) != int(
            r[
                "A0_class_aligned_top1"
            ]
        ):
            raise RuntimeError(
                "A0 top1 mismatch"
            )

        if int(
            a4["indices"][0]
        ) != int(
            r[
                "A4_class_aligned_top1"
            ]
        ):
            raise RuntimeError(
                "A4 top1 mismatch"
            )

        e10, e20 = edge_fractions(
            a0["indices"]
        )

        e14, e24 = edge_fractions(
            a4["indices"]
        )

        # ----------------------------------------------------
        # Top-10 edge statistics are diagnostic only.
        #
        # The frozen ablation CSV contains derived per-sample
        # Top-10 fractions, but the original formal per-sample
        # exporter did not preserve Top-10 indices. Tiny numerical
        # differences around the rank-10 cutoff can therefore
        # change one discrete patch while leaving logit/top1
        # effectively identical.
        #
        # For overlay validity we strictly require:
        #   1) same feature/sample
        #   2) frozen logit reproduction
        #   3) frozen top1 reproduction
        #
        # We record Top-10 fraction agreement instead of treating
        # it as an absolute per-sample identity check.
        # ----------------------------------------------------

        frozen_e10 = float(
            r["A0_top10_edge1_fraction"]
        )
        frozen_e20 = float(
            r["A0_top10_edge2_fraction"]
        )
        frozen_e14 = float(
            r["A4_top10_edge1_fraction"]
        )
        frozen_e24 = float(
            r["A4_top10_edge2_fraction"]
        )

        de10 = abs(e10 - frozen_e10)
        de20 = abs(e20 - frozen_e20)
        de14 = abs(e14 - frozen_e14)
        de24 = abs(e24 - frozen_e24)

        top10_edge_exact = (
            de10 < 1e-8
            and de20 < 1e-8
            and de14 < 1e-8
            and de24 < 1e-8
        )

        if not top10_edge_exact:
            print(
                "  [Top10 diagnostic mismatch] "
                f"A0 E1 {frozen_e10:.1f}->{e10:.1f}, "
                f"E2 {frozen_e20:.1f}->{e20:.1f}; "
                f"A4 E1 {frozen_e14:.1f}->{e14:.1f}, "
                f"E2 {frozen_e24:.1f}->{e24:.1f}"
            )

        # ----------------------------------------------------
        # Candidate visualization
        # ----------------------------------------------------

        original_native = (
            Image.open(path)
            .convert("RGB")
        )

        view = make_model_input_view(
            original_native
        )

        naive = overlay(
            view,
            a0["indices"],
        )

        evi = overlay(
            view,
            a4["indices"],
        )

        comparison = triptych(
            view,
            naive,
            evi,
        )

        d = (
            out_root
            / safe(source)
        )

        d.mkdir(
            parents=True,
            exist_ok=True,
        )

        stem = (
            f"{rank:02d}_"
            f"{safe(shard)}_"
            f"{idx:04d}"
        )

        f_orig = (
            d / f"{stem}_original.png"
        )

        f_a0 = (
            d / f"{stem}_naive.png"
        )

        f_a4 = (
            d / f"{stem}_eviroute.png"
        )

        f_cmp = (
            d / f"{stem}_comparison.png"
        )

        view.resize(
            (448, 448),
            Image.Resampling.BICUBIC,
        ).save(f_orig)

        naive.resize(
            (448, 448),
            Image.Resampling.BICUBIC,
        ).save(f_a0)

        evi.resize(
            (448, 448),
            Image.Resampling.BICUBIC,
        ).save(f_a4)

        comparison.save(f_cmp)

        print(
            f"{source:28s} "
            f"{rank:02d} "
            f"A0err={err0:.2e} "
            f"A4err={err4:.2e} "
            f"A0E1={e10:.2f} "
            f"A4E1={e14:.2f}"
        )

        records.append({
            "dataset":
                "GenImage",

            "source":
                source,

            "shard":
                shard,

            "local_idx":
                idx,

            "candidate_rank":
                rank,

            "seed":
                args.seed,

            "image_path":
                str(path),

            "A0_prob":
                float(r["A0_prob"]),

            "A4_prob":
                float(r["A4_prob"]),

            "A0_frozen_logit":
                float(r["A0_logit"]),

            "A0_recomputed_logit":
                a0["logit"],

            "A0_logit_error":
                err0,

            "A4_frozen_logit":
                float(r["A4_logit"]),

            "A4_recomputed_logit":
                a4["logit"],

            "A4_logit_error":
                err4,

            "A0_top10":
                " ".join(
                    map(
                        str,
                        a0["indices"],
                    )
                ),

            "A4_top10":
                " ".join(
                    map(
                        str,
                        a4["indices"],
                    )
                ),

            "A0_edge1":
                e10,

            "A4_edge1":
                e14,

            "A0_edge2":
                e20,

            "A4_edge2":
                e24,

            "A0_frozen_edge1":
                frozen_e10,

            "A0_frozen_edge2":
                frozen_e20,

            "A4_frozen_edge1":
                frozen_e14,

            "A4_frozen_edge2":
                frozen_e24,

            "top10_edge_exact_match":
                bool(top10_edge_exact),

            "original_file":
                str(f_orig),

            "naive_file":
                str(f_a0),

            "eviroute_file":
                str(f_a4),

            "comparison_file":
                str(f_cmp),
        })

    # Final release of the last large shard.
    if current_payload is not None:
        del current_payload
        current_payload = None

    gc.collect()

    manifest = pd.DataFrame(
        records
    )

    manifest.to_csv(
        out_root
        / "selection_manifest.csv",
        index=False,
    )

    summary = (
        manifest
        .groupby("source")
        .agg(
            n=(
                "local_idx",
                "size",
            ),
            A0_edge1=(
                "A0_edge1",
                "mean",
            ),
            A4_edge1=(
                "A4_edge1",
                "mean",
            ),
            A0_edge2=(
                "A0_edge2",
                "mean",
            ),
            A4_edge2=(
                "A4_edge2",
                "mean",
            ),
            max_A0_logit_error=(
                "A0_logit_error",
                "max",
            ),
            max_A4_logit_error=(
                "A4_logit_error",
                "max",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        out_root
        / "selection_summary.csv",
        index=False,
    )

    protocol = {
        "dataset":
            "GenImage",

        "subset":
            "fake",

        "training_source_excluded":
            TRAIN_SOURCE,

        "common_correct":
            "A0_prob >= 0.5 and A4_prob >= 0.5",

        "source_selection":
            "top two unseen sources by common-correct candidate count",

        "sample_selection":
            "uniform random within each source",

        "seed":
            args.seed,

        "n_per_source":
            args.n_per_source,

        "selected_sources":
            sources,

        "topk":
            TOPK,

        "feature_grid":
            "14x14",

        "evidence_semantics":
            "detector-faithful evidence proposals; not ground-truth artifact localization",
    }

    with (
        out_root
        / "selection_protocol.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            protocol,
            f,
            indent=2,
        )

    print()
    print("=" * 100)
    print("EXPORT COMPLETE")
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
        )
    )

    print(
        "\nOutput:",
        out_root,
    )


if __name__ == "__main__":
    main()
