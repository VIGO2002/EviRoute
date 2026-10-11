#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(
    "eval/ablation_v2b_components_seed42"
)
OUT_DIR = ROOT / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "A0",
    "A1",
    "A2",
    "A3",
    "A5",
    "A4",
]

COMPONENTS = {
    "A0": {
        "dropout": False,
        "penalty": False,
        "selection": False,
    },
    "A1": {
        "dropout": True,
        "penalty": False,
        "selection": False,
    },
    "A2": {
        "dropout": False,
        "penalty": True,
        "selection": False,
    },
    "A3": {
        "dropout": True,
        "penalty": True,
        "selection": False,
    },
    "A5": {
        "dropout": False,
        "penalty": False,
        "selection": True,
    },
    "A4": {
        "dropout": True,
        "penalty": True,
        "selection": True,
    },
}

DATASETS = {
    "GenImage": {
        "slug": "genimage",
        "summary_main": (
            ROOT
            / "genimage_full"
            / "genimage_ablation_summary.csv"
        ),
        "group_main": (
            ROOT
            / "genimage_full"
            / "genimage_ablation_by_group.csv"
        ),
        "per_sample_main": (
            ROOT
            / "genimage_full"
            / "genimage_ablation_per_sample.csv.gz"
        ),
        "summary_a5": (
            ROOT
            / "A5_selection_only"
            / "genimage_full"
            / "genimage_ablation_summary.csv"
        ),
        "group_a5": (
            ROOT
            / "A5_selection_only"
            / "genimage_full"
            / "genimage_ablation_by_group.csv"
        ),
        "per_sample_a5": (
            ROOT
            / "A5_selection_only"
            / "genimage_full"
            / "genimage_ablation_per_sample.csv.gz"
        ),
        "stratum_scope": "source",
    },
    "AIGI-Now": {
        "slug": "aiginow",
        "summary_main": (
            ROOT
            / "aiginow_full"
            / "aiginow_ablation_summary.csv"
        ),
        "group_main": (
            ROOT
            / "aiginow_full"
            / "aiginow_ablation_by_group.csv"
        ),
        "per_sample_main": (
            ROOT
            / "aiginow_full"
            / "aiginow_ablation_per_sample.csv.gz"
        ),
        "summary_a5": (
            ROOT
            / "A5_selection_only"
            / "aiginow_full"
            / "aiginow_ablation_summary.csv"
        ),
        "group_a5": (
            ROOT
            / "A5_selection_only"
            / "aiginow_full"
            / "aiginow_ablation_by_group.csv"
        ),
        "per_sample_a5": (
            ROOT
            / "A5_selection_only"
            / "aiginow_full"
            / "aiginow_ablation_per_sample.csv.gz"
        ),
        "stratum_scope": "source_mode",
    },
}

METRICS = {
    "auc": "higher",
    "ap": "higher",
    "acc_thr05": "higher",
    "best_acc": "higher",
    "fake_top1_edge1": "lower",
    "fake_top1_edge2": "lower",
    "fake_top10_edge1_fraction": "lower",
    "fake_top10_edge2_fraction": "lower",
}

COMPARISONS = [
    ("A1", "A0", "Dropout only"),
    ("A2", "A0", "Penalty only"),
    ("A3", "A1", "Penalty added to dropout"),
    ("A3", "A0", "Training constraints"),
    ("A5", "A0", "Selection only"),
    ("A4", "A3", "Selection after constraints"),
    ("A4", "A5", "Constraints plus selection"),
    ("A4", "A0", "Full method"),
]

N_BOOT = 20000
SEED = 123


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def add_components(frame):
    frame = frame.copy()

    frame["dropout"] = frame["model"].map(
        lambda x: COMPONENTS[x]["dropout"]
    )
    frame["penalty"] = frame["model"].map(
        lambda x: COMPONENTS[x]["penalty"]
    )
    frame["selection"] = frame["model"].map(
        lambda x: COMPONENTS[x]["selection"]
    )

    frame["_order"] = frame["model"].map(
        {
            model: index
            for index, model in enumerate(MODEL_ORDER)
        }
    )

    frame = frame.sort_values(
        "_order"
    ).drop(columns="_order")

    return frame


def bootstrap_mean_ci(
    values,
    n_boot=N_BOOT,
    seed=SEED,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        len(values),
        size=(n_boot, len(values)),
    )

    bootstrap_means = values[
        indices
    ].mean(axis=1)

    return (
        float(
            np.percentile(
                bootstrap_means,
                2.5,
            )
        ),
        float(
            np.percentile(
                bootstrap_means,
                97.5,
            )
        ),
        float(
            np.mean(bootstrap_means > 0)
        ),
    )


def merge_per_sample(
    dataset_name,
    config,
):
    main_path = config["per_sample_main"]
    a5_path = config["per_sample_a5"]

    require_file(main_path)
    require_file(a5_path)

    main = pd.read_csv(main_path)
    a5 = pd.read_csv(a5_path)

    keys = [
        "dataset",
        "source",
        "mode",
        "shard",
        "local_idx",
        "label",
    ]

    missing_main = [
        key for key in keys
        if key not in main.columns
    ]
    missing_a5 = [
        key for key in keys
        if key not in a5.columns
    ]

    if missing_main:
        raise KeyError(
            f"{dataset_name} main per-sample "
            f"missing keys: {missing_main}"
        )

    if missing_a5:
        raise KeyError(
            f"{dataset_name} A5 per-sample "
            f"missing keys: {missing_a5}"
        )

    a5_columns = [
        column
        for column in a5.columns
        if column.startswith("A5_")
    ]

    if not a5_columns:
        raise KeyError(
            f"No A5 columns found in {a5_path}"
        )

    a5_subset = a5[
        keys + a5_columns
    ].copy()

    merged = main.merge(
        a5_subset,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(main):
        raise RuntimeError(
            f"{dataset_name}: per-sample merge "
            f"lost rows: {len(main)} -> {len(merged)}"
        )

    output_path = (
        OUT_DIR
        / (
            f"{config['slug']}_"
            "ablation_per_sample_A0_A5.csv.gz"
        )
    )

    merged.to_csv(
        output_path,
        index=False,
        compression="gzip",
    )

    print(
        f"[Saved] {output_path} "
        f"rows={len(merged)}"
    )


def load_all_results():
    summary_frames = []
    group_frames = []

    for dataset_name, config in DATASETS.items():
        for path in [
            config["summary_main"],
            config["summary_a5"],
            config["group_main"],
            config["group_a5"],
        ]:
            require_file(path)

        main_summary = pd.read_csv(
            config["summary_main"]
        )
        a5_summary = pd.read_csv(
            config["summary_a5"]
        )

        summary = pd.concat(
            [main_summary, a5_summary],
            ignore_index=True,
        )

        summary = (
            summary
            .drop_duplicates(
                subset=["model"],
                keep="last",
            )
        )

        found_models = set(
            summary["model"]
        )

        expected_models = set(
            MODEL_ORDER
        )

        if found_models != expected_models:
            raise RuntimeError(
                f"{dataset_name}: model mismatch\n"
                f"expected={expected_models}\n"
                f"found={found_models}"
            )

        summary["dataset_name"] = (
            dataset_name
        )
        summary["dataset_slug"] = (
            config["slug"]
        )

        summary = add_components(summary)
        summary_frames.append(summary)

        main_group = pd.read_csv(
            config["group_main"]
        )
        a5_group = pd.read_csv(
            config["group_a5"]
        )

        group = pd.concat(
            [main_group, a5_group],
            ignore_index=True,
        )

        group["dataset_name"] = (
            dataset_name
        )
        group["dataset_slug"] = (
            config["slug"]
        )

        group_frames.append(group)

        merge_per_sample(
            dataset_name,
            config,
        )

    summary_all = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    group_all = pd.concat(
        group_frames,
        ignore_index=True,
    )

    return summary_all, group_all


def create_master_tables(summary_all):
    master_columns = [
        "dataset_name",
        "model",
        "dropout",
        "penalty",
        "selection",
        "auc",
        "ap",
        "acc_thr05",
        "best_acc",
        "fake_top1_edge1",
        "fake_top1_edge2",
        "fake_top10_edge1_fraction",
        "fake_top10_edge2_fraction",
        "source_macro_auc",
        "source_macro_ap",
    ]

    master = summary_all[
        master_columns
    ].copy()

    master_path = (
        OUT_DIR
        / "component_ablation_master.csv"
    )

    master.to_csv(
        master_path,
        index=False,
    )

    print("[Saved]", master_path)

    rows = []

    for model in MODEL_ORDER:
        row = {
            "Model": model,
            "Drop.": (
                "Yes"
                if COMPONENTS[model]["dropout"]
                else "No"
            ),
            "Pen.": (
                "Yes"
                if COMPONENTS[model]["penalty"]
                else "No"
            ),
            "Sel.": (
                "Yes"
                if COMPONENTS[model]["selection"]
                else "No"
            ),
        }

        for dataset_name, prefix in [
            ("GenImage", "G"),
            ("AIGI-Now", "A"),
        ]:
            matched = summary_all[
                (summary_all["dataset_name"]
                 == dataset_name)
                & (summary_all["model"] == model)
            ]

            if len(matched) != 1:
                raise RuntimeError(
                    f"Expected one row for "
                    f"{dataset_name}/{model}"
                )

            values = matched.iloc[0]

            row[f"{prefix}-AUC"] = (
                100.0 * values["auc"]
            )
            row[f"{prefix}-AP"] = (
                100.0 * values["ap"]
            )
            row[f"{prefix}-BestAcc"] = (
                100.0 * values["best_acc"]
            )
            row[f"{prefix}-E1"] = (
                100.0
                * values["fake_top1_edge1"]
            )
            row[f"{prefix}-E2"] = (
                100.0
                * values["fake_top1_edge2"]
            )
            row[f"{prefix}-T10E1"] = (
                100.0
                * values[
                    "fake_top10_edge1_fraction"
                ]
            )
            row[f"{prefix}-T10E2"] = (
                100.0
                * values[
                    "fake_top10_edge2_fraction"
                ]
            )

        rows.append(row)

    wide = pd.DataFrame(rows)

    wide_path = (
        OUT_DIR
        / "component_ablation_main_wide.csv"
    )

    wide.to_csv(
        wide_path,
        index=False,
    )

    print("[Saved]", wide_path)

    # ICASSP 正文紧凑表：
    # 每个数据集保留 AUC、Top-1 E1、Top-10 E1。
    paper = wide[
        [
            "Model",
            "Drop.",
            "Pen.",
            "Sel.",
            "G-AUC",
            "G-E1",
            "G-T10E1",
            "A-AUC",
            "A-E1",
            "A-T10E1",
        ]
    ].copy()

    paper = paper.rename(
        columns={
            "G-AUC": "G AUC",
            "G-E1": "G E1",
            "G-T10E1": "G T10-E1",
            "A-AUC": "A AUC",
            "A-E1": "A E1",
            "A-T10E1": "A T10-E1",
        }
    )

    for column in [
        "G AUC",
        "G E1",
        "G T10-E1",
        "A AUC",
        "A E1",
        "A T10-E1",
    ]:
        paper[column] = paper[
            column
        ].map(lambda x: f"{x:.2f}")

    for column in [
        "Drop.",
        "Pen.",
        "Sel.",
    ]:
        paper[column] = paper[
            column
        ].map(
            lambda x: (
                r"$\checkmark$"
                if x == "Yes"
                else "--"
            )
        )

    latex_path = (
        OUT_DIR
        / "component_ablation_paper_table.tex"
    )

    latex = paper.to_latex(
        index=False,
        escape=False,
        column_format="lcccrrrrrr",
        caption=(
            "Component ablation on GenImage "
            "and AIGI-Now. E1 and T10-E1 "
            "denote fake-sample Top-1 and "
            "Top-10 outer-ring rates (\\%)."
        ),
        label="tab:component_ablation",
    )

    latex_path.write_text(
        latex,
        encoding="utf-8",
    )

    print("[Saved]", latex_path)

    print(
        "\n========== Final wide table =========="
    )
    print(
        wide.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    return master, wide


def paired_stratum_bootstrap(
    summary_all,
    group_all,
):
    bootstrap_rows = []
    stratum_delta_rows = []

    for dataset_name, config in DATASETS.items():
        dataset_group = group_all[
            group_all["dataset_name"]
            == dataset_name
        ].copy()

        scope = config["stratum_scope"]

        dataset_group = dataset_group[
            dataset_group["scope_type"]
            == scope
        ].copy()

        if dataset_name == "GenImage":
            dataset_group["stratum"] = (
                dataset_group["source"]
                .astype(str)
            )
        else:
            dataset_group["stratum"] = (
                dataset_group["source"]
                .astype(str)
                + "||"
                + dataset_group["mode"]
                .astype(str)
            )

        expected_strata = (
            8 if dataset_name == "GenImage"
            else 18
        )

        actual_strata = dataset_group[
            "stratum"
        ].nunique()

        if actual_strata != expected_strata:
            raise RuntimeError(
                f"{dataset_name}: expected "
                f"{expected_strata} strata, "
                f"found {actual_strata}"
            )

        dataset_summary = summary_all[
            summary_all["dataset_name"]
            == dataset_name
        ].set_index("model")

        for challenger, reference, description in COMPARISONS:
            for metric_id, (
                metric,
                direction,
            ) in enumerate(METRICS.items()):

                pivot = dataset_group.pivot_table(
                    index="stratum",
                    columns="model",
                    values=metric,
                    aggfunc="first",
                )

                if (
                    challenger not in pivot.columns
                    or reference not in pivot.columns
                ):
                    raise KeyError(
                        f"{dataset_name}/{metric}: "
                        f"missing {challenger} or "
                        f"{reference}"
                    )

                challenger_values = pd.to_numeric(
                    pivot[challenger],
                    errors="coerce",
                )

                reference_values = pd.to_numeric(
                    pivot[reference],
                    errors="coerce",
                )

                valid = (
                    challenger_values.notna()
                    & reference_values.notna()
                )

                challenger_values = (
                    challenger_values[valid]
                )
                reference_values = (
                    reference_values[valid]
                )

                if direction == "higher":
                    advantage = (
                        challenger_values
                        - reference_values
                    )
                else:
                    # 正数仍表示 challenger 更好，
                    # 即边缘率更低。
                    advantage = (
                        reference_values
                        - challenger_values
                    )

                ci_low, ci_high, prob_positive = (
                    bootstrap_mean_ci(
                        advantage.to_numpy(),
                        seed=(
                            SEED
                            + 1000
                            * list(DATASETS).index(
                                dataset_name
                            )
                            + 100
                            * COMPARISONS.index(
                                (
                                    challenger,
                                    reference,
                                    description,
                                )
                            )
                            + metric_id
                        ),
                    )
                )

                challenger_global = float(
                    dataset_summary.loc[
                        challenger,
                        metric,
                    ]
                )
                reference_global = float(
                    dataset_summary.loc[
                        reference,
                        metric,
                    ]
                )

                if direction == "higher":
                    global_advantage = (
                        challenger_global
                        - reference_global
                    )
                else:
                    global_advantage = (
                        reference_global
                        - challenger_global
                    )

                bootstrap_rows.append({
                    "dataset": dataset_name,
                    "challenger": challenger,
                    "reference": reference,
                    "comparison":
                        f"{challenger}_vs_{reference}",
                    "description": description,
                    "metric": metric,
                    "direction": direction,
                    "n_strata": int(
                        len(advantage)
                    ),
                    "reference_macro_mean": float(
                        reference_values.mean()
                    ),
                    "challenger_macro_mean": float(
                        challenger_values.mean()
                    ),
                    "macro_advantage_mean": float(
                        advantage.mean()
                    ),
                    "macro_advantage_ci95_low":
                        ci_low,
                    "macro_advantage_ci95_high":
                        ci_high,
                    "bootstrap_prob_advantage_gt0":
                        prob_positive,
                    "global_reference":
                        reference_global,
                    "global_challenger":
                        challenger_global,
                    "global_advantage":
                        global_advantage,
                })

                for stratum, value in (
                    advantage.items()
                ):
                    stratum_delta_rows.append({
                        "dataset": dataset_name,
                        "comparison":
                            f"{challenger}_vs_{reference}",
                        "challenger": challenger,
                        "reference": reference,
                        "description": description,
                        "metric": metric,
                        "stratum": stratum,
                        "advantage": float(value),
                    })

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    stratum_deltas = pd.DataFrame(
        stratum_delta_rows
    )

    bootstrap_path = (
        OUT_DIR
        / "component_ablation_"
        "paired_stratum_bootstrap.csv"
    )

    delta_path = (
        OUT_DIR
        / "component_ablation_"
        "per_stratum_deltas.csv"
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    stratum_deltas.to_csv(
        delta_path,
        index=False,
    )

    print("[Saved]", bootstrap_path)
    print("[Saved]", delta_path)

    key_metrics = [
        "auc",
        "ap",
        "fake_top1_edge1",
        "fake_top10_edge1_fraction",
    ]

    key = bootstrap[
        bootstrap["metric"].isin(
            key_metrics
        )
    ].copy()

    # 转换成百分点，便于阅读。
    for column in [
        "global_advantage",
        "macro_advantage_mean",
        "macro_advantage_ci95_low",
        "macro_advantage_ci95_high",
    ]:
        key[column] = (
            100.0 * key[column]
        )

    key_path = (
        OUT_DIR
        / "component_ablation_"
        "paired_bootstrap_key_metrics.csv"
    )

    key.to_csv(
        key_path,
        index=False,
    )

    print("[Saved]", key_path)

    print(
        "\n========== Paired bootstrap "
        "key metrics (percentage points) =========="
    )

    display = key[
        [
            "dataset",
            "comparison",
            "metric",
            "n_strata",
            "global_advantage",
            "macro_advantage_mean",
            "macro_advantage_ci95_low",
            "macro_advantage_ci95_high",
            "bootstrap_prob_advantage_gt0",
        ]
    ]

    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    return bootstrap


def create_pareto_plots(summary_all):
    for dataset_name, config in DATASETS.items():
        subset = summary_all[
            summary_all["dataset_name"]
            == dataset_name
        ].copy()

        subset["_order"] = subset[
            "model"
        ].map(
            {
                model: index
                for index, model
                in enumerate(MODEL_ORDER)
            }
        )

        subset = subset.sort_values(
            "_order"
        )

        # Plot 1: AUC vs Top-1 Edge1
        figure = plt.figure(
            figsize=(6.2, 4.8)
        )
        axis = figure.add_subplot(111)

        x = (
            100.0
            * subset["fake_top1_edge1"]
            .to_numpy()
        )
        y = (
            100.0
            * subset["auc"]
            .to_numpy()
        )

        for model, x_value, y_value in zip(
            subset["model"],
            x,
            y,
        ):
            marker = (
                "*"
                if model == "A4"
                else "o"
            )
            size = (
                180
                if model == "A4"
                else 75
            )

            axis.scatter(
                [x_value],
                [y_value],
                marker=marker,
                s=size,
            )

            axis.annotate(
                model,
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        axis.set_xlabel(
            "Fake Top-1 Edge1 Rate (%)"
        )
        axis.set_ylabel("AUC (%)")
        axis.set_title(
            f"{dataset_name}: "
            "Detection–Edge-Bias Trade-off"
        )
        axis.grid(
            True,
            alpha=0.3,
        )

        figure.tight_layout()

        stem = (
            OUT_DIR
            / (
                f"{config['slug']}_"
                "pareto_auc_top1_edge1"
            )
        )

        figure.savefig(
            stem.with_suffix(".png"),
            dpi=300,
            bbox_inches="tight",
        )
        figure.savefig(
            stem.with_suffix(".pdf"),
            bbox_inches="tight",
        )
        plt.close(figure)

        print(
            "[Saved]",
            stem.with_suffix(".png"),
        )
        print(
            "[Saved]",
            stem.with_suffix(".pdf"),
        )

        # Plot 2: AUC vs Top-10 Edge1
        figure = plt.figure(
            figsize=(6.2, 4.8)
        )
        axis = figure.add_subplot(111)

        x = (
            100.0
            * subset[
                "fake_top10_edge1_fraction"
            ].to_numpy()
        )

        for model, x_value, y_value in zip(
            subset["model"],
            x,
            y,
        ):
            marker = (
                "*"
                if model == "A4"
                else "o"
            )
            size = (
                180
                if model == "A4"
                else 75
            )

            axis.scatter(
                [x_value],
                [y_value],
                marker=marker,
                s=size,
            )

            axis.annotate(
                model,
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        axis.set_xlabel(
            "Fake Top-10 Edge1 Fraction (%)"
        )
        axis.set_ylabel("AUC (%)")
        axis.set_title(
            f"{dataset_name}: "
            "Detection–Top-10 Bias Trade-off"
        )
        axis.grid(
            True,
            alpha=0.3,
        )

        figure.tight_layout()

        stem = (
            OUT_DIR
            / (
                f"{config['slug']}_"
                "pareto_auc_top10_edge1"
            )
        )

        figure.savefig(
            stem.with_suffix(".png"),
            dpi=300,
            bbox_inches="tight",
        )
        figure.savefig(
            stem.with_suffix(".pdf"),
            bbox_inches="tight",
        )
        plt.close(figure)

        print(
            "[Saved]",
            stem.with_suffix(".png"),
        )
        print(
            "[Saved]",
            stem.with_suffix(".pdf"),
        )


def save_manifest():
    manifest = {
        "models": MODEL_ORDER,
        "components": COMPONENTS,
        "comparisons": [
            {
                "challenger": challenger,
                "reference": reference,
                "description": description,
            }
            for challenger, reference, description
            in COMPARISONS
        ],
        "bootstrap": {
            "unit": {
                "GenImage": "generator",
                "AIGI-Now": "source-mode",
            },
            "num_strata": {
                "GenImage": 8,
                "AIGI-Now": 18,
            },
            "num_replicates": N_BOOT,
            "seed": SEED,
            "advantage_definition": (
                "Positive values always indicate "
                "that the challenger is better. "
                "For edge metrics, advantage is "
                "reference minus challenger."
            ),
        },
    }

    path = (
        OUT_DIR
        / "component_ablation_manifest.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("[Saved]", path)


def main():
    summary_all, group_all = (
        load_all_results()
    )

    summary_all.to_csv(
        OUT_DIR
        / "component_ablation_"
        "all_global_results.csv",
        index=False,
    )

    group_all.to_csv(
        OUT_DIR
        / "component_ablation_"
        "all_group_results.csv",
        index=False,
    )

    create_master_tables(
        summary_all
    )

    paired_stratum_bootstrap(
        summary_all,
        group_all,
    )

    create_pareto_plots(
        summary_all
    )

    save_manifest()

    print(
        "\n========================================"
    )
    print(
        "Component ablation finalization done."
    )
    print(
        "Output directory:",
        OUT_DIR,
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
