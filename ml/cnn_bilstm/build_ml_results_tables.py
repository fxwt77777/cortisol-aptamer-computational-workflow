from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.home() / "rna_nna_project"

BASELINE_METRICS = (
    ROOT
    / "baselines"
    / "results"
    / "baseline_metrics_test.csv"
)

CNN_MEAN_SD = (
    ROOT
    / "revised_cnn"
    / "remote_run_20260806"
    / "results"
    / "cnn_bilstm_mean_sd.csv"
)

CNN_ENSEMBLE = (
    ROOT
    / "revised_cnn"
    / "remote_run_20260806"
    / "results"
    / "cnn_bilstm_ensemble_metrics.csv"
)

CNN_ENSEMBLE_CI = (
    ROOT
    / "revised_cnn"
    / "remote_run_20260806"
    / "results"
    / "cnn_bilstm_ensemble_bootstrap_ci.csv"
)

PEARSON_FILE = (
    ROOT
    / "revised_cnn"
    / "pearson_analysis"
    / "test_pearson_all_models.csv"
)

PEARSON_SEED_SUMMARY = (
    ROOT
    / "revised_cnn"
    / "pearson_analysis"
    / "cnn_seed_pearson_mean_sd.csv"
)

PAIRED_POINT = (
    ROOT
    / "revised_cnn"
    / "paired_comparison_vs_kmer_rf"
    / "paired_point_metrics.csv"
)

PAIRED_BOOTSTRAP = (
    ROOT
    / "revised_cnn"
    / "paired_comparison_vs_kmer_rf"
    / "paired_bootstrap_summary.csv"
)

OUT_DIR = ROOT / "revised_cnn" / "final_ml_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


baseline = pd.read_csv(BASELINE_METRICS)
cnn_mean_sd = pd.read_csv(CNN_MEAN_SD)
ensemble = pd.read_csv(CNN_ENSEMBLE).iloc[0]
ensemble_ci = pd.read_csv(CNN_ENSEMBLE_CI)
pearson = pd.read_csv(PEARSON_FILE)
pearson_seed = pd.read_csv(PEARSON_SEED_SUMMARY).iloc[0]
paired_point = pd.read_csv(PAIRED_POINT)
paired_bootstrap = pd.read_csv(PAIRED_BOOTSTRAP)


model_name_map = {
    "length_gc_lr": "Length + GC LR",
    "simple_lr": "Simple-feature LR",
    "kmer_lr": "1–4-mer LR",
    "kmer_rf": "1–4-mer RF",
}

pearson_name_map = {
    "length_gc_lr": "Length_GC_LR",
    "simple_lr": "Simple_LR",
    "kmer_lr": "Kmer_LR",
    "kmer_rf": "Kmer_RF",
}


# ---------------------------------------------------------
# 1. Numeric point-estimate table
# ---------------------------------------------------------
point_rows = []

for _, row in baseline.iterrows():
    model_key = row["model"]

    pearson_row = pearson.loc[
        pearson["model"] == pearson_name_map[model_key]
    ].iloc[0]

    point_rows.append({
        "model": model_name_map[model_key],
        "model_type": "Baseline",
        "roc_auc": row["test_roc_auc"],
        "pr_auc": row["test_pr_auc"],
        "accuracy": row["test_accuracy"],
        "balanced_accuracy": row[
            "test_balanced_accuracy"
        ],
        "f1": row["test_f1"],
        "precision": row["test_precision"],
        "sensitivity": row["test_sensitivity"],
        "specificity": row["test_specificity"],
        "mcc": row["test_mcc"],
        "brier": row["test_brier"],
        "pearson_r": pearson_row["pearson_r"],
        "pearson_ci_low": pearson_row[
            "bootstrap_ci_low"
        ],
        "pearson_ci_high": pearson_row[
            "bootstrap_ci_high"
        ],
    })


ensemble_pearson = pearson.loc[
    pearson["model"] == "CNN_BiLSTM_ensemble"
].iloc[0]

point_rows.append({
    "model": "CNN–BiLSTM ensemble",
    "model_type": "Deep learning",
    "roc_auc": ensemble["test_roc_auc"],
    "pr_auc": ensemble["test_pr_auc"],
    "accuracy": ensemble["test_accuracy"],
    "balanced_accuracy": ensemble[
        "test_balanced_accuracy"
    ],
    "f1": ensemble["test_f1"],
    "precision": ensemble["test_precision"],
    "sensitivity": ensemble["test_sensitivity"],
    "specificity": ensemble["test_specificity"],
    "mcc": ensemble["test_mcc"],
    "brier": ensemble["test_brier"],
    "pearson_r": ensemble_pearson["pearson_r"],
    "pearson_ci_low": ensemble_pearson[
        "bootstrap_ci_low"
    ],
    "pearson_ci_high": ensemble_pearson[
        "bootstrap_ci_high"
    ],
})

point_table = pd.DataFrame(point_rows)


# ---------------------------------------------------------
# 2. CNN five-seed mean ± SD table
# ---------------------------------------------------------
metric_map = {
    "test_roc_auc": "ROC-AUC",
    "test_pr_auc": "PR-AUC",
    "test_accuracy": "Accuracy",
    "test_balanced_accuracy": "Balanced accuracy",
    "test_f1": "F1",
    "test_mcc": "MCC",
    "test_brier": "Brier score",
}

cnn_variability = cnn_mean_sd.loc[
    cnn_mean_sd["metric"].isin(metric_map)
].copy()

cnn_variability["metric_display"] = (
    cnn_variability["metric"].map(metric_map)
)

cnn_variability = cnn_variability[
    [
        "metric_display",
        "n_seeds",
        "mean",
        "sd",
        "minimum",
        "maximum",
    ]
].rename(
    columns={"metric_display": "metric"}
)

cnn_variability = pd.concat(
    [
        cnn_variability,
        pd.DataFrame([{
            "metric": "Pearson r",
            "n_seeds": int(
                pearson_seed["n_seeds"]
            ),
            "mean": pearson_seed["mean"],
            "sd": pearson_seed["sd"],
            "minimum": pearson_seed["min"],
            "maximum": pearson_seed["max"],
        }]),
    ],
    ignore_index=True,
)


# ---------------------------------------------------------
# 3. Manuscript-style display table
# ---------------------------------------------------------
def fmt(value):
    return f"{float(value):.3f}"


def fmt_mean_sd(mean, sd):
    return f"{float(mean):.3f} ± {float(sd):.3f}"


mean_lookup = cnn_variability.set_index("metric")

display_rows = []

for _, row in point_table.iterrows():
    display_rows.append({
        "Model": row["model"],
        "ROC-AUC": fmt(row["roc_auc"]),
        "PR-AUC": fmt(row["pr_auc"]),
        "Accuracy": fmt(row["accuracy"]),
        "Balanced accuracy": fmt(
            row["balanced_accuracy"]
        ),
        "F1": fmt(row["f1"]),
        "MCC": fmt(row["mcc"]),
        "Brier score": fmt(row["brier"]),
        "Pearson r": fmt(row["pearson_r"]),
    })


display_rows.insert(
    4,
    {
        "Model": "CNN–BiLSTM, five-seed mean",
        "ROC-AUC": fmt_mean_sd(
            mean_lookup.loc["ROC-AUC", "mean"],
            mean_lookup.loc["ROC-AUC", "sd"],
        ),
        "PR-AUC": fmt_mean_sd(
            mean_lookup.loc["PR-AUC", "mean"],
            mean_lookup.loc["PR-AUC", "sd"],
        ),
        "Accuracy": fmt_mean_sd(
            mean_lookup.loc["Accuracy", "mean"],
            mean_lookup.loc["Accuracy", "sd"],
        ),
        "Balanced accuracy": fmt_mean_sd(
            mean_lookup.loc[
                "Balanced accuracy",
                "mean",
            ],
            mean_lookup.loc[
                "Balanced accuracy",
                "sd",
            ],
        ),
        "F1": fmt_mean_sd(
            mean_lookup.loc["F1", "mean"],
            mean_lookup.loc["F1", "sd"],
        ),
        "MCC": fmt_mean_sd(
            mean_lookup.loc["MCC", "mean"],
            mean_lookup.loc["MCC", "sd"],
        ),
        "Brier score": fmt_mean_sd(
            mean_lookup.loc[
                "Brier score",
                "mean",
            ],
            mean_lookup.loc[
                "Brier score",
                "sd",
            ],
        ),
        "Pearson r": fmt_mean_sd(
            mean_lookup.loc["Pearson r", "mean"],
            mean_lookup.loc["Pearson r", "sd"],
        ),
    },
)

manuscript_table = pd.DataFrame(display_rows)


# ---------------------------------------------------------
# 4. CNN ensemble bootstrap CI table
# ---------------------------------------------------------
ensemble_point_map = {
    "roc_auc": ensemble["test_roc_auc"],
    "pr_auc": ensemble["test_pr_auc"],
    "accuracy": ensemble["test_accuracy"],
    "balanced_accuracy": ensemble[
        "test_balanced_accuracy"
    ],
    "f1": ensemble["test_f1"],
    "mcc": ensemble["test_mcc"],
}

ensemble_ci_table = ensemble_ci.copy()

ensemble_ci_table["point_estimate"] = (
    ensemble_ci_table["metric"].map(
        ensemble_point_map
    )
)

ensemble_ci_table = ensemble_ci_table[
    [
        "metric",
        "point_estimate",
        "bootstrap_n",
        "mean",
        "ci_lower_2p5",
        "ci_upper_97p5",
    ]
]

ensemble_ci_table = pd.concat(
    [
        ensemble_ci_table,
        pd.DataFrame([{
            "metric": "pearson_r",
            "point_estimate":
                ensemble_pearson["pearson_r"],
            "bootstrap_n": 10000,
            "mean":
                ensemble_pearson["bootstrap_mean_r"],
            "ci_lower_2p5":
                ensemble_pearson[
                    "bootstrap_ci_low"
                ],
            "ci_upper_97p5":
                ensemble_pearson[
                    "bootstrap_ci_high"
                ],
        }]),
    ],
    ignore_index=True,
)


# ---------------------------------------------------------
# 5. CNN ensemble vs k-mer RF paired comparison
# ---------------------------------------------------------
paired = paired_bootstrap.merge(
    paired_point[
        [
            "metric",
            "cnn_bilstm_ensemble",
            "kmer_rf",
            "raw_difference_cnn_minus_rf",
        ]
    ],
    left_on="metric",
    right_on="metric",
    how="left",
)

# Brier uses a different metric name in bootstrap file.
brier_mask = paired["metric"] == "brier_improvement"

if brier_mask.any():
    brier_point = paired_point.loc[
        paired_point["metric"] == "brier"
    ].iloc[0]

    paired.loc[
        brier_mask,
        "cnn_bilstm_ensemble",
    ] = brier_point["cnn_bilstm_ensemble"]

    paired.loc[
        brier_mask,
        "kmer_rf",
    ] = brier_point["kmer_rf"]

    paired.loc[
        brier_mask,
        "raw_difference_cnn_minus_rf",
    ] = brier_point[
        "raw_difference_cnn_minus_rf"
    ]


paired = paired[
    [
        "metric",
        "cnn_bilstm_ensemble",
        "kmer_rf",
        "raw_difference_cnn_minus_rf",
        "point_improvement",
        "ci_lower_2p5",
        "ci_upper_97p5",
        "probability_improvement_gt_zero",
        "paired_bootstrap_p_value",
        "interpretation",
    ]
]


# ---------------------------------------------------------
# Save outputs
# ---------------------------------------------------------
point_table.to_csv(
    OUT_DIR / "model_point_estimates.csv",
    index=False,
)

cnn_variability.to_csv(
    OUT_DIR / "cnn_five_seed_variability.csv",
    index=False,
)

manuscript_table.to_csv(
    OUT_DIR / "manuscript_main_model_table.csv",
    index=False,
)

ensemble_ci_table.to_csv(
    OUT_DIR / "cnn_ensemble_bootstrap_ci.csv",
    index=False,
)

paired.to_csv(
    OUT_DIR / "cnn_vs_kmer_rf_paired_comparison.csv",
    index=False,
)


excel_path = OUT_DIR / "machine_learning_results_tables.xlsx"

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl",
) as writer:
    manuscript_table.to_excel(
        writer,
        sheet_name="main_table",
        index=False,
    )

    point_table.to_excel(
        writer,
        sheet_name="point_estimates",
        index=False,
    )

    cnn_variability.to_excel(
        writer,
        sheet_name="cnn_seed_variability",
        index=False,
    )

    ensemble_ci_table.to_excel(
        writer,
        sheet_name="ensemble_CI",
        index=False,
    )

    paired.to_excel(
        writer,
        sheet_name="paired_vs_RF",
        index=False,
    )


print("===== MANUSCRIPT MAIN MODEL TABLE =====")
print(manuscript_table.to_string(index=False))

print("\n===== CNN ENSEMBLE 95% CI =====")
print(ensemble_ci_table.to_string(index=False))

print("\n===== CNN VS K-MER RF =====")
print(
    paired[
        [
            "metric",
            "point_improvement",
            "ci_lower_2p5",
            "ci_upper_97p5",
        ]
    ].to_string(index=False)
)

print("\n[OK] Saved:", excel_path)
