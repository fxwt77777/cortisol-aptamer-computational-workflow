from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path.home() / "rna_nna_project"

CNN_DIR = (
    ROOT
    / "revised_cnn"
    / "remote_run_20260806"
    / "predictions"
)

BASELINE_DIR = ROOT / "baselines" / "predictions"

OUT_DIR = (
    ROOT
    / "revised_cnn"
    / "pearson_analysis"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 10000
RANDOM_SEED = 20260806


def normalize_sequence(series):
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
    )


def stratified_bootstrap_ci(
    labels,
    probabilities,
    n_bootstrap=N_BOOTSTRAP,
    random_seed=RANDOM_SEED,
):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    negative_indices = np.flatnonzero(labels == 0)
    positive_indices = np.flatnonzero(labels == 1)

    rng = np.random.default_rng(random_seed)
    bootstrap_values = np.empty(n_bootstrap)

    for iteration in range(n_bootstrap):
        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )

        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [
                sampled_negative,
                sampled_positive,
            ]
        )

        sampled_labels = labels[sampled_indices]
        sampled_probabilities = probabilities[
            sampled_indices
        ]

        bootstrap_values[iteration] = pearsonr(
            sampled_labels,
            sampled_probabilities,
        ).statistic

    return {
        "bootstrap_mean": float(
            np.mean(bootstrap_values)
        ),
        "ci_low": float(
            np.quantile(
                bootstrap_values,
                0.025,
            )
        ),
        "ci_high": float(
            np.quantile(
                bootstrap_values,
                0.975,
            )
        ),
    }


ensemble_path = (
    CNN_DIR
    / "cnn_bilstm_ensemble_test.csv"
)

ensemble = pd.read_csv(ensemble_path)

ensemble["dna_sequence"] = normalize_sequence(
    ensemble["dna_sequence"]
)

required_ensemble_columns = {
    "seq_id",
    "dna_sequence",
    "class_top5",
    "ensemble_probability",
    "probability_seed_42",
    "probability_seed_43",
    "probability_seed_44",
    "probability_seed_45",
    "probability_seed_46",
}

missing = (
    required_ensemble_columns
    - set(ensemble.columns)
)

if missing:
    raise ValueError(
        "Ensemble file missing columns: "
        + ", ".join(sorted(missing))
    )

if len(ensemble) != 1716:
    raise ValueError(
        f"Expected 1716 test rows, found "
        f"{len(ensemble)}"
    )

if ensemble["seq_id"].duplicated().any():
    raise ValueError(
        "Duplicate seq_id values in ensemble file"
    )

if ensemble["dna_sequence"].duplicated().any():
    raise ValueError(
        "Duplicate sequences in ensemble file"
    )

master = ensemble[
    [
        "seq_id",
        "dna_sequence",
        "class_top5",
        "probability_seed_42",
        "probability_seed_43",
        "probability_seed_44",
        "probability_seed_45",
        "probability_seed_46",
        "ensemble_probability",
    ]
].copy()

master.rename(
    columns={
        "class_top5": "label",
    },
    inplace=True,
)

baseline_files = {
    "length_gc_lr":
        BASELINE_DIR
        / "length_gc_lr_test_predictions.csv",

    "simple_lr":
        BASELINE_DIR
        / "simple_lr_test_predictions.csv",

    "kmer_lr":
        BASELINE_DIR
        / "kmer_lr_test_predictions.csv",

    "kmer_rf":
        BASELINE_DIR
        / "kmer_rf_test_predictions.csv",
}

for model_name, path in baseline_files.items():
    baseline = pd.read_csv(path)

    required = {
        "seq_id",
        "dna_sequence",
        "label",
        "predicted_probability",
    }

    missing = required - set(baseline.columns)

    if missing:
        raise ValueError(
            f"{model_name} missing columns: "
            + ", ".join(sorted(missing))
        )

    baseline["dna_sequence"] = normalize_sequence(
        baseline["dna_sequence"]
    )

    if len(baseline) != 1716:
        raise ValueError(
            f"{model_name}: expected 1716 rows, "
            f"found {len(baseline)}"
        )

    if baseline["seq_id"].duplicated().any():
        raise ValueError(
            f"{model_name}: duplicate seq_id values"
        )

    aligned = master[
        [
            "seq_id",
            "dna_sequence",
            "label",
        ]
    ].merge(
        baseline[
            [
                "seq_id",
                "dna_sequence",
                "label",
                "predicted_probability",
            ]
        ],
        on="seq_id",
        how="left",
        suffixes=("_cnn", "_baseline"),
        validate="one_to_one",
    )

    if aligned["predicted_probability"].isna().any():
        raise ValueError(
            f"{model_name}: missing aligned predictions"
        )

    if not (
        aligned["dna_sequence_cnn"]
        == aligned["dna_sequence_baseline"]
    ).all():
        raise ValueError(
            f"{model_name}: sequence alignment mismatch"
        )

    if not (
        aligned["label_cnn"]
        == aligned["label_baseline"]
    ).all():
        raise ValueError(
            f"{model_name}: label alignment mismatch"
        )

    probability_by_id = baseline.set_index(
        "seq_id"
    )["predicted_probability"]

    master[model_name] = (
        master["seq_id"].map(
            probability_by_id
        )
    )


model_columns = {
    "CNN_BiLSTM_seed_42":
        "probability_seed_42",

    "CNN_BiLSTM_seed_43":
        "probability_seed_43",

    "CNN_BiLSTM_seed_44":
        "probability_seed_44",

    "CNN_BiLSTM_seed_45":
        "probability_seed_45",

    "CNN_BiLSTM_seed_46":
        "probability_seed_46",

    "CNN_BiLSTM_ensemble":
        "ensemble_probability",

    "Length_GC_LR":
        "length_gc_lr",

    "Simple_LR":
        "simple_lr",

    "Kmer_LR":
        "kmer_lr",

    "Kmer_RF":
        "kmer_rf",
}


labels = master["label"].astype(int).to_numpy()

result_rows = []

for model_index, (
    model_name,
    probability_column,
) in enumerate(
    model_columns.items()
):
    probabilities = pd.to_numeric(
        master[probability_column],
        errors="coerce",
    ).to_numpy()

    if np.isnan(probabilities).any():
        raise ValueError(
            f"{model_name}: missing probabilities"
        )

    correlation = pearsonr(
        labels,
        probabilities,
    )

    bootstrap = stratified_bootstrap_ci(
        labels,
        probabilities,
        random_seed=(
            RANDOM_SEED + model_index
        ),
    )

    result_rows.append({
        "model": model_name,
        "n_test": len(labels),
        "n_negative": int(
            np.sum(labels == 0)
        ),
        "n_positive": int(
            np.sum(labels == 1)
        ),
        "pearson_r": float(
            correlation.statistic
        ),
        "pearson_p": float(
            correlation.pvalue
        ),
        "bootstrap_mean_r":
            bootstrap["bootstrap_mean"],
        "bootstrap_ci_low":
            bootstrap["ci_low"],
        "bootstrap_ci_high":
            bootstrap["ci_high"],
    })


results = pd.DataFrame(result_rows)

cnn_seed_rows = results.loc[
    results["model"].str.match(
        r"CNN_BiLSTM_seed_\d+"
    )
].copy()

cnn_seed_summary = pd.DataFrame([
    {
        "metric": "Pearson_r",
        "n_seeds": len(cnn_seed_rows),
        "mean": cnn_seed_rows[
            "pearson_r"
        ].mean(),
        "sd": cnn_seed_rows[
            "pearson_r"
        ].std(ddof=1),
        "min": cnn_seed_rows[
            "pearson_r"
        ].min(),
        "max": cnn_seed_rows[
            "pearson_r"
        ].max(),
    }
])


results.to_csv(
    OUT_DIR / "test_pearson_all_models.csv",
    index=False,
)

cnn_seed_summary.to_csv(
    OUT_DIR / "cnn_seed_pearson_mean_sd.csv",
    index=False,
)

master.to_csv(
    OUT_DIR / "aligned_test_predictions.csv",
    index=False,
)


excel_path = (
    OUT_DIR
    / "test_pearson_analysis.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl",
) as writer:
    results.to_excel(
        writer,
        sheet_name="all_models",
        index=False,
    )

    cnn_seed_summary.to_excel(
        writer,
        sheet_name="cnn_seed_summary",
        index=False,
    )

    master.to_excel(
        writer,
        sheet_name="aligned_predictions",
        index=False,
    )


print("===== ALIGNMENT AUDIT =====")
print("test rows:", len(master))
print(
    "labels:",
    master["label"].value_counts()
    .sort_index()
    .to_dict(),
)
print(
    "unique sequences:",
    master["dna_sequence"].nunique(),
)

print("\n===== TEST PEARSON CORRELATIONS =====")
print(
    results[
        [
            "model",
            "pearson_r",
            "pearson_p",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ]
    ].to_string(index=False)
)

print("\n===== CNN FIVE-SEED SUMMARY =====")
print(
    cnn_seed_summary.to_string(
        index=False
    )
)

print("\n[OK] Saved:", excel_path)
