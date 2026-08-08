from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


ROOT = Path.home() / "rna_nna_project"

CNN_PREDICTIONS = (
    ROOT
    / "revised_cnn"
    / "remote_run_20260806"
    / "predictions"
    / "cnn_bilstm_ensemble_test.csv"
)

CNN_METRICS = (
    ROOT
    / "revised_cnn"
    / "remote_run_20260806"
    / "results"
    / "cnn_bilstm_ensemble_metrics.csv"
)

RF_PREDICTIONS = (
    ROOT
    / "baselines"
    / "predictions"
    / "kmer_rf_test_predictions.csv"
)

RF_METRICS = (
    ROOT
    / "baselines"
    / "results"
    / "baseline_metrics_test.csv"
)

OUTPUT_DIR = (
    ROOT
    / "revised_cnn"
    / "paired_comparison_vs_kmer_rf"
)

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260806


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        f"Could not find {description}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    return {
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                labels,
                predictions,
            )
        ),
        "brier": float(
            brier_score_loss(
                labels,
                probabilities,
            )
        ),
    }


def calculate_improvements(
    cnn_metrics: dict[str, float],
    rf_metrics: dict[str, float],
) -> dict[str, float]:
    """
    Positive values always mean that CNN–BiLSTM is better.

    For ROC-AUC, PR-AUC, accuracy, balanced accuracy,
    F1 and MCC:
        CNN minus RF

    For Brier score:
        RF minus CNN
    because a lower Brier score is better.
    """
    return {
        "roc_auc": (
            cnn_metrics["roc_auc"]
            - rf_metrics["roc_auc"]
        ),
        "pr_auc": (
            cnn_metrics["pr_auc"]
            - rf_metrics["pr_auc"]
        ),
        "accuracy": (
            cnn_metrics["accuracy"]
            - rf_metrics["accuracy"]
        ),
        "balanced_accuracy": (
            cnn_metrics["balanced_accuracy"]
            - rf_metrics["balanced_accuracy"]
        ),
        "f1": (
            cnn_metrics["f1"]
            - rf_metrics["f1"]
        ),
        "mcc": (
            cnn_metrics["mcc"]
            - rf_metrics["mcc"]
        ),
        "brier_improvement": (
            rf_metrics["brier"]
            - cnn_metrics["brier"]
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cnn = pd.read_csv(CNN_PREDICTIONS)
    rf = pd.read_csv(RF_PREDICTIONS)

    cnn_sequence_column = find_column(
        cnn,
        [
            "dna_sequence",
            "sequence",
            "seq",
        ],
        "CNN sequence column",
    )

    rf_sequence_column = find_column(
        rf,
        [
            "dna_sequence",
            "sequence",
            "seq",
        ],
        "random-forest sequence column",
    )

    cnn_label_column = find_column(
        cnn,
        [
            "class_top5",
            "true_label",
            "label",
            "y_true",
        ],
        "CNN true-label column",
    )

    rf_label_column = find_column(
        rf,
        [
            "class_top5",
            "true_label",
            "label",
            "y_true",
        ],
        "random-forest true-label column",
    )

    cnn_probability_column = find_column(
        cnn,
        [
            "ensemble_probability",
            "predicted_probability",
            "probability",
            "y_probability",
            "y_prob",
        ],
        "CNN probability column",
    )

    rf_probability_column = find_column(
        rf,
        [
            "predicted_probability",
            "probability",
            "y_probability",
            "y_prob",
            "positive_probability",
        ],
        "random-forest probability column",
    )

    cnn_small = cnn[
        [
            cnn_sequence_column,
            cnn_label_column,
            cnn_probability_column,
        ]
    ].copy()

    cnn_small.columns = [
        "dna_sequence",
        "cnn_label",
        "cnn_probability",
    ]

    rf_small = rf[
        [
            rf_sequence_column,
            rf_label_column,
            rf_probability_column,
        ]
    ].copy()

    rf_small.columns = [
        "dna_sequence",
        "rf_label",
        "rf_probability",
    ]

    if cnn_small["dna_sequence"].duplicated().any():
        raise ValueError(
            "CNN prediction file contains duplicate sequences"
        )

    if rf_small["dna_sequence"].duplicated().any():
        raise ValueError(
            "RF prediction file contains duplicate sequences"
        )

    merged = cnn_small.merge(
        rf_small,
        on="dna_sequence",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(cnn_small):
        raise ValueError(
            "Not all CNN test sequences matched RF predictions: "
            f"{len(merged)} / {len(cnn_small)}"
        )

    if len(merged) != len(rf_small):
        raise ValueError(
            "Not all RF test sequences matched CNN predictions: "
            f"{len(merged)} / {len(rf_small)}"
        )

    if not np.array_equal(
        merged["cnn_label"].to_numpy(),
        merged["rf_label"].to_numpy(),
    ):
        raise ValueError(
            "True labels differ between prediction files"
        )

    if len(merged) != 1716:
        raise ValueError(
            f"Expected 1716 matched test sequences, "
            f"found {len(merged)}"
        )

    cnn_metrics_table = pd.read_csv(CNN_METRICS)

    cnn_threshold = float(
        cnn_metrics_table.loc[
            0,
            "validation_threshold",
        ]
    )

    rf_metrics_table = pd.read_csv(RF_METRICS)

    rf_row = rf_metrics_table.loc[
        rf_metrics_table["model"].eq("kmer_rf")
    ]

    if len(rf_row) != 1:
        raise ValueError(
            "Could not uniquely identify kmer_rf metrics"
        )

    rf_threshold = float(
        rf_row.iloc[0]["validation_threshold"]
    )

    labels = merged[
        "cnn_label"
    ].to_numpy(dtype=np.int64)

    cnn_probabilities = merged[
        "cnn_probability"
    ].to_numpy(dtype=float)

    rf_probabilities = merged[
        "rf_probability"
    ].to_numpy(dtype=float)

    cnn_point_metrics = calculate_metrics(
        labels,
        cnn_probabilities,
        cnn_threshold,
    )

    rf_point_metrics = calculate_metrics(
        labels,
        rf_probabilities,
        rf_threshold,
    )

    point_improvements = calculate_improvements(
        cnn_point_metrics,
        rf_point_metrics,
    )

    point_rows = []

    for metric in cnn_point_metrics:
        point_rows.append({
            "metric": metric,
            "cnn_bilstm_ensemble": (
                cnn_point_metrics[metric]
            ),
            "kmer_rf": rf_point_metrics[metric],
            "raw_difference_cnn_minus_rf": (
                cnn_point_metrics[metric]
                - rf_point_metrics[metric]
            ),
        })

    point_table = pd.DataFrame(point_rows)

    point_table.to_csv(
        OUTPUT_DIR / "paired_point_metrics.csv",
        index=False,
    )

    merged["cnn_predicted_label"] = (
        cnn_probabilities >= cnn_threshold
    ).astype(np.int64)

    merged["rf_predicted_label"] = (
        rf_probabilities >= rf_threshold
    ).astype(np.int64)

    merged.to_csv(
        OUTPUT_DIR / "paired_test_predictions.csv",
        index=False,
    )

    negative_indices = np.where(labels == 0)[0]
    positive_indices = np.where(labels == 1)[0]

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    bootstrap_values = {
        metric: np.empty(
            BOOTSTRAP_ITERATIONS,
            dtype=np.float64,
        )
        for metric in point_improvements
    }

    for iteration in range(
        BOOTSTRAP_ITERATIONS
    ):
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

        sampled_labels = labels[
            sampled_indices
        ]

        sampled_cnn_probabilities = (
            cnn_probabilities[sampled_indices]
        )

        sampled_rf_probabilities = (
            rf_probabilities[sampled_indices]
        )

        sampled_cnn_metrics = calculate_metrics(
            sampled_labels,
            sampled_cnn_probabilities,
            cnn_threshold,
        )

        sampled_rf_metrics = calculate_metrics(
            sampled_labels,
            sampled_rf_probabilities,
            rf_threshold,
        )

        sampled_improvements = (
            calculate_improvements(
                sampled_cnn_metrics,
                sampled_rf_metrics,
            )
        )

        for metric, value in (
            sampled_improvements.items()
        ):
            bootstrap_values[metric][
                iteration
            ] = value

        if (
            iteration == 0
            or (iteration + 1) % 1000 == 0
        ):
            print(
                f"[INFO] Bootstrap "
                f"{iteration + 1}/"
                f"{BOOTSTRAP_ITERATIONS}",
                flush=True,
            )

    bootstrap_long_rows = []
    summary_rows = []

    for metric, differences in (
        bootstrap_values.items()
    ):
        point_difference = (
            point_improvements[metric]
        )

        ci_lower = float(
            np.quantile(differences, 0.025)
        )

        ci_upper = float(
            np.quantile(differences, 0.975)
        )

        probability_positive = float(
            np.mean(differences > 0)
        )

        probability_nonpositive = (
            np.sum(differences <= 0) + 1
        ) / (BOOTSTRAP_ITERATIONS + 1)

        probability_nonnegative = (
            np.sum(differences >= 0) + 1
        ) / (BOOTSTRAP_ITERATIONS + 1)

        paired_p_value = min(
            1.0,
            2.0
            * min(
                probability_nonpositive,
                probability_nonnegative,
            ),
        )

        if ci_lower > 0:
            interpretation = (
                "CNN-BiLSTM better; "
                "95% CI excludes zero"
            )
        elif ci_upper < 0:
            interpretation = (
                "k-mer RF better; "
                "95% CI excludes zero"
            )
        else:
            interpretation = (
                "Difference uncertain; "
                "95% CI includes zero"
            )

        summary_rows.append({
            "metric": metric,
            "difference_definition": (
                "positive means CNN-BiLSTM better"
            ),
            "point_improvement": (
                point_difference
            ),
            "bootstrap_mean_improvement": float(
                differences.mean()
            ),
            "bootstrap_sd": float(
                differences.std(ddof=1)
            ),
            "ci_lower_2p5": ci_lower,
            "ci_upper_97p5": ci_upper,
            "probability_improvement_gt_zero": (
                probability_positive
            ),
            "paired_bootstrap_p_value": (
                paired_p_value
            ),
            "interpretation": interpretation,
        })

        for bootstrap_index, difference in (
            enumerate(differences, start=1)
        ):
            bootstrap_long_rows.append({
                "bootstrap_iteration": (
                    bootstrap_index
                ),
                "metric": metric,
                "improvement": float(difference),
            })

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        OUTPUT_DIR
        / "paired_bootstrap_summary.csv",
        index=False,
    )

    pd.DataFrame(
        bootstrap_long_rows
    ).to_csv(
        OUTPUT_DIR
        / "paired_bootstrap_differences.csv",
        index=False,
    )

    audit = {
        "matched_test_sequences": int(
            len(merged)
        ),
        "negative_sequences": int(
            (labels == 0).sum()
        ),
        "positive_sequences": int(
            (labels == 1).sum()
        ),
        "cnn_validation_threshold": (
            cnn_threshold
        ),
        "rf_validation_threshold": (
            rf_threshold
        ),
        "bootstrap_iterations": (
            BOOTSTRAP_ITERATIONS
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_method": (
            "paired stratified nonparametric "
            "bootstrap over test observations"
        ),
        "point_metrics": {
            "cnn_bilstm_ensemble": (
                cnn_point_metrics
            ),
            "kmer_rf": rf_point_metrics,
        },
    }

    with (
        OUTPUT_DIR / "paired_comparison_audit.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            audit,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("===== Alignment audit =====")
    print("matched sequences:", len(merged))
    print(
        "labels:",
        dict(
            pd.Series(labels)
            .value_counts()
            .sort_index()
        ),
    )
    print(
        "CNN threshold:",
        cnn_threshold,
    )
    print(
        "RF threshold:",
        rf_threshold,
    )

    print()
    print("===== Point metrics =====")
    print(
        point_table.to_string(index=False)
    )

    print()
    print(
        "===== Paired-bootstrap "
        "improvement summary ====="
    )
    print(
        summary.to_string(index=False)
    )

    print()
    print(
        f"[OK] Results saved to "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
