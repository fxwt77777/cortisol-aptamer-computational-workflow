from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = (
    ROOT
    / "similarity_split/splits/id70_seed20260803"
)

TRAIN_FILE = SPLIT_DIR / "train_top5_id70.csv"
VAL_FILE = SPLIT_DIR / "validation_top5_id70.csv"
TEST_FILE = SPLIT_DIR / "test_top5_id70.csv"

OUT_ROOT = ROOT / "baselines"
MODEL_DIR = OUT_ROOT / "models"
PRED_DIR = OUT_ROOT / "predictions"
RESULT_DIR = OUT_ROOT / "results"

SEED = 20260803
BOOTSTRAP_ITERATIONS = 1000

SIMPLE_FEATURE_NAMES = [
    "length_nt",
    "gc_content",
    "max_run",
    "fraction_A",
    "fraction_C",
    "fraction_G",
    "fraction_T",
]


def load_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "seq_id",
        "dna_sequence",
        "label",
        "length_nt",
        "gc_content",
        "max_run",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}缺少字段：{sorted(missing)}"
        )

    df["dna_sequence"] = (
        df["dna_sequence"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["label"] = df["label"].astype(int)

    return df


def build_simple_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    rows = []

    for row in df.itertuples(index=False):
        sequence = row.dna_sequence
        length = len(sequence)
        counts = Counter(sequence)

        rows.append([
            float(row.length_nt),
            float(row.gc_content),
            float(row.max_run),
            counts.get("A", 0) / length,
            counts.get("C", 0) / length,
            counts.get("G", 0) / length,
            counts.get("T", 0) / length,
        ])

    return (
        np.asarray(rows, dtype=np.float32),
        SIMPLE_FEATURE_NAMES.copy(),
    )


def make_kmer_index(
    minimum_k: int = 1,
    maximum_k: int = 4,
) -> tuple[dict[str, int], list[str]]:
    kmers = []

    for k in range(minimum_k, maximum_k + 1):
        for letters in product("ACGT", repeat=k):
            kmers.append("".join(letters))

    return (
        {kmer: index for index, kmer in enumerate(kmers)},
        kmers,
    )


KMER_INDEX, KMER_NAMES = make_kmer_index(1, 4)


def build_kmer_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    simple, simple_names = build_simple_features(df)

    kmer_matrix = np.zeros(
        (len(df), len(KMER_NAMES)),
        dtype=np.float32,
    )

    sequences = df["dna_sequence"].tolist()

    for row_index, sequence in enumerate(sequences):
        sequence_length = len(sequence)

        for k in range(1, 5):
            denominator = sequence_length - k + 1

            if denominator <= 0:
                continue

            for position in range(denominator):
                kmer = sequence[position:position + k]
                feature_index = KMER_INDEX[kmer]
                kmer_matrix[row_index, feature_index] += 1.0

            start = sum(4 ** previous_k for previous_k in range(1, k))

            end = start + 4 ** k

            kmer_matrix[
                row_index,
                start:end,
            ] /= denominator

    combined = np.concatenate(
        [simple, kmer_matrix],
        axis=1,
    )

    feature_names = (
        simple_names
        + [f"kmer_freq_{name}" for name in KMER_NAMES]
    )

    return combined, feature_names


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(y_true, probabilities)
    )

    youden_j = true_positive_rate - false_positive_rate

    finite = np.isfinite(thresholds)

    if not finite.any():
        return 0.5

    valid_indices = np.where(finite)[0]

    best_local = int(
        np.argmax(youden_j[finite])
    )

    best_index = int(valid_indices[best_local])

    return float(thresholds[best_index])


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "f1": float(
            f1_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "sensitivity": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "mcc": float(
            matthews_corrcoef(y_true, predictions)
        ),
        "brier": float(
            brier_score_loss(y_true, probabilities)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    seed: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample_count = len(y_true)

    metric_values = {
        "accuracy": [],
        "balanced_accuracy": [],
        "roc_auc": [],
        "pr_auc": [],
        "f1": [],
        "mcc": [],
    }

    for _ in range(iterations):
        indices = rng.integers(
            0,
            sample_count,
            size=sample_count,
        )

        y_boot = y_true[indices]
        probability_boot = probabilities[indices]

        if np.unique(y_boot).size < 2:
            continue

        prediction_boot = (
            probability_boot >= threshold
        ).astype(int)

        metric_values["accuracy"].append(
            accuracy_score(y_boot, prediction_boot)
        )

        metric_values["balanced_accuracy"].append(
            balanced_accuracy_score(
                y_boot,
                prediction_boot,
            )
        )

        metric_values["roc_auc"].append(
            roc_auc_score(y_boot, probability_boot)
        )

        metric_values["pr_auc"].append(
            average_precision_score(
                y_boot,
                probability_boot,
            )
        )

        metric_values["f1"].append(
            f1_score(
                y_boot,
                prediction_boot,
                zero_division=0,
            )
        )

        metric_values["mcc"].append(
            matthews_corrcoef(
                y_boot,
                prediction_boot,
            )
        )

    rows = []

    for metric_name, values in metric_values.items():
        array = np.asarray(values, dtype=float)

        rows.append({
            "metric": metric_name,
            "bootstrap_n": int(len(array)),
            "mean": float(array.mean()),
            "ci_lower_2p5": float(
                np.quantile(array, 0.025)
            ),
            "ci_upper_97p5": float(
                np.quantile(array, 0.975)
            ),
        })

    return pd.DataFrame(rows)


def fit_logistic_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
):
    candidate_c_values = [
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ]

    candidates = []

    for c_value in candidate_c_values:
        model = Pipeline([
            (
                "scale",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ])

        model.fit(x_train, y_train)

        validation_probability = (
            model.predict_proba(x_validation)[:, 1]
        )

        validation_auc = roc_auc_score(
            y_validation,
            validation_probability,
        )

        candidates.append({
            "model": model,
            "C": c_value,
            "validation_auc": validation_auc,
        })

    best = max(
        candidates,
        key=lambda item: item["validation_auc"],
    )

    tuning_table = pd.DataFrame([
        {
            "C": item["C"],
            "validation_auc": item["validation_auc"],
        }
        for item in candidates
    ])

    return best["model"], best["C"], tuning_table


def fit_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
):
    parameter_grid = [
        {
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        {
            "max_depth": None,
            "min_samples_leaf": 2,
        },
        {
            "max_depth": 16,
            "min_samples_leaf": 1,
        },
        {
            "max_depth": 16,
            "min_samples_leaf": 2,
        },
        {
            "max_depth": 24,
            "min_samples_leaf": 1,
        },
        {
            "max_depth": 24,
            "min_samples_leaf": 2,
        },
    ]

    candidates = []

    for parameters in parameter_grid:
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            max_features="sqrt",
            n_jobs=8,
            random_state=SEED,
            class_weight=None,
        )

        model.fit(x_train, y_train)

        validation_probability = (
            model.predict_proba(x_validation)[:, 1]
        )

        validation_auc = roc_auc_score(
            y_validation,
            validation_probability,
        )

        candidates.append({
            "model": model,
            "validation_auc": validation_auc,
            **parameters,
        })

    best = max(
        candidates,
        key=lambda item: item["validation_auc"],
    )

    tuning_table = pd.DataFrame([
        {
            "max_depth": item["max_depth"],
            "min_samples_leaf": item[
                "min_samples_leaf"
            ],
            "validation_auc": item[
                "validation_auc"
            ],
        }
        for item in candidates
    ])

    best_parameters = {
        "max_depth": best["max_depth"],
        "min_samples_leaf": best[
            "min_samples_leaf"
        ],
    }

    return (
        best["model"],
        best_parameters,
        tuning_table,
    )


def evaluate_model(
    model_name: str,
    model,
    tuning_information: dict,
    tuning_table: pd.DataFrame,
    feature_names: list[str],
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
) -> dict:
    validation_probability = (
        model.predict_proba(x_validation)[:, 1]
    )

    threshold = choose_threshold(
        y_validation,
        validation_probability,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probability,
        threshold,
    )

    test_probability = (
        model.predict_proba(x_test)[:, 1]
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probability,
        threshold,
    )

    bootstrap_table = bootstrap_confidence_intervals(
        y_test,
        test_probability,
        threshold,
        seed=SEED + 100,
    )

    predictions = test_df[
        [
            "seq_id",
            "dna_sequence",
            "label",
            "length_nt",
            "gc_content",
        ]
    ].copy()

    predictions["predicted_probability"] = (
        test_probability
    )

    predictions["predicted_label"] = (
        test_probability >= threshold
    ).astype(int)

    predictions.to_csv(
        PRED_DIR / f"{model_name}_test_predictions.csv",
        index=False,
    )

    tuning_table.to_csv(
        RESULT_DIR / f"{model_name}_tuning.csv",
        index=False,
    )

    bootstrap_table.insert(
        0,
        "model",
        model_name,
    )

    bootstrap_table.to_csv(
        RESULT_DIR / f"{model_name}_bootstrap_ci.csv",
        index=False,
    )

    joblib.dump(
        model,
        MODEL_DIR / f"{model_name}.joblib",
    )

    (
        RESULT_DIR / f"{model_name}_feature_names.txt"
    ).write_text(
        "\n".join(feature_names),
        encoding="utf-8",
    )

    result = {
        "model": model_name,
        "feature_count": len(feature_names),
        **tuning_information,
        **{
            f"validation_{key}": value
            for key, value in validation_metrics.items()
        },
        **{
            f"test_{key}": value
            for key, value in test_metrics.items()
        },
    }

    return result


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split(TRAIN_FILE)
    validation_df = load_split(VAL_FILE)
    test_df = load_split(TEST_FILE)

    y_train = train_df["label"].to_numpy()
    y_validation = validation_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    print(
        "[INFO] Split sizes:",
        len(train_df),
        len(validation_df),
        len(test_df),
    )

    simple_train, simple_names = (
        build_simple_features(train_df)
    )
    simple_validation, _ = (
        build_simple_features(validation_df)
    )
    simple_test, _ = (
        build_simple_features(test_df)
    )

    kmer_train, kmer_names = (
        build_kmer_features(train_df)
    )
    kmer_validation, _ = (
        build_kmer_features(validation_df)
    )
    kmer_test, _ = (
        build_kmer_features(test_df)
    )

    feature_sets = {
        "length_gc": {
            "train": simple_train[:, 0:2],
            "validation": simple_validation[:, 0:2],
            "test": simple_test[:, 0:2],
            "names": simple_names[0:2],
        },
        "simple": {
            "train": simple_train,
            "validation": simple_validation,
            "test": simple_test,
            "names": simple_names,
        },
        "kmer": {
            "train": kmer_train,
            "validation": kmer_validation,
            "test": kmer_test,
            "names": kmer_names,
        },
    }

    results = []

    logistic_jobs = [
        ("length_gc_lr", "length_gc"),
        ("simple_lr", "simple"),
        ("kmer_lr", "kmer"),
    ]

    for model_name, feature_key in logistic_jobs:
        print(f"[INFO] 训练 {model_name}")
        start = time.time()

        features = feature_sets[feature_key]

        model, best_c, tuning_table = (
            fit_logistic_model(
                features["train"],
                y_train,
                features["validation"],
                y_validation,
            )
        )

        result = evaluate_model(
            model_name=model_name,
            model=model,
            tuning_information={
                "selected_C": float(best_c),
            },
            tuning_table=tuning_table,
            feature_names=features["names"],
            x_validation=features["validation"],
            y_validation=y_validation,
            x_test=features["test"],
            y_test=y_test,
            test_df=test_df,
        )

        result["runtime_seconds"] = (
            time.time() - start
        )

        results.append(result)

        print(
            f"[OK] {model_name}: "
            f"test AUC={result['test_roc_auc']:.4f}, "
            f"test Acc={result['test_accuracy']:.4f}"
        )

    print("[INFO] 训练 kmer_rf")
    start = time.time()

    model, best_parameters, tuning_table = (
        fit_random_forest(
            kmer_train,
            y_train,
            kmer_validation,
            y_validation,
        )
    )

    result = evaluate_model(
        model_name="kmer_rf",
        model=model,
        tuning_information={
            "selected_max_depth": (
                "None"
                if best_parameters["max_depth"] is None
                else int(best_parameters["max_depth"])
            ),
            "selected_min_samples_leaf": int(
                best_parameters["min_samples_leaf"]
            ),
        },
        tuning_table=tuning_table,
        feature_names=kmer_names,
        x_validation=kmer_validation,
        y_validation=y_validation,
        x_test=kmer_test,
        y_test=y_test,
        test_df=test_df,
    )

    result["runtime_seconds"] = time.time() - start

    results.append(result)

    print(
        f"[OK] kmer_rf: "
        f"test AUC={result['test_roc_auc']:.4f}, "
        f"test Acc={result['test_accuracy']:.4f}"
    )

    results_df = pd.DataFrame(results)

    preferred_columns = [
        "model",
        "feature_count",
        "selected_C",
        "selected_max_depth",
        "selected_min_samples_leaf",
        "validation_roc_auc",
        "validation_pr_auc",
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_threshold",
        "test_roc_auc",
        "test_pr_auc",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "test_precision",
        "test_sensitivity",
        "test_specificity",
        "test_mcc",
        "test_brier",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
        "runtime_seconds",
    ]

    for column in preferred_columns:
        if column not in results_df.columns:
            results_df[column] = np.nan

    results_df = results_df[preferred_columns]

    results_df.to_csv(
        RESULT_DIR / "baseline_metrics_test.csv",
        index=False,
    )

    with open(
        RESULT_DIR / "baseline_run_config.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "seed": SEED,
                "bootstrap_iterations": (
                    BOOTSTRAP_ITERATIONS
                ),
                "split_directory": str(SPLIT_DIR),
                "selection_metric": (
                    "validation ROC-AUC"
                ),
                "classification_threshold": (
                    "Youden J selected on validation set"
                ),
                "kmer_range": [1, 4],
                "random_forest_estimators": 400,
                "random_forest_threads": 8,
            },
            handle,
            indent=2,
        )

    print()
    print("[OK] 所有基线模型完成")
    print(
        results_df[
            [
                "model",
                "validation_roc_auc",
                "test_roc_auc",
                "test_pr_auc",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_mcc",
            ]
        ].to_string(index=False)
    )

    print(
        f"\n[OK] 结果目录：{RESULT_DIR}"
    )


if __name__ == "__main__":
    main()
