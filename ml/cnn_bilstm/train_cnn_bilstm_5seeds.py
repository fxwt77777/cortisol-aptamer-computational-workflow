from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
from torch.utils.data import DataLoader, Dataset

from config import (
    BATCH_SIZE,
    DROPOUT,
    EARLY_STOP_PATIENCE,
    HIDDEN_DIM,
    KERNEL_SIZES,
    CONV_FILTERS,
    LR,
    L_MAX,
    N_EPOCHS,
    NUC_VOCAB,
    WEIGHT_DECAY,
)
from model_cls import CNN_BiLSTM_Classifier


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
PREDICTION_DIR = ROOT / "predictions"
HISTORY_DIR = ROOT / "history"

TRAIN_FILE = DATA_DIR / "train_top5.csv"
VALIDATION_FILE = DATA_DIR / "val_top5.csv"
TEST_FILE = DATA_DIR / "test_top5.csv"

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
BOOTSTRAP_N = 1000

NUC2IDX = {
    nucleotide: index
    for index, nucleotide in enumerate(NUC_VOCAB)
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_sequence(sequence: str) -> np.ndarray:
    sequence = str(sequence).upper().strip()

    encoded = np.zeros(
        (L_MAX, len(NUC_VOCAB)),
        dtype=np.float32,
    )

    for position, nucleotide in enumerate(
        sequence[:L_MAX]
    ):
        if nucleotide not in NUC2IDX:
            raise ValueError(
                f"Illegal nucleotide {nucleotide!r} "
                f"in sequence {sequence!r}"
            )

        encoded[
            position,
            NUC2IDX[nucleotide],
        ] = 1.0

    return encoded


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    dataframe = pd.read_csv(path)

    required_columns = {
        "dna_sequence",
        "class_top5",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{path.name} missing columns: "
            f"{sorted(missing)}"
        )

    dataframe["dna_sequence"] = (
        dataframe["dna_sequence"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    dataframe["class_top5"] = (
        dataframe["class_top5"].astype(int)
    )

    if not dataframe[
        "dna_sequence"
    ].str.fullmatch(r"[ACGT]+").all():
        raise ValueError(
            f"{path.name} contains invalid DNA sequences"
        )

    if not dataframe[
        "class_top5"
    ].isin([0, 1]).all():
        raise ValueError(
            f"{path.name} contains invalid labels"
        )

    return dataframe.reset_index(drop=True)


class AptamerDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.inputs = np.stack([
            encode_sequence(sequence)
            for sequence in dataframe["dna_sequence"]
        ])

        self.labels = dataframe[
            "class_top5"
        ].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.inputs[index]),
            torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        )


def make_loaders(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    seed: int,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": True,
    }

    train_loader = DataLoader(
        AptamerDataset(train_dataframe),
        shuffle=True,
        generator=generator,
        **common,
    )

    validation_loader = DataLoader(
        AptamerDataset(validation_dataframe),
        shuffle=False,
        **common,
    )

    test_loader = DataLoader(
        AptamerDataset(test_dataframe),
        shuffle=False,
        **common,
    )

    return train_loader, validation_loader, test_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_n = 0

    for inputs, labels in loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_n = labels.size(0)

        total_loss += float(loss.item()) * batch_n
        total_n += batch_n

    return total_loss / total_n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()

    all_labels = []
    all_probabilities = []

    total_loss = 0.0
    total_n = 0

    for inputs, labels in loader:
        inputs = inputs.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(inputs)
        loss = criterion(logits, labels)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        batch_n = labels.size(0)

        total_loss += float(loss.item()) * batch_n
        total_n += batch_n

        all_labels.append(
            labels.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    labels_array = np.concatenate(all_labels)
    probabilities_array = np.concatenate(
        all_probabilities
    )

    return (
        total_loss / total_n,
        labels_array,
        probabilities_array,
    )


def validation_youden_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    fpr, tpr, thresholds = roc_curve(
        labels,
        probabilities,
    )

    finite = np.isfinite(thresholds)

    if not finite.any():
        return 0.5

    valid_indices = np.where(finite)[0]
    local_best = int(
        np.argmax((tpr - fpr)[finite])
    )

    return float(
        thresholds[valid_indices[local_best]]
    )


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    predictions_0p5 = (
        probabilities >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),
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
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "sensitivity": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
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
        "accuracy_at_0p5": float(
            accuracy_score(
                labels,
                predictions_0p5,
            )
        ),
        "mcc_at_0p5": float(
            matthews_corrcoef(
                labels,
                predictions_0p5,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_predictions(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    output = dataframe.copy()

    output["predicted_probability"] = (
        probabilities
    )

    output["predicted_label"] = (
        probabilities >= threshold
    ).astype(int)

    output.to_csv(
        output_path,
        index=False,
    )


def bootstrap_ensemble(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    seed: int = 20260806,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    values = {
        "roc_auc": [],
        "pr_auc": [],
        "accuracy": [],
        "balanced_accuracy": [],
        "f1": [],
        "mcc": [],
    }

    sample_n = len(labels)

    for _ in range(BOOTSTRAP_N):
        indices = rng.integers(
            0,
            sample_n,
            size=sample_n,
        )

        sampled_labels = labels[indices]
        sampled_probabilities = probabilities[indices]

        if np.unique(sampled_labels).size < 2:
            continue

        metrics = calculate_metrics(
            sampled_labels,
            sampled_probabilities,
            threshold,
        )

        for metric_name in values:
            values[metric_name].append(
                metrics[metric_name]
            )

    rows = []

    for metric_name, metric_values in values.items():
        array = np.asarray(
            metric_values,
            dtype=float,
        )

        rows.append({
            "metric": metric_name,
            "bootstrap_n": len(array),
            "mean": float(array.mean()),
            "ci_lower_2p5": float(
                np.quantile(array, 0.025)
            ),
            "ci_upper_97p5": float(
                np.quantile(array, 0.975)
            ),
        })

    return pd.DataFrame(rows)


def smoke_test() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available"
        )

    dataframe = load_dataframe(TRAIN_FILE)

    dataset = AptamerDataset(
        dataframe.head(BATCH_SIZE)
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
    )

    model = CNN_BiLSTM_Classifier().cuda()

    inputs, labels = next(iter(loader))

    inputs = inputs.cuda()
    logits = model(inputs)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("===== Smoke test =====")
    print("torch:", torch.__version__)
    print("torch_cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    print("input_shape:", tuple(inputs.shape))
    print("logits_shape:", tuple(logits.shape))
    print("parameter_count:", parameter_count)
    print("finite_logits:", bool(
        torch.isfinite(logits).all()
    ))
    print("[OK] CNN-BiLSTM smoke test passed")


def train_one_seed(
    seed: int,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    device: torch.device,
):
    set_seed(seed)

    train_loader, validation_loader, test_loader = (
        make_loaders(
            train_dataframe,
            validation_dataframe,
            test_dataframe,
            seed,
        )
    )

    model = CNN_BiLSTM_Classifier().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    best_validation_auc = -np.inf
    best_validation_loss = np.inf
    best_state = None
    best_epoch = 0
    no_improvement = 0

    history = []
    start_time = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        (
            validation_loss,
            validation_labels,
            validation_probabilities,
        ) = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        validation_auc = roc_auc_score(
            validation_labels,
            validation_probabilities,
        )

        validation_pr_auc = (
            average_precision_score(
                validation_labels,
                validation_probabilities,
            )
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_roc_auc": validation_auc,
            "validation_pr_auc": validation_pr_auc,
        })

        improved_auc = (
            validation_auc
            > best_validation_auc + 1e-6
        )

        tied_auc_lower_loss = (
            abs(
                validation_auc
                - best_validation_auc
            ) <= 1e-6
            and validation_loss
            < best_validation_loss
        )

        if improved_auc or tied_auc_lower_loss:
            best_validation_auc = validation_auc
            best_validation_loss = validation_loss
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            no_improvement = 0
        else:
            no_improvement += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved_auc
        ):
            print(
                f"[seed={seed}] "
                f"epoch={epoch} "
                f"train_loss={train_loss:.5f} "
                f"val_loss={validation_loss:.5f} "
                f"val_auc={validation_auc:.5f} "
                f"val_pr_auc={validation_pr_auc:.5f}",
                flush=True,
            )

        if no_improvement >= EARLY_STOP_PATIENCE:
            print(
                f"[seed={seed}] "
                f"early stopping at epoch {epoch}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"No best model was saved for seed {seed}"
        )

    model.load_state_dict(best_state)

    (
        validation_loss,
        validation_labels,
        validation_probabilities,
    ) = evaluate(
        model,
        validation_loader,
        criterion,
        device,
    )

    threshold = validation_youden_threshold(
        validation_labels,
        validation_probabilities,
    )

    validation_metrics = calculate_metrics(
        validation_labels,
        validation_probabilities,
        threshold,
    )

    (
        test_loss,
        test_labels,
        test_probabilities,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_probabilities,
        threshold,
    )

    runtime_seconds = time.time() - start_time

    seed_model_dir = MODEL_DIR / f"seed_{seed}"
    seed_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_threshold": threshold,
            "state_dict": best_state,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
        seed_model_dir / "best_model.pt",
    )

    pd.DataFrame(history).to_csv(
        HISTORY_DIR / f"seed_{seed}.csv",
        index=False,
    )

    save_predictions(
        validation_dataframe,
        validation_probabilities,
        threshold,
        PREDICTION_DIR
        / f"seed_{seed}_validation.csv",
    )

    save_predictions(
        test_dataframe,
        test_probabilities,
        threshold,
        PREDICTION_DIR
        / f"seed_{seed}_test.csv",
    )

    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "runtime_seconds": runtime_seconds,
        "validation_loss": validation_loss,
        "test_loss": test_loss,
    }

    result.update({
        f"validation_{name}": value
        for name, value
        in validation_metrics.items()
    })

    result.update({
        f"test_{name}": value
        for name, value
        in test_metrics.items()
    })

    print(
        f"[OK] seed={seed} "
        f"best_epoch={best_epoch} "
        f"test_auc={test_metrics['roc_auc']:.5f} "
        f"test_pr_auc={test_metrics['pr_auc']:.5f} "
        f"test_acc={test_metrics['accuracy']:.5f} "
        f"test_mcc={test_metrics['mcc']:.5f}",
        flush=True,
    )

    del model
    torch.cuda.empty_cache()

    return {
        "metrics": result,
        "validation_labels": validation_labels,
        "validation_probabilities": (
            validation_probabilities
        ),
        "test_labels": test_labels,
        "test_probabilities": test_probabilities,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    train_dataframe = load_dataframe(TRAIN_FILE)
    validation_dataframe = load_dataframe(
        VALIDATION_FILE
    )
    test_dataframe = load_dataframe(TEST_FILE)

    device = torch.device("cuda:0")

    print("===== Run information =====", flush=True)
    print(
        "device:",
        torch.cuda.get_device_name(0),
        flush=True,
    )
    print("torch:", torch.__version__, flush=True)
    print(
        "torch_cuda:",
        torch.version.cuda,
        flush=True,
    )
    print(
        "split_sizes:",
        len(train_dataframe),
        len(validation_dataframe),
        len(test_dataframe),
        flush=True,
    )
    print("seeds:", args.seeds, flush=True)
    print(
        "model_parameters:",
        sum(
            parameter.numel()
            for parameter
            in CNN_BiLSTM_Classifier().parameters()
        ),
        flush=True,
    )

    all_results = []
    validation_probabilities_by_seed = []
    test_probabilities_by_seed = []

    validation_labels_reference = None
    test_labels_reference = None

    for seed in args.seeds:
        output = train_one_seed(
            seed=seed,
            train_dataframe=train_dataframe,
            validation_dataframe=(
                validation_dataframe
            ),
            test_dataframe=test_dataframe,
            device=device,
        )

        all_results.append(output["metrics"])

        validation_probabilities_by_seed.append(
            output["validation_probabilities"]
        )

        test_probabilities_by_seed.append(
            output["test_probabilities"]
        )

        validation_labels_reference = (
            output["validation_labels"]
        )

        test_labels_reference = output["test_labels"]

    per_seed = pd.DataFrame(all_results)

    per_seed.to_csv(
        RESULT_DIR / "cnn_bilstm_per_seed_metrics.csv",
        index=False,
    )

    summary_metrics = [
        "test_roc_auc",
        "test_pr_auc",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "test_mcc",
        "test_brier",
        "test_accuracy_at_0p5",
        "test_mcc_at_0p5",
    ]

    summary_rows = []

    for metric_name in summary_metrics:
        values = per_seed[
            metric_name
        ].to_numpy(dtype=float)

        summary_rows.append({
            "metric": metric_name,
            "n_seeds": len(values),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        })

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        RESULT_DIR / "cnn_bilstm_mean_sd.csv",
        index=False,
    )

    validation_matrix = np.vstack(
        validation_probabilities_by_seed
    )

    test_matrix = np.vstack(
        test_probabilities_by_seed
    )

    ensemble_validation_probability = (
        validation_matrix.mean(axis=0)
    )

    ensemble_test_probability = (
        test_matrix.mean(axis=0)
    )

    ensemble_threshold = (
        validation_youden_threshold(
            validation_labels_reference,
            ensemble_validation_probability,
        )
    )

    ensemble_validation_metrics = (
        calculate_metrics(
            validation_labels_reference,
            ensemble_validation_probability,
            ensemble_threshold,
        )
    )

    ensemble_test_metrics = calculate_metrics(
        test_labels_reference,
        ensemble_test_probability,
        ensemble_threshold,
    )

    ensemble_row = {
        "n_seeds": len(args.seeds),
    }

    ensemble_row.update({
        f"validation_{name}": value
        for name, value
        in ensemble_validation_metrics.items()
    })

    ensemble_row.update({
        f"test_{name}": value
        for name, value
        in ensemble_test_metrics.items()
    })

    pd.DataFrame([ensemble_row]).to_csv(
        RESULT_DIR
        / "cnn_bilstm_ensemble_metrics.csv",
        index=False,
    )

    ensemble_predictions = test_dataframe.copy()

    for index, seed in enumerate(args.seeds):
        ensemble_predictions[
            f"probability_seed_{seed}"
        ] = test_matrix[index]

    ensemble_predictions[
        "ensemble_probability"
    ] = ensemble_test_probability

    ensemble_predictions[
        "ensemble_predicted_label"
    ] = (
        ensemble_test_probability
        >= ensemble_threshold
    ).astype(int)

    ensemble_predictions.to_csv(
        PREDICTION_DIR
        / "cnn_bilstm_ensemble_test.csv",
        index=False,
    )

    bootstrap = bootstrap_ensemble(
        test_labels_reference,
        ensemble_test_probability,
        ensemble_threshold,
    )

    bootstrap.to_csv(
        RESULT_DIR
        / "cnn_bilstm_ensemble_bootstrap_ci.csv",
        index=False,
    )

    run_configuration = {
        "seeds": args.seeds,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "data_files": {
            str(path.name): {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in [
                TRAIN_FILE,
                VALIDATION_FILE,
                TEST_FILE,
            ]
        },
        "model": {
            "name": "CNN_BiLSTM_Classifier",
            "maximum_sequence_length": L_MAX,
            "nucleotide_vocabulary": NUC_VOCAB,
            "convolution_filters": CONV_FILTERS,
            "kernel_sizes": KERNEL_SIZES,
            "hidden_dimension": HIDDEN_DIM,
            "dropout": DROPOUT,
            "parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter
                    in CNN_BiLSTM_Classifier().parameters()
                )
            ),
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "maximum_epochs": N_EPOCHS,
            "early_stopping_patience": (
                EARLY_STOP_PATIENCE
            ),
            "model_selection_metric": (
                "validation ROC-AUC"
            ),
            "threshold_selection": (
                "Youden J on validation set"
            ),
        },
        "bootstrap_iterations": BOOTSTRAP_N,
    }

    with (
        RESULT_DIR / "cnn_bilstm_run_config.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            run_configuration,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("===== Five-seed mean ± SD =====")
    print(summary.to_string(index=False))

    print()
    print("===== Ensemble test metrics =====")
    print(
        pd.DataFrame([
            ensemble_test_metrics
        ]).to_string(index=False)
    )

    print()
    print(
        f"[OK] Results saved to {RESULT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
