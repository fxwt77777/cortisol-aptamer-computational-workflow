from pathlib import Path
import hashlib

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    ROOT
    / "similarity_split/splits/id70_seed20260803"
)

OUTPUT_DIR = ROOT / "cnn_bilstm/data"

FILES = {
    "train": (
        SOURCE_DIR / "train_top5_id70.csv",
        OUTPUT_DIR / "train_top5.csv",
    ),
    "validation": (
        SOURCE_DIR / "validation_top5_id70.csv",
        OUTPUT_DIR / "val_top5.csv",
    ),
    "test": (
        SOURCE_DIR / "test_top5_id70.csv",
        OUTPUT_DIR / "test_top5.csv",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_sequences = {}
    report_rows = []

    for split_name, (source, output) in FILES.items():
        dataframe = pd.read_csv(source)

        required = {
            "seq_id",
            "cluster_id",
            "dna_sequence",
            "label",
            "affinity_pic50",
            "length_nt",
            "gc_content",
            "max_run",
            "source_run",
            "split",
        }

        missing = required - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{source.name}缺少字段：{sorted(missing)}"
            )

        dataframe["dna_sequence"] = (
            dataframe["dna_sequence"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        dataframe["class_top5"] = (
            dataframe["label"].astype(int)
        )

        dataframe["seq_len"] = (
            dataframe["dna_sequence"].str.len()
        )

        if not (
            dataframe["seq_len"]
            == dataframe["length_nt"]
        ).all():
            raise ValueError(
                f"{split_name}中seq_len与length_nt不一致"
            )

        output_columns = [
            "seq_id",
            "cluster_id",
            "dna_sequence",
            "length_nt",
            "seq_len",
            "gc_content",
            "max_run",
            "affinity_pic50",
            "source_run",
            "class_top5",
            "split",
        ]

        dataframe[output_columns].to_csv(
            output,
            index=False,
        )

        sequences = set(dataframe["dna_sequence"])
        all_sequences[split_name] = sequences

        report_rows.append({
            "split": split_name,
            "n": len(dataframe),
            "label_0": int(
                (dataframe["class_top5"] == 0).sum()
            ),
            "label_1": int(
                (dataframe["class_top5"] == 1).sum()
            ),
            "unique_sequences": (
                dataframe["dna_sequence"].nunique()
            ),
            "unique_clusters": (
                dataframe["cluster_id"].nunique()
            ),
            "mean_length": float(
                dataframe["seq_len"].mean()
            ),
            "mean_gc": float(
                dataframe["gc_content"].mean()
            ),
            "output_file": str(output),
            "sha256": sha256(output),
        })

    overlaps = {
        "train_validation": len(
            all_sequences["train"]
            & all_sequences["validation"]
        ),
        "train_test": len(
            all_sequences["train"]
            & all_sequences["test"]
        ),
        "validation_test": len(
            all_sequences["validation"]
            & all_sequences["test"]
        ),
    }

    if any(overlaps.values()):
        raise RuntimeError(
            f"检测到序列重叠：{overlaps}"
        )

    report = pd.DataFrame(report_rows)

    report.to_csv(
        OUTPUT_DIR / "revised_split_manifest.csv",
        index=False,
    )

    print(report.to_string(index=False))

    print()
    print("Sequence overlap:")
    for key, value in overlaps.items():
        print(f"{key}: {value}")

    print()
    print("[OK] 修订版CNN数据已生成")
    print("[OK] 标签字段：class_top5")
    print("[OK] 所有跨集合序列重叠为0")


if __name__ == "__main__":
    main()
