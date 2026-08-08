from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

METADATA = (
    ROOT
    / "similarity_split/input/top5_extremes_metadata.csv"
)

CLUSTER_TSV = (
    ROOT
    / "similarity_split/mmseqs_linclust/"
      "identity70/top5_id70_cluster.tsv"
)

OUTDIR = (
    ROOT
    / "similarity_split/splits/id70_seed20260803"
)

SEED = 20260803


def normalize_seq_id(value: str) -> str:
    """删除FASTA标题中的附加标签，例如 |label=0。"""
    return str(value).strip().split("|", 1)[0]


def count_overlap(series_a, series_b) -> int:
    return len(
        set(series_a.astype(str))
        & set(series_b.astype(str))
    )


def describe_split(df: pd.DataFrame, name: str) -> dict:
    return {
        "split": name,
        "n": int(len(df)),
        "label_0": int((df["label"] == 0).sum()),
        "label_1": int((df["label"] == 1).sum()),
        "mean_length": float(df["length_nt"].mean()),
        "sd_length": float(df["length_nt"].std()),
        "mean_gc": float(df["gc_content"].mean()),
        "sd_gc": float(df["gc_content"].std()),
        "unique_sequences": int(
            df["dna_sequence"].nunique()
        ),
        "unique_clusters": int(
            df["cluster_id"].nunique()
        ),
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not METADATA.is_file():
        raise FileNotFoundError(
            f"找不到metadata文件：{METADATA}"
        )

    if not CLUSTER_TSV.is_file():
        raise FileNotFoundError(
            f"找不到cluster文件：{CLUSTER_TSV}"
        )

    metadata = pd.read_csv(METADATA)

    required_columns = {
        "seq_id",
        "dna_sequence",
        "label",
        "affinity_pic50",
        "length_nt",
        "gc_content",
        "max_run",
        "source_run",
    }

    missing_columns = (
        required_columns - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            f"metadata缺少字段：{sorted(missing_columns)}"
        )

    cluster_df = pd.read_csv(
        CLUSTER_TSV,
        sep="\t",
        header=None,
        names=["representative", "member"],
        dtype=str,
    )

    cluster_df["representative"] = (
        cluster_df["representative"]
        .map(normalize_seq_id)
    )

    cluster_df["member"] = (
        cluster_df["member"]
        .map(normalize_seq_id)
    )

    if cluster_df["member"].duplicated().any():
        duplicated = (
            cluster_df.loc[
                cluster_df["member"].duplicated(),
                "member",
            ]
            .head()
            .tolist()
        )
        raise ValueError(
            f"cluster文件中存在重复成员：{duplicated}"
        )

    cluster_map = dict(
        zip(
            cluster_df["member"],
            cluster_df["representative"],
        )
    )

    metadata["cluster_id"] = (
        metadata["seq_id"].map(cluster_map)
    )

    missing_mapping = int(
        metadata["cluster_id"].isna().sum()
    )

    if missing_mapping:
        raise ValueError(
            f"有{missing_mapping}条序列没有cluster映射"
        )

    cluster_sizes = (
        metadata.groupby("cluster_id")
        .size()
        .sort_values(ascending=False)
    )

    maximum_cluster_size = int(
        cluster_sizes.iloc[0]
    )

    multi_member_clusters = int(
        (cluster_sizes > 1).sum()
    )

    print(
        f"[INFO] 输入序列：{len(metadata)}"
    )
    print(
        f"[INFO] cluster数量："
        f"{metadata['cluster_id'].nunique()}"
    )
    print(
        f"[INFO] 多成员cluster："
        f"{multi_member_clusters}"
    )
    print(
        f"[INFO] 最大cluster大小："
        f"{maximum_cluster_size}"
    )

    if maximum_cluster_size != 1:
        raise RuntimeError(
            "检测到多成员cluster，需要使用group-wise划分；"
            "当前脚本只适用于本次全部为单例的结果。"
        )

    # 同时控制分类标签和序列长度。
    metadata["split_stratum"] = (
        metadata["label"].astype(str)
        + "_L"
        + metadata["length_nt"].astype(str)
    )

    stratum_counts = (
        metadata["split_stratum"].value_counts()
    )

    if int(stratum_counts.min()) < 4:
        raise RuntimeError(
            "部分label+length分层样本过少，"
            "无法稳定完成两阶段分层划分。"
        )

    train, temporary = train_test_split(
        metadata,
        test_size=0.30,
        random_state=SEED,
        shuffle=True,
        stratify=metadata["split_stratum"],
    )

    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=SEED + 1,
        shuffle=True,
        stratify=temporary["split_stratum"],
    )

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    overlap_report = {
        "sequence_train_validation": count_overlap(
            train["dna_sequence"],
            validation["dna_sequence"],
        ),
        "sequence_train_test": count_overlap(
            train["dna_sequence"],
            test["dna_sequence"],
        ),
        "sequence_validation_test": count_overlap(
            validation["dna_sequence"],
            test["dna_sequence"],
        ),
        "cluster_train_validation": count_overlap(
            train["cluster_id"],
            validation["cluster_id"],
        ),
        "cluster_train_test": count_overlap(
            train["cluster_id"],
            test["cluster_id"],
        ),
        "cluster_validation_test": count_overlap(
            validation["cluster_id"],
            test["cluster_id"],
        ),
    }

    if any(overlap_report.values()):
        raise RuntimeError(
            f"检测到跨数据集重叠：{overlap_report}"
        )

    output_columns = [
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
    ]

    train[output_columns].to_csv(
        OUTDIR / "train_top5_id70.csv",
        index=False,
    )

    validation[output_columns].to_csv(
        OUTDIR / "validation_top5_id70.csv",
        index=False,
    )

    test[output_columns].to_csv(
        OUTDIR / "test_top5_id70.csv",
        index=False,
    )

    all_assignments = pd.concat(
        [train, validation, test],
        ignore_index=True,
    )

    all_assignments[output_columns].to_csv(
        OUTDIR / "all_split_assignments.csv",
        index=False,
    )

    summary = pd.DataFrame([
        describe_split(train, "train"),
        describe_split(validation, "validation"),
        describe_split(test, "test"),
    ])

    summary.to_csv(
        OUTDIR / "split_summary.csv",
        index=False,
    )

    distribution_by_length = (
        all_assignments
        .groupby(
            ["split", "label", "length_nt"]
        )
        .size()
        .rename("n")
        .reset_index()
    )

    distribution_by_length.to_csv(
        OUTDIR / "distribution_by_label_length.csv",
        index=False,
    )

    distribution_by_source = (
        all_assignments
        .groupby(
            ["split", "source_run", "label"]
        )
        .size()
        .rename("n")
        .reset_index()
    )

    distribution_by_source.to_csv(
        OUTDIR / "distribution_by_source_run.csv",
        index=False,
    )

    audit = {
        "input_sequences": int(len(metadata)),
        "input_clusters": int(
            metadata["cluster_id"].nunique()
        ),
        "identity_threshold": 0.70,
        "minimum_bidirectional_coverage": 0.80,
        "multi_member_clusters": (
            multi_member_clusters
        ),
        "maximum_cluster_size": (
            maximum_cluster_size
        ),
        "random_seeds": [SEED, SEED + 1],
        "stratification": (
            "binary label plus exact sequence length"
        ),
        "observed_split_counts": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "overlap_audit": overlap_report,
    }

    with open(
        OUTDIR / "split_audit.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            audit,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    report_lines = [
        "Similarity-aware Top/Bottom 5% split",
        "=" * 65,
        f"Input sequences: {len(metadata)}",
        (
            "Input clusters: "
            f"{metadata['cluster_id'].nunique()}"
        ),
        (
            "Multi-member clusters: "
            f"{multi_member_clusters}"
        ),
        (
            "Maximum cluster size: "
            f"{maximum_cluster_size}"
        ),
        "",
        summary.to_string(index=False),
        "",
        "Overlap audit",
        "-" * 30,
    ]

    for key, value in overlap_report.items():
        report_lines.append(f"{key}: {value}")

    report_lines.extend([
        "",
        (
            "Stratification: binary label + "
            "exact sequence length"
        ),
        f"Random seeds: {SEED}, {SEED + 1}",
    ])

    report_path = OUTDIR / "split_report.txt"

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("[OK] 数据划分完成")
    print(summary.to_string(index=False))
    print()
    print("[OK] 所有sequence overlap = 0")
    print("[OK] 所有cluster overlap = 0")
    print(f"[OK] 输出目录：{OUTDIR}")


if __name__ == "__main__":
    main()
