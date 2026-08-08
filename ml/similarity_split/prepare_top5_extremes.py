from pathlib import Path
from itertools import groupby
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "all_merged_with_audit_labels.csv"
OUTDIR = Path(__file__).resolve().parent / "input"

OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

required = [
    "dna_sequence",
    "length_nt",
    "affinity_pic50",
    "class_top5",
    "source_run",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"缺少字段: {missing}")

# 只保留Top/Bottom 5%
extreme = df[df["class_top5"].isin([0, 1])].copy()
extreme = extreme.reset_index(drop=True)

def gc_content(seq):
    seq = str(seq)
    return (seq.count("G") + seq.count("C")) / len(seq)

def max_run(seq):
    return max(len(list(group)) for _, group in groupby(str(seq)))

extreme["seq_id"] = [
    f"SEQ_{i:06d}"
    for i in range(1, len(extreme) + 1)
]

extreme["label"] = extreme["class_top5"].astype(int)
extreme["gc_content"] = extreme["dna_sequence"].map(gc_content)
extreme["max_run"] = extreme["dna_sequence"].map(max_run)

columns = [
    "seq_id",
    "dna_sequence",
    "label",
    "affinity_pic50",
    "length_nt",
    "gc_content",
    "max_run",
    "source_run",
]

extreme[columns].to_csv(
    OUTDIR / "top5_extremes_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)

# FASTA用于CD-HIT聚类
with open(
    OUTDIR / "top5_extremes.fasta",
    "w",
    encoding="utf-8"
) as handle:
    for row in extreme.itertuples(index=False):
        handle.write(
            f">{row.seq_id}|label={row.label}\n"
            f"{row.dna_sequence}\n"
        )

counts = extreme["label"].value_counts().sort_index()

report = []
report.append("Top/Bottom 5% dataset preparation")
report.append("=" * 60)
report.append(f"Total sequences: {len(extreme)}")
report.append(f"Weak label 0: {int(counts.get(0, 0))}")
report.append(f"Strong label 1: {int(counts.get(1, 0))}")
report.append(
    f"Unique sequences: {extreme['dna_sequence'].nunique()}"
)
report.append(
    f"Length range: "
    f"{extreme['length_nt'].min()}-"
    f"{extreme['length_nt'].max()} nt"
)
report.append(
    f"Mean GC: {extreme['gc_content'].mean():.4f}"
)
report.append(
    f"Mean GC label 0: "
    f"{extreme.loc[extreme['label'] == 0, 'gc_content'].mean():.4f}"
)
report.append(
    f"Mean GC label 1: "
    f"{extreme.loc[extreme['label'] == 1, 'gc_content'].mean():.4f}"
)
report.append(
    f"Mean length label 0: "
    f"{extreme.loc[extreme['label'] == 0, 'length_nt'].mean():.4f}"
)
report.append(
    f"Mean length label 1: "
    f"{extreme.loc[extreme['label'] == 1, 'length_nt'].mean():.4f}"
)

report_path = OUTDIR / "top5_extremes_report.txt"
report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("[OK] Top/Bottom 5%数据集生成完成")
print(f"[OK] 总序列数: {len(extreme)}")
print(f"[OK] Metadata: {OUTDIR / 'top5_extremes_metadata.csv'}")
print(f"[OK] FASTA: {OUTDIR / 'top5_extremes.fasta'}")
print(f"[OK] Report: {report_path}")
