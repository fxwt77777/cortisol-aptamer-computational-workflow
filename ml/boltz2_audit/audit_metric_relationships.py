from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data_raw"
AUDIT = Path(__file__).resolve().parent / "output"
DATA = ROOT / "data"

AUDIT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

files = [
    RAW / f"boltz2_dna_hcy_batch_results_run{i}.xlsx"
    for i in range(6)
]

frames = []

for run_id, path in enumerate(files):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_excel(path, engine="openpyxl")
    df["source_run"] = run_id
    df["source_file"] = path.name
    df["source_row"] = np.arange(1, len(df) + 1)
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

all_df["dna_sequence"] = (
    all_df["dna_sequence"]
    .astype(str)
    .str.upper()
    .str.strip()
)

numeric_cols = [
    "length_nt",
    "confidence_score",
    "affinity_pred_value_log10_IC50_uM",
    "IC50_uM",
    "IC50_M",
    "binding_affinity_likelihood",
    "affinity_pic50",
]

for col in numeric_cols:
    all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

# 根据常规定义重新计算字段
all_df["calc_log10_IC50_uM"] = np.log10(all_df["IC50_uM"])
all_df["calc_IC50_M"] = all_df["IC50_uM"] * 1e-6
all_df["calc_pIC50_from_M"] = -np.log10(all_df["IC50_M"])

# 检查潜在的复合评分公式
all_df["pic50_minus_physical_pIC50"] = (
    all_df["affinity_pic50"] -
    all_df["calc_pIC50_from_M"]
)

all_df["candidate_pic50_physical_plus_5conf"] = (
    all_df["calc_pIC50_from_M"] +
    5.0 * all_df["confidence_score"]
)

all_df["residual_candidate_formula"] = (
    all_df["affinity_pic50"] -
    all_df["candidate_pic50_physical_plus_5conf"]
)

# 分位数
analysis_cols = [
    "confidence_score",
    "affinity_pred_value_log10_IC50_uM",
    "IC50_uM",
    "IC50_M",
    "binding_affinity_likelihood",
    "affinity_pic50",
    "calc_pIC50_from_M",
    "pic50_minus_physical_pIC50",
]

quantiles = (
    all_df[analysis_cols]
    .quantile([0.05, 0.10, 0.50, 0.90, 0.95])
    .T
)

quantiles.columns = ["q05", "q10", "median", "q90", "q95"]
quantiles.to_csv(
    AUDIT / "metric_quantiles.csv",
    encoding="utf-8-sig"
)

# 相关矩阵
corr_cols = [
    "length_nt",
    "confidence_score",
    "affinity_pred_value_log10_IC50_uM",
    "IC50_uM",
    "IC50_M",
    "binding_affinity_likelihood",
    "affinity_pic50",
    "calc_pIC50_from_M",
]

corr = all_df[corr_cols].corr(method="pearson")
corr.to_csv(
    AUDIT / "metric_correlations.csv",
    encoding="utf-8-sig"
)

# Top / Bottom 5%标签复现
q05 = all_df["affinity_pic50"].quantile(0.05)
q10 = all_df["affinity_pic50"].quantile(0.10)
q90 = all_df["affinity_pic50"].quantile(0.90)
q95 = all_df["affinity_pic50"].quantile(0.95)

all_df["class_top5"] = -1
all_df.loc[all_df["affinity_pic50"] <= q05, "class_top5"] = 0
all_df.loc[all_df["affinity_pic50"] >= q95, "class_top5"] = 1

all_df["class_top10"] = -1
all_df.loc[all_df["affinity_pic50"] <= q10, "class_top10"] = 0
all_df.loc[all_df["affinity_pic50"] >= q90, "class_top10"] = 1

# 序列质量
all_df["sequence_length_recalc"] = all_df["dna_sequence"].str.len()
all_df["valid_acgt"] = all_df["dna_sequence"].str.fullmatch(r"[ACGT]+")
duplicate_rows = int(all_df["dna_sequence"].duplicated(keep=False).sum())
unique_sequences = int(all_df["dna_sequence"].nunique())

# 保存完整数据
all_df.to_csv(
    DATA / "all_merged_with_audit_labels.csv",
    index=False,
    encoding="utf-8-sig"
)

# 汇总报告
lines = []

lines.append("Boltz-2 metric and label audit")
lines.append("=" * 70)
lines.append(f"Total rows: {len(all_df)}")
lines.append(f"Unique sequences: {unique_sequences}")
lines.append(f"Rows involved in duplicated sequences: {duplicate_rows}")
lines.append(f"Valid ACGT rows: {int(all_df['valid_acgt'].sum())}")
lines.append("")

lines.append("Historical label reproduction")
lines.append("-" * 70)
lines.append(f"affinity_pic50 q05: {q05:.8f}")
lines.append(f"affinity_pic50 q10: {q10:.8f}")
lines.append(f"affinity_pic50 q90: {q90:.8f}")
lines.append(f"affinity_pic50 q95: {q95:.8f}")
lines.append("")
lines.append("class_top5 counts:")
lines.append(all_df["class_top5"].value_counts().sort_index().to_string())
lines.append("")
lines.append("class_top10 counts:")
lines.append(all_df["class_top10"].value_counts().sort_index().to_string())
lines.append("")

lines.append("Field consistency checks")
lines.append("-" * 70)

log_resid = (
    all_df["affinity_pred_value_log10_IC50_uM"] -
    all_df["calc_log10_IC50_uM"]
).abs()

molar_resid = (
    all_df["IC50_M"] -
    all_df["calc_IC50_M"]
).abs()

lines.append(
    "max abs difference: "
    "affinity_pred_value_log10_IC50_uM vs log10(IC50_uM) = "
    f"{log_resid.max():.12g}"
)

lines.append(
    "max abs difference: "
    "IC50_M vs IC50_uM*1e-6 = "
    f"{molar_resid.max():.12g}"
)

lines.append(
    "mean physical pIC50 from IC50_M = "
    f"{all_df['calc_pIC50_from_M'].mean():.8f}"
)

lines.append(
    "mean affinity_pic50 = "
    f"{all_df['affinity_pic50'].mean():.8f}"
)

lines.append(
    "mean affinity_pic50 - physical pIC50 = "
    f"{all_df['pic50_minus_physical_pIC50'].mean():.8f}"
)

lines.append(
    "candidate formula residual "
    "[affinity_pic50 - (physical pIC50 + 5*confidence_score)]:"
)

lines.append(
    f"  mean = {all_df['residual_candidate_formula'].mean():.12g}"
)

lines.append(
    f"  mean absolute = "
    f"{all_df['residual_candidate_formula'].abs().mean():.12g}"
)

lines.append(
    f"  max absolute = "
    f"{all_df['residual_candidate_formula'].abs().max():.12g}"
)

report_path = AUDIT / "metric_formula_audit.txt"
report_path.write_text("\n".join(lines), encoding="utf-8")

print("[OK] 审计完成")
print(f"[OK] q05 = {q05:.8f}")
print(f"[OK] q10 = {q10:.8f}")
print(f"[OK] q90 = {q90:.8f}")
print(f"[OK] q95 = {q95:.8f}")
print(f"[OK] 完整报告: {report_path}")
print(f"[OK] 分位数表: {AUDIT / 'metric_quantiles.csv'}")
print(f"[OK] 相关矩阵: {AUDIT / 'metric_correlations.csv'}")
print(f"[OK] 合并数据: {DATA / 'all_merged_with_audit_labels.csv'}")
