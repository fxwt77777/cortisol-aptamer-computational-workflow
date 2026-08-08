from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path.home() / "rna_nna_project"

RESULT_FILE = (
    ROOT
    / "ga_boltz2_rescore"
    / "ga60_boltz2_rescore_results.xlsx"
)

INPUT_FILE = (
    ROOT
    / "legacy_original_20251225"
    / "data"
    / "ga60_boltz2_rescore_input.csv"
)

OUTPUT_DIR = ROOT / "ga_boltz2_rescore" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

Q05 = 10.03761287
Q10 = 10.15805178
Q90 = 10.76717315
Q95 = 10.82687504


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .fillna(False)
        .astype(bool)
    )


def safe_correlation(x, y, method):
    valid = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).dropna()

    if len(valid) < 3:
        return np.nan, np.nan

    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan, np.nan

    if method == "pearson":
        return pearsonr(valid["x"], valid["y"])

    if method == "spearman":
        return spearmanr(valid["x"], valid["y"])

    raise ValueError(method)


results = pd.read_excel(RESULT_FILE)
source = pd.read_csv(INPUT_FILE)

results["dna_sequence"] = (
    results["dna_sequence"]
    .astype(str)
    .str.upper()
    .str.strip()
)

source["sequence"] = (
    source["sequence"]
    .astype(str)
    .str.upper()
    .str.strip()
)

required_result_columns = {
    "candidate_id",
    "dna_sequence",
    "affinity_pic50",
    "confidence_score",
    "binding_affinity_likelihood",
}

missing = required_result_columns - set(results.columns)

if missing:
    raise ValueError(
        "Result file missing columns: "
        + ", ".join(sorted(missing))
    )

if len(results) != 60:
    raise ValueError(
        f"Expected 60 result rows, found {len(results)}"
    )

if results["candidate_id"].duplicated().any():
    raise ValueError("Duplicate candidate_id values detected")

if results["dna_sequence"].duplicated().any():
    raise ValueError("Duplicate sequences detected")

if results["affinity_pic50"].isna().any():
    raise ValueError("Missing affinity_pic50 values detected")

metadata_columns = [
    "candidate_id",
    "sequence",
    "score",
    "pred_strong_prob_cls",
    "GC_content",
    "max_run",
    "in_diverse10",
    "in_best5",
]

metadata_columns = [
    c for c in metadata_columns
    if c in source.columns
]

source_metadata = source[metadata_columns].copy()

merged = results.merge(
    source_metadata,
    left_on=["candidate_id", "dna_sequence"],
    right_on=["candidate_id", "sequence"],
    how="left",
    validate="one_to_one",
    suffixes=("", "_input"),
)

if merged["sequence"].isna().any():
    raise ValueError(
        "Some Boltz-2 results could not be matched "
        "to the original GA input"
    )

for column in ["in_diverse10", "in_best5"]:
    if column in merged.columns:
        merged[column] = normalize_bool(merged[column])

merged["above_original_q90"] = (
    merged["affinity_pic50"] >= Q90
)

merged["above_original_q95"] = (
    merged["affinity_pic50"] >= Q95
)

merged["below_original_q10"] = (
    merged["affinity_pic50"] <= Q10
)

merged["below_original_q05"] = (
    merged["affinity_pic50"] <= Q05
)


# Locate the original 114,371-sequence score distribution.
original_candidates = [
    ROOT
    / "legacy_original_20251225"
    / "data"
    / "all_merged_clean.csv",

    ROOT
    / "data"
    / "all_merged_clean.csv",
]

original_path = None
original_scores = None

for path in original_candidates:
    if not path.exists():
        continue

    try:
        original_df = pd.read_csv(
            path,
            usecols=["affinity_pic50"],
        )

        values = pd.to_numeric(
            original_df["affinity_pic50"],
            errors="coerce",
        ).dropna()

        if len(values) >= 100000:
            original_path = path
            original_scores = values.to_numpy(dtype=float)
            break

    except Exception:
        continue


if original_scores is not None:
    original_sorted = np.sort(original_scores)

    merged["original_empirical_percentile"] = (
        np.searchsorted(
            original_sorted,
            merged["affinity_pic50"].to_numpy(),
            side="right",
        )
        / len(original_sorted)
        * 100.0
    )
else:
    merged["original_empirical_percentile"] = np.nan


def summarize_group(name: str, frame: pd.DataFrame) -> dict:
    scores = frame["affinity_pic50"].astype(float)

    return {
        "group": name,
        "n": len(frame),
        "mean_affinity_pic50": scores.mean(),
        "sd_affinity_pic50": scores.std(ddof=1),
        "median_affinity_pic50": scores.median(),
        "min_affinity_pic50": scores.min(),
        "max_affinity_pic50": scores.max(),
        "q25_affinity_pic50": scores.quantile(0.25),
        "q75_affinity_pic50": scores.quantile(0.75),
        "n_above_original_q90": int(
            frame["above_original_q90"].sum()
        ),
        "pct_above_original_q90": (
            frame["above_original_q90"].mean() * 100
        ),
        "n_above_original_q95": int(
            frame["above_original_q95"].sum()
        ),
        "pct_above_original_q95": (
            frame["above_original_q95"].mean() * 100
        ),
        "mean_original_percentile": (
            frame["original_empirical_percentile"].mean()
        ),
    }


group_rows = [
    summarize_group("GA_all_60", merged),
    summarize_group(
        "GA_diverse10",
        merged.loc[merged["in_diverse10"]],
    ),
    summarize_group(
        "GA_best5",
        merged.loc[merged["in_best5"]],
    ),
]

group_summary = pd.DataFrame(group_rows)


correlation_rows = []

for predictor in [
    "pred_strong_prob_cls",
    "score",
    "confidence_score",
    "binding_affinity_likelihood",
]:
    if predictor not in merged.columns:
        continue

    pearson_r, pearson_p = safe_correlation(
        merged[predictor],
        merged["affinity_pic50"],
        "pearson",
    )

    spearman_rho, spearman_p = safe_correlation(
        merged[predictor],
        merged["affinity_pic50"],
        "spearman",
    )

    correlation_rows.append({
        "predictor": predictor,
        "unique_values": merged[predictor].nunique(),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    })

correlations = pd.DataFrame(correlation_rows)


threshold_summary = pd.DataFrame([
    {
        "threshold_name": "original_q05",
        "threshold": Q05,
        "ga_count_at_or_below": int(
            (merged["affinity_pic50"] <= Q05).sum()
        ),
        "ga_count_at_or_above": int(
            (merged["affinity_pic50"] >= Q05).sum()
        ),
    },
    {
        "threshold_name": "original_q10",
        "threshold": Q10,
        "ga_count_at_or_below": int(
            (merged["affinity_pic50"] <= Q10).sum()
        ),
        "ga_count_at_or_above": int(
            (merged["affinity_pic50"] >= Q10).sum()
        ),
    },
    {
        "threshold_name": "original_q90",
        "threshold": Q90,
        "ga_count_at_or_below": int(
            (merged["affinity_pic50"] <= Q90).sum()
        ),
        "ga_count_at_or_above": int(
            (merged["affinity_pic50"] >= Q90).sum()
        ),
    },
    {
        "threshold_name": "original_q95",
        "threshold": Q95,
        "ga_count_at_or_below": int(
            (merged["affinity_pic50"] <= Q95).sum()
        ),
        "ga_count_at_or_above": int(
            (merged["affinity_pic50"] >= Q95).sum()
        ),
    },
])


ranked = merged.sort_values(
    [
        "affinity_pic50",
        "confidence_score",
    ],
    ascending=[False, False],
).reset_index(drop=True)

ranked.insert(
    0,
    "boltz2_rescore_rank",
    np.arange(1, len(ranked) + 1),
)


ranked.to_csv(
    OUTPUT_DIR / "ga60_rescore_ranked.csv",
    index=False,
)

group_summary.to_csv(
    OUTPUT_DIR / "ga60_group_summary.csv",
    index=False,
)

threshold_summary.to_csv(
    OUTPUT_DIR / "ga60_threshold_summary.csv",
    index=False,
)

correlations.to_csv(
    OUTPUT_DIR / "ga60_score_correlations.csv",
    index=False,
)


excel_path = (
    OUTPUT_DIR
    / "ga60_boltz2_rescore_analysis.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl",
) as writer:
    ranked.to_excel(
        writer,
        sheet_name="ranked_60",
        index=False,
    )

    ranked.loc[
        ranked["in_diverse10"]
    ].to_excel(
        writer,
        sheet_name="diverse10",
        index=False,
    )

    ranked.loc[
        ranked["in_best5"]
    ].to_excel(
        writer,
        sheet_name="best5",
        index=False,
    )

    group_summary.to_excel(
        writer,
        sheet_name="group_summary",
        index=False,
    )

    threshold_summary.to_excel(
        writer,
        sheet_name="thresholds",
        index=False,
    )

    correlations.to_excel(
        writer,
        sheet_name="correlations",
        index=False,
    )


print("===== DATA AUDIT =====")
print("GA results:", len(merged))
print("Unique sequences:", merged["dna_sequence"].nunique())
print("Missing affinity_pic50:", merged["affinity_pic50"].isna().sum())
print("Original distribution:", original_path)
print()

print("===== GROUP SUMMARY =====")
print(group_summary.to_string(index=False))
print()

print("===== SCORE CORRELATIONS =====")
print(correlations.to_string(index=False))
print()

print("===== TOP 15 AFTER ORIGINAL BOLTZ-2 RESCORING =====")

display_columns = [
    "boltz2_rescore_rank",
    "candidate_id",
    "affinity_pic50",
    "original_empirical_percentile",
    "confidence_score",
    "pred_strong_prob_cls",
    "in_diverse10",
    "in_best5",
]

print(
    ranked[display_columns]
    .head(15)
    .to_string(index=False)
)

print()
print("[OK] Saved:", excel_path)
