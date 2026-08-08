from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    mannwhitneyu,
    pearsonr,
    spearmanr,
    ttest_ind,
)

ROOT = Path.home() / "rna_nna_project"
GA_FILE = ROOT / "ga_boltz2_rescore" / "ga60_boltz2_rescore_results.xlsx"
SEED_FILE = ROOT / "seed40_boltz2_rescore" / "seed40_boltz2_rescore_results.xlsx"
OUT_DIR = ROOT / "seed40_boltz2_rescore" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Q90 = 10.76717315
Q95 = 10.82687504
RNG = np.random.default_rng(20260806)
N_BOOT = 10000


def audit(df, expected_n, name):
    if len(df) != expected_n:
        raise ValueError(f"{name}: expected {expected_n}, found {len(df)}")
    if df["dna_sequence"].duplicated().any():
        raise ValueError(f"{name}: duplicate sequences")
    if df["affinity_pic50"].isna().any():
        raise ValueError(f"{name}: missing affinity_pic50")


def summary(name, df):
    score = df["affinity_pic50"].astype(float)
    likelihood = pd.to_numeric(
        df["binding_affinity_likelihood"],
        errors="coerce",
    )

    return {
        "group": name,
        "n": len(df),
        "mean_pic50": score.mean(),
        "sd_pic50": score.std(ddof=1),
        "median_pic50": score.median(),
        "min_pic50": score.min(),
        "max_pic50": score.max(),
        "unique_pic50": score.nunique(),
        "unique_likelihood": likelihood.nunique(),
        "n_likelihood_1": int(np.isclose(likelihood, 1.0).sum()),
        "pct_likelihood_1": np.isclose(likelihood, 1.0).mean() * 100,
        "n_above_q90": int((score >= Q90).sum()),
        "pct_above_q90": (score >= Q90).mean() * 100,
        "n_above_q95": int((score >= Q95).sum()),
        "pct_above_q95": (score >= Q95).mean() * 100,
    }


def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return (
        np.sum(x[:, None] > y[None, :])
        - np.sum(x[:, None] < y[None, :])
    ) / (len(x) * len(y))


def compare_groups(name, ga_values, seed_values):
    ga_values = np.asarray(ga_values, dtype=float)
    seed_values = np.asarray(seed_values, dtype=float)

    mean_deltas = np.empty(N_BOOT)
    median_deltas = np.empty(N_BOOT)

    for i in range(N_BOOT):
        ga_boot = RNG.choice(
            ga_values,
            size=len(ga_values),
            replace=True,
        )
        seed_boot = RNG.choice(
            seed_values,
            size=len(seed_values),
            replace=True,
        )

        mean_deltas[i] = ga_boot.mean() - seed_boot.mean()
        median_deltas[i] = (
            np.median(ga_boot) - np.median(seed_boot)
        )

    welch = ttest_ind(
        ga_values,
        seed_values,
        equal_var=False,
    )

    mw = mannwhitneyu(
        ga_values,
        seed_values,
        alternative="two-sided",
    )

    return {
        "comparison": name,
        "n_ga": len(ga_values),
        "n_seed": len(seed_values),
        "ga_mean": ga_values.mean(),
        "seed_mean": seed_values.mean(),
        "mean_delta_ga_minus_seed":
            ga_values.mean() - seed_values.mean(),
        "mean_delta_boot_ci_low":
            np.quantile(mean_deltas, 0.025),
        "mean_delta_boot_ci_high":
            np.quantile(mean_deltas, 0.975),
        "ga_median": np.median(ga_values),
        "seed_median": np.median(seed_values),
        "median_delta_ga_minus_seed":
            np.median(ga_values) - np.median(seed_values),
        "median_delta_boot_ci_low":
            np.quantile(median_deltas, 0.025),
        "median_delta_boot_ci_high":
            np.quantile(median_deltas, 0.975),
        "welch_t_p": welch.pvalue,
        "mann_whitney_p": mw.pvalue,
        "cliffs_delta_ga_vs_seed":
            cliffs_delta(ga_values, seed_values),
    }


ga = pd.read_excel(GA_FILE)
seed = pd.read_excel(SEED_FILE)

for df in (ga, seed):
    df["dna_sequence"] = (
        df["dna_sequence"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

audit(ga, 60, "GA60")
audit(seed, 40, "Seed40")

group_summary = pd.DataFrame([
    summary("GA60_all", ga),
    summary("Seed40_all", seed),
    summary(
        "Seed40_65nt",
        seed.loc[seed["length_nt"] == 65],
    ),
])

comparisons = pd.DataFrame([
    compare_groups(
        "GA60_vs_Seed40_all",
        ga["affinity_pic50"],
        seed["affinity_pic50"],
    ),
    compare_groups(
        "GA60_vs_Seed40_65nt",
        ga["affinity_pic50"],
        seed.loc[
            seed["length_nt"] == 65,
            "affinity_pic50",
        ],
    ),
])

old_score = pd.to_numeric(
    seed["original_affinity_pic50"],
    errors="coerce",
)
new_score = pd.to_numeric(
    seed["affinity_pic50"],
    errors="coerce",
)

valid = old_score.notna() & new_score.notna()
old_score = old_score[valid].to_numpy()
new_score = new_score[valid].to_numpy()
difference = new_score - old_score

pearson = pearsonr(old_score, new_score)
spearman = spearmanr(old_score, new_score)

bootstrap_mean_change = np.empty(N_BOOT)

for i in range(N_BOOT):
    sample = RNG.choice(
        difference,
        size=len(difference),
        replace=True,
    )
    bootstrap_mean_change[i] = sample.mean()

repeatability = pd.DataFrame([{
    "n": len(difference),
    "old_mean": old_score.mean(),
    "current_mean": new_score.mean(),
    "mean_change_current_minus_old": difference.mean(),
    "mean_change_ci_low":
        np.quantile(bootstrap_mean_change, 0.025),
    "mean_change_ci_high":
        np.quantile(bootstrap_mean_change, 0.975),
    "mae": np.mean(np.abs(difference)),
    "rmse": np.sqrt(np.mean(difference ** 2)),
    "pearson_r": pearson.statistic,
    "pearson_p": pearson.pvalue,
    "spearman_rho": spearman.statistic,
    "spearman_p": spearman.pvalue,
}])

seed_change = seed[
    [
        "candidate_id",
        "dna_sequence",
        "length_nt",
        "original_affinity_pic50",
        "affinity_pic50",
        "binding_affinity_likelihood",
        "confidence_score",
    ]
].copy()

seed_change["pic50_change_current_minus_old"] = (
    seed_change["affinity_pic50"]
    - seed_change["original_affinity_pic50"]
)

seed_change.sort_values(
    "pic50_change_current_minus_old",
    inplace=True,
)

group_summary.to_csv(
    OUT_DIR / "group_summary.csv",
    index=False,
)
comparisons.to_csv(
    OUT_DIR / "group_comparisons.csv",
    index=False,
)
repeatability.to_csv(
    OUT_DIR / "seed40_api_repeatability.csv",
    index=False,
)
seed_change.to_csv(
    OUT_DIR / "seed40_old_vs_current.csv",
    index=False,
)

with pd.ExcelWriter(
    OUT_DIR / "ga60_vs_seed40_analysis.xlsx",
    engine="openpyxl",
) as writer:
    group_summary.to_excel(
        writer,
        sheet_name="group_summary",
        index=False,
    )
    comparisons.to_excel(
        writer,
        sheet_name="comparisons",
        index=False,
    )
    repeatability.to_excel(
        writer,
        sheet_name="repeatability",
        index=False,
    )
    seed_change.to_excel(
        writer,
        sheet_name="seed_old_vs_current",
        index=False,
    )

print("===== GROUP SUMMARY =====")
print(group_summary.to_string(index=False))

print("\n===== GA60 VS SEED40 =====")
print(comparisons.to_string(index=False))

print("\n===== SEED40 OLD VS CURRENT API =====")
print(repeatability.to_string(index=False))

print(
    "\n[OK] Saved:",
    OUT_DIR / "ga60_vs_seed40_analysis.xlsx",
)
