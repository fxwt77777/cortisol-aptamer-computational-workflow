from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV = REPO_ROOT / "data" / "all_merged_with_audit_labels.csv"
OUT = Path(__file__).resolve().parent / "output" / "formula_reverse_engineering.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

physical = -np.log10(df["IC50_M"].astype(float))
conf = df["confidence_score"].astype(float).to_numpy()
like = df["binding_affinity_likelihood"].astype(float).to_numpy()
target = df["affinity_pic50"].astype(float).to_numpy()
bonus = target - physical.to_numpy()

eps = 1e-9
logit_like = np.log(
    np.clip(like, eps, 1-eps) /
    (1-np.clip(like, eps, 1-eps))
)

def evaluate(name, X, feature_names):
    X = np.column_stack([np.ones(len(X)), X])
    names = ["intercept"] + feature_names

    coef, _, _, _ = np.linalg.lstsq(X, bonus, rcond=None)
    pred = X @ coef
    resid = bonus - pred

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((bonus - bonus.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    lines = []
    lines.append(f"MODEL: {name}")
    for n, c in zip(names, coef):
        lines.append(f"  {n}: {c:.12f}")
    lines.append(f"  R2: {r2:.12f}")
    lines.append(f"  MAE: {np.mean(np.abs(resid)):.12f}")
    lines.append(f"  RMSE: {np.sqrt(np.mean(resid**2)):.12f}")
    lines.append(f"  MAX_ABS: {np.max(np.abs(resid)):.12f}")
    return "\n".join(lines)

reports = []

reports.append("Affinity_pic50 formula reverse engineering")
reports.append("=" * 80)

reports.append("\nCORRELATIONS WITH BONUS = affinity_pic50 - physical_pIC50")
reports.append(f"confidence_score: {np.corrcoef(bonus, conf)[0,1]:.12f}")
reports.append(f"binding_affinity_likelihood: {np.corrcoef(bonus, like)[0,1]:.12f}")
reports.append(f"confidence*likelihood: {np.corrcoef(bonus, conf*like)[0,1]:.12f}")
reports.append(f"logit(likelihood): {np.corrcoef(bonus, logit_like)[0,1]:.12f}")

models = [
    (
        "confidence_only",
        np.column_stack([conf]),
        ["confidence"]
    ),
    (
        "confidence_and_likelihood",
        np.column_stack([conf, like]),
        ["confidence", "likelihood"]
    ),
    (
        "confidence_times_likelihood",
        np.column_stack([conf * like]),
        ["confidence_x_likelihood"]
    ),
    (
        "interaction_model",
        np.column_stack([conf, like, conf * like]),
        ["confidence", "likelihood", "confidence_x_likelihood"]
    ),
    (
        "quadratic_model",
        np.column_stack([
            conf,
            like,
            conf * like,
            conf ** 2,
            like ** 2
        ]),
        [
            "confidence",
            "likelihood",
            "confidence_x_likelihood",
            "confidence_squared",
            "likelihood_squared"
        ]
    ),
    (
        "logit_likelihood_model",
        np.column_stack([
            conf,
            logit_like,
            conf * logit_like
        ]),
        [
            "confidence",
            "logit_likelihood",
            "confidence_x_logit_likelihood"
        ]
    )
]

for name, X, names in models:
    reports.append("")
    reports.append(evaluate(name, X, names))

reports.append("")
reports.append("DIRECT CANDIDATE FORMULAS")
reports.append("-" * 80)

candidates = {
    "physical + 5*confidence":
        physical.to_numpy() + 5*conf,

    "physical + 5*confidence*likelihood":
        physical.to_numpy() + 5*conf*like,

    "physical + 5*confidence + likelihood":
        physical.to_numpy() + 5*conf + like,

    "physical + 5*confidence + logit(likelihood)":
        physical.to_numpy() + 5*conf + logit_like,

    "physical + confidence*logit(likelihood)":
        physical.to_numpy() + conf*logit_like,
}

for name, pred in candidates.items():
    resid = target - pred
    reports.append(
        f"{name}\n"
        f"  mean residual: {resid.mean():.12f}\n"
        f"  MAE: {np.mean(np.abs(resid)):.12f}\n"
        f"  max abs: {np.max(np.abs(resid)):.12f}"
    )

OUT.write_text("\n".join(reports), encoding="utf-8")

print("[OK] 公式反推完成")
print(f"[OK] 结果保存至: {OUT}")
