import pandas as pd
import glob
import re
from pathlib import Path

root = Path.home() / "aptamer_hcy_project"

aptamer_dirs = [
    "apt1-new",
    "apt2-new",
    "apt3-new",
    "apt4-new",
    "apt5-new",
    "aptC_clean"
]

records = []
detail_tables = {}

for apt in aptamer_dirs:

    files = sorted(glob.glob(
        str(root / apt / "gmx/mmpbsa_parallel_runs/k*/FINAL_RESULTS_k*.dat")
    ))

    rows = []

    for f in files:

        k = re.search(r'k(\d+)', f).group(1)

        with open(f) as fh:
            lines = fh.readlines()

        for line in lines:
            if "DELTA TOTAL" in line or "ΔTOTAL" in line:
                val = float(line.split()[-1])
                rows.append({"aptamer": apt, "replicate": int(k), "dG": val})
                records.append({"aptamer": apt, "replicate": int(k), "dG": val})

    detail_tables[apt] = pd.DataFrame(rows)

summary = (
    pd.DataFrame(records)
    .groupby("aptamer")["dG"]
    .agg(["mean", "std", "min", "max", "count"])
    .reset_index()
)

writer = pd.ExcelWriter("MMGBSA_all_results.xlsx", engine="openpyxl")

summary.to_excel(writer, sheet_name="SUMMARY", index=False)

for apt, df in detail_tables.items():
    df.to_excel(writer, sheet_name=apt, index=False)

writer.close()

print("Excel written: MMGBSA_all_results.xlsx")
