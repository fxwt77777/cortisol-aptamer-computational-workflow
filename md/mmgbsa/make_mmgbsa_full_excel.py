import glob
import pandas as pd
import os

root=os.environ.get("APTAMER_PROJECT_ROOT", ".")

rows=[]

for aptdir in glob.glob(root+"/apt*"):

    apt=os.path.basename(aptdir)

    if "extend" in apt or "long" in apt:
        run_type="extended"
    else:
        run_type="original"

    pattern=f"{aptdir}/gmx/mmpbsa_parallel_runs/k*/FINAL_RESULTS_k*.dat"

    files=glob.glob(pattern)

    for f in files:

        rep=os.path.basename(f).split("_k")[1].split(".")[0]

        with open(f) as fh:
            for line in fh:
                if "ΔTOTAL" in line:
                    dg=float(line.split()[1])

                    rows.append([
                        apt,
                        run_type,
                        rep,
                        dg
                    ])

df=pd.DataFrame(rows,columns=[
"aptamer",
"run_type",
"replicate",
"dG_kcal_mol"
])

summary=df.groupby(["aptamer","run_type"])["dG_kcal_mol"].agg(
mean="mean",
SD="std",
min="min",
max="max",
n="count"
).reset_index()

outfile=os.environ.get("MMGBSA_OUT", "MMGBSA_FULL_dataset.xlsx")

writer=pd.ExcelWriter(outfile)

df.to_excel(writer, sheet_name="raw_data", index=False)
summary.to_excel(writer, sheet_name="summary", index=False)

writer.close()

print("Excel created:",outfile)
