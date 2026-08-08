#!/usr/bin/env bash
set -eo pipefail

ROOT="${APTAMER_PROJECT_ROOT:-$(pwd)}"
OUT="${IONMIX_OUT:-$ROOT/analysis_output/ionmix_final_all}"
export IONMIX_OUT="$OUT"

mkdir -p "$OUT"/{tables,plots,pdb,logs,raw_xvg,raw_mmgbsa}

# final input directories
APTC_DIR="${APTC_DIR:-$ROOT/ionmix_final_aptC30_apt3apt4_50_20260521/aptC_ionmix_30ns}"
APT3_DIR="${APT3_DIR:-$ROOT/ionmix_boundstart_apt3apt4_50_20260524_144252/apt3_ionmix_boundstart_50ns/gmx}"
APT4_DIR="${APT4_DIR:-$ROOT/ionmix_boundstart_apt3apt4_50_20260524_144252/apt4_ionmix_boundstart_50ns/gmx}"

# name:dir:tpr:xtc:top:ndx:ns
RUNS=(
  "aptC:${APTC_DIR}:aptC_ionmix_30ns.tpr:aptC_ionmix_30ns.xtc:system.top:index_clean.ndx:30"
  "apt3:${APT3_DIR}:md_ionmix_boundstart_50ns.tpr:md_ionmix_boundstart_50ns.xtc:system.top:index_clean.ndx:50"
  "apt4:${APT4_DIR}:md_ionmix_boundstart_50ns.tpr:md_ionmix_boundstart_50ns.xtc:system.top:index_clean.ndx:50"
)

echo "name	ns	frames	last_time_ps" > "$OUT/tables/trajectory_check.tsv"

############################################
# 1. GROMACS post-processing
############################################

# Run this script from an environment where GROMACS is already available.
set +u
# GROMACS environment activation is intentionally left to the user.
set -u

for item in "${RUNS[@]}"; do
  name="$(echo "$item" | cut -d: -f1)"
  dir="$(echo "$item" | cut -d: -f2)"
  tpr="$(echo "$item" | cut -d: -f3)"
  xtc="$(echo "$item" | cut -d: -f4)"
  top="$(echo "$item" | cut -d: -f5)"
  ndx="$(echo "$item" | cut -d: -f6)"
  ns="$(echo "$item" | cut -d: -f7)"

  echo "=============================="
  echo "[GROMACS] $name ${ns} ns"
  echo "dir=$dir"
  echo "=============================="

  cd "$dir"
  mkdir -p analysis snapshots mmpbsa_fixed mmpbsa_parallel_runs

  # trajectory check
  gmx check -f "$xtc" > analysis/${name}_gmx_check.txt 2>&1 || true
  frames=$(grep -A4 "Item" analysis/${name}_gmx_check.txt | awk '$1=="Step"{print $2}' || echo "NA")
  last_time=$(grep "Last frame" analysis/${name}_gmx_check.txt | awk '{print $5}' || echo "NA")
  echo -e "${name}\t${ns}\t${frames}\t${last_time}" >> "$OUT/tables/trajectory_check.tsv"

  # mindist
  printf "DNA\nLIG\n" | gmx mindist \
    -s "$tpr" \
    -f "$xtc" \
    -n "$ndx" \
    -od analysis/${name}_mindist_DNA_LIG.xvg \
    -on analysis/${name}_contacts_num_0p35.xvg \
    -d 0.35 \
    > analysis/${name}_mindist.log 2>&1

  cp analysis/${name}_mindist_DNA_LIG.xvg "$OUT/raw_xvg/"
  cp analysis/${name}_contacts_num_0p35.xvg "$OUT/raw_xvg/"

  awk -v nm="$name" '
  !/^[@#]/{
    t=$1; d=$2;
    n++; s+=d;
    if(n==1 || d<mn){mn=d; mt=t}
    if(n==1 || d>mx){mx=d}
    if(d<0.35)c++
  }
  END{
    printf "%s\t%.4f\t%.4f\t%.1f\t%.4f\t%.4f\t%d\n", nm, s/n, mn, mt, mx, c/n, n
  }' analysis/${name}_mindist_DNA_LIG.xvg >> "$OUT/tables/mindist_summary.tmp"

  BEST_PS=$(awk '
  !/^[@#]/{
    t=$1; d=$2;
    n++;
    if(n==1 || d<mn){mn=d; mt=t}
  }
  END{printf "%.0f", mt}
  ' analysis/${name}_mindist_DNA_LIG.xvg)

  LAST_PS=$((ns * 1000))

  # PDB: best contact
  printf "DNA_LIG\nDNA_LIG\n" | gmx trjconv \
    -s "$tpr" \
    -f "$xtc" \
    -n "$ndx" \
    -o snapshots/${name}_ionmix_best_contact_${BEST_PS}ps_DNA_LIG_centered.pdb \
    -dump "$BEST_PS" \
    -pbc mol -center -ur compact \
    > analysis/${name}_trjconv_best.log 2>&1

  # PDB: last frame
  printf "DNA_LIG\nDNA_LIG\n" | gmx trjconv \
    -s "$tpr" \
    -f "$xtc" \
    -n "$ndx" \
    -o snapshots/${name}_ionmix_lastframe_${LAST_PS}ps_DNA_LIG_centered.pdb \
    -dump "$LAST_PS" \
    -pbc mol -center -ur compact \
    > analysis/${name}_trjconv_last.log 2>&1

  cp snapshots/${name}_ionmix_*_DNA_LIG_centered.pdb "$OUT/pdb/"

  # RMSD DNA
  printf "DNA\nDNA\n" | gmx rms \
    -s "$tpr" \
    -f "$xtc" \
    -n "$ndx" \
    -o analysis/${name}_RMSD_DNA.xvg \
    -tu ns \
    > analysis/${name}_rmsd_DNA.log 2>&1

  # RMSD ligand after DNA fit
  printf "DNA\nLIG\n" | gmx rms \
    -s "$tpr" \
    -f "$xtc" \
    -n "$ndx" \
    -o analysis/${name}_RMSD_LIG_fitDNA.xvg \
    -tu ns \
    > analysis/${name}_rmsd_LIG_fitDNA.log 2>&1

  cp analysis/${name}_RMSD_DNA.xvg "$OUT/raw_xvg/"
  cp analysis/${name}_RMSD_LIG_fitDNA.xvg "$OUT/raw_xvg/"

  # MMGBSA trajectory: DNA+LIG only, 50 ps/frame
  printf "DNA_LIG\nDNA_LIG\n" | gmx trjconv \
    -s "$tpr" \
    -f "$xtc" \
    -n "$ndx" \
    -o mmpbsa_fixed/${name}_xtc_DNA_LIG_50ps.xtc \
    -pbc mol -center -ur compact -dt 50 \
    > analysis/${name}_trjconv_mmpbsa_50ps.log 2>&1

  cp "$tpr" mmpbsa_fixed/${name}_mmgbsa.tpr
  cp "$top" mmpbsa_fixed/${name}_mmgbsa_system.top
  cp "$ndx" mmpbsa_fixed/${name}_mmgbsa_index.ndx

done

# mindist summary header
{
  echo -e "name\tmean_nm\tmin_nm\tmin_time_ps\tmax_nm\tcontact_fraction\tn"
  cat "$OUT/tables/mindist_summary.tmp"
} > "$OUT/tables/mindist_summary.tsv"
rm "$OUT/tables/mindist_summary.tmp"

############################################
# 2. MM/GBSA k1-k6
############################################

set +u
conda activate ambertools
set -u

echo -e "name\tk\tDG_TOTAL_kcal_per_mol" > "$OUT/tables/mmgbsa_dtotal_by_k.tsv"

for item in "${RUNS[@]}"; do
  name="$(echo "$item" | cut -d: -f1)"
  dir="$(echo "$item" | cut -d: -f2)"
  ns="$(echo "$item" | cut -d: -f7)"

  echo "=============================="
  echo "[MMGBSA] $name"
  echo "=============================="

  cd "$dir"

  for k in 1 2 3 4 5 6; do
    mkdir -p mmpbsa_parallel_runs/k${k}

    cat > mmpbsa_parallel_runs/k${k}/gbsa_k${k}.in <<EOF
Input file for running GBSA
&general
  startframe=${k},
  endframe=999999,
  interval=1,
  verbose=1,
/
&gb
  igb=5,
  saltcon=0.150,
/
EOF

    cd mmpbsa_parallel_runs/k${k}

    gmx_MMPBSA -O \
      -i gbsa_k${k}.in \
      -cs ../../mmpbsa_fixed/${name}_mmgbsa.tpr \
      -ct ../../mmpbsa_fixed/${name}_xtc_DNA_LIG_50ps.xtc \
      -cp ../../mmpbsa_fixed/${name}_mmgbsa_system.top \
      -ci ../../mmpbsa_fixed/${name}_mmgbsa_index.ndx \
      -cg 0 1 \
      -prefix _GMXMMPBSA_${name}_k${k}_ \
      -o FINAL_RESULTS_k${k}.dat \
      -eo FINAL_RESULTS_k${k}.csv \
      -nogui > run_k${k}.log 2>&1

    dg=$(grep "^ΔTOTAL" FINAL_RESULTS_k${k}.dat | awk '{print $2}')
    echo -e "${name}\tk${k}\t${dg}" >> "$OUT/tables/mmgbsa_dtotal_by_k.tsv"

    mkdir -p "$OUT/raw_mmgbsa/${name}/k${k}"
    cp FINAL_RESULTS_k${k}.dat FINAL_RESULTS_k${k}.csv run_k${k}.log "$OUT/raw_mmgbsa/${name}/k${k}/"

    cd ../../
  done
done

############################################
# 3. Summaries + plots
############################################

set +u
# GROMACS environment activation is intentionally left to the user.
set -u

python3 - <<'PY'
from pathlib import Path
import os
import math
import statistics as stats
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(os.environ["IONMIX_OUT"])
RAW = OUT / "raw_xvg"
PLOTS = OUT / "plots"
TABLES = OUT / "tables"
PLOTS.mkdir(exist_ok=True)

def read_xvg(path):
    xs, ys = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("@") or line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
    return xs, ys

def save_combined_plot(patterns, title, ylabel, outname, xlabel="Time"):
    plt.figure(figsize=(8, 5))
    for name, path in patterns:
        x, y = read_xvg(path)
        plt.plot(x, y, label=name, linewidth=1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / outname, dpi=300)
    plt.close()

# mindist combined, x in ps
save_combined_plot(
    [
        ("aptC", RAW / "aptC_mindist_DNA_LIG.xvg"),
        ("apt3", RAW / "apt3_mindist_DNA_LIG.xvg"),
        ("apt4", RAW / "apt4_mindist_DNA_LIG.xvg"),
    ],
    "DNA-LIG minimum distance under ion-mixed condition",
    "Minimum distance (nm)",
    "combined_mindist_DNA_LIG.png",
    xlabel="Time (ps)"
)

# RMSD combined, x in ns
save_combined_plot(
    [
        ("aptC", RAW / "aptC_RMSD_DNA.xvg"),
        ("apt3", RAW / "apt3_RMSD_DNA.xvg"),
        ("apt4", RAW / "apt4_RMSD_DNA.xvg"),
    ],
    "DNA RMSD under ion-mixed condition",
    "RMSD (nm)",
    "combined_RMSD_DNA.png",
    xlabel="Time (ns)"
)

save_combined_plot(
    [
        ("aptC", RAW / "aptC_RMSD_LIG_fitDNA.xvg"),
        ("apt3", RAW / "apt3_RMSD_LIG_fitDNA.xvg"),
        ("apt4", RAW / "apt4_RMSD_LIG_fitDNA.xvg"),
    ],
    "Ligand RMSD after DNA fitting",
    "RMSD (nm)",
    "combined_RMSD_LIG_fitDNA.png",
    xlabel="Time (ns)"
)

# single plots
for name in ["aptC", "apt3", "apt4"]:
    for metric, ylabel, xlabel in [
        ("mindist_DNA_LIG", "Minimum distance (nm)", "Time (ps)"),
        ("RMSD_DNA", "RMSD (nm)", "Time (ns)"),
        ("RMSD_LIG_fitDNA", "RMSD (nm)", "Time (ns)"),
    ]:
        path = RAW / f"{name}_{metric}.xvg"
        x, y = read_xvg(path)
        plt.figure(figsize=(7,4.5))
        plt.plot(x, y, linewidth=1.2)
        if metric == "mindist_DNA_LIG":
            plt.axhline(0.35, linestyle="--", linewidth=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{name} {metric}")
        plt.tight_layout()
        plt.savefig(PLOTS / f"{name}_{metric}.png", dpi=300)
        plt.close()

# MMGBSA summary and plot
mm = {}
with open(TABLES / "mmgbsa_dtotal_by_k.tsv", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        name = row["name"]
        val = float(row["DG_TOTAL_kcal_per_mol"])
        mm.setdefault(name, []).append(val)

with open(TABLES / "mmgbsa_summary.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["name", "mean_DG_TOTAL", "sd", "n"])
    for name, vals in mm.items():
        mean = stats.mean(vals)
        sd = stats.stdev(vals) if len(vals) > 1 else 0.0
        w.writerow([name, f"{mean:.4f}", f"{sd:.4f}", len(vals)])

names = list(mm.keys())
means = [stats.mean(mm[n]) for n in names]
sds = [stats.stdev(mm[n]) if len(mm[n]) > 1 else 0.0 for n in names]

plt.figure(figsize=(7,4.5))
plt.bar(names, means, yerr=sds, capsize=5)
plt.axhline(0, linewidth=1)
plt.ylabel("MM/GBSA ΔG_TOTAL (kcal/mol)")
plt.title("MM/GBSA binding free energy under ion-mixed condition")
plt.tight_layout()
plt.savefig(PLOTS / "combined_MMGBSA_DG_TOTAL_bar.png", dpi=300)
plt.close()

# make a brief README
readme = OUT / "README_ionmix_final_analysis.txt"
readme.write_text(
"""Ion-mixed final analysis archive

Contents:
- tables/trajectory_check.tsv
- tables/mindist_summary.tsv
- tables/mmgbsa_dtotal_by_k.tsv
- tables/mmgbsa_summary.tsv
- raw_xvg/*.xvg
- raw_mmgbsa/*/k*/FINAL_RESULTS_k*.dat
- pdb/*best_contact*.pdb
- pdb/*lastframe*.pdb
- plots/*.png

Final systems:
- aptC ionmix 30 ns
- apt3 bound-start ionmix 50 ns
- apt4 bound-start ionmix 50 ns

Notes:
- apt3 and apt4 final validation used representative bound-start conformations.
- mindist contact threshold: 0.35 nm.
- MM/GBSA used 50 ps/frame DNA_LIG trajectories and k1-k6 startframe robustness checks.
""",
encoding="utf-8"
)

print("Plots and summaries generated in:", OUT)
PY

echo "ALL FINAL ANALYSIS DONE"
echo "Output saved to:"
echo "$OUT"
