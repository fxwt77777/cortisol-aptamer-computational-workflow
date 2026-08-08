#!/usr/bin/env bash

ROOT="${APTAMER_PROJECT_ROOT:-$(pwd)}"
ANALYSIS=$ROOT/analysis_3x50ns_20260807
export ANALYSIS
GMX="${GMX:-$(command -v gmx || true)}"

mkdir -p \
  "$ANALYSIS/results/rmsd" \
  "$ANALYSIS/results/rg" \
  "$ANALYSIS/results/mindist" \
  "$ANALYSIS/results/contacts" \
  "$ANALYSIS/results/hbond" \
  "$ANALYSIS/results/ligand_rmsd" \
  "$ANALYSIS/results/rmsf"

for NAME in APT3_rep2 APT4_rep2
do
    echo
    echo "============================================================"
    echo "$NAME"
    echo "============================================================"

    TPR="$ANALYSIS/processed/${NAME}_DNA_LIG.tpr"
    XTC="$ANALYSIS/processed/${NAME}_DNA_LIG_centered.xtc"
    NDX="$ANALYSIS/analysis_index/${NAME}.ndx"

    if [ ! -s "$TPR" ] || [ ! -s "$XTC" ] || [ ! -s "$NDX" ]; then
        echo "[ERROR] Required file missing for $NAME"
        exit 1
    fi


    echo
    echo "===== DNA RMSD ====="

    printf "DNA\nDNA\n" |
    "$GMX" rms \
      -s "$TPR" \
      -f "$XTC" \
      -n "$NDX" \
      -o "$ANALYSIS/results/rmsd/${NAME}_DNA_rmsd.xvg" \
      -tu ns \
      >/tmp/${NAME}_rmsd.log 2>&1

    if [ $? -ne 0 ]; then
        cat /tmp/${NAME}_rmsd.log
        exit 1
    fi


    echo "===== DNA Rg ====="

    printf "DNA\n" |
    "$GMX" gyrate \
      -s "$TPR" \
      -f "$XTC" \
      -n "$NDX" \
      -o "$ANALYSIS/results/rg/${NAME}_DNA_rg.xvg" \
      >/tmp/${NAME}_rg.log 2>&1

    if [ $? -ne 0 ]; then
        cat /tmp/${NAME}_rg.log
        exit 1
    fi


    echo "===== DNA-LIG MINDIST ====="

    printf "DNA\nLIG\n" |
    "$GMX" mindist \
      -s "$TPR" \
      -f "$XTC" \
      -n "$NDX" \
      -od "$ANALYSIS/results/mindist/${NAME}_DNA_LIG_mindist.xvg" \
      -d 0.35 \
      >/tmp/${NAME}_mindist.log 2>&1

    if [ $? -ne 0 ]; then
        cat /tmp/${NAME}_mindist.log
        exit 1
    fi


    echo "===== DNA-LIG CONTACT COUNT ====="

    printf "DNA\nLIG\n" |
    "$GMX" mindist \
      -s "$TPR" \
      -f "$XTC" \
      -n "$NDX" \
      -on "$ANALYSIS/results/contacts/${NAME}_DNA_LIG_contacts.xvg" \
      -d 0.35 \
      >/tmp/${NAME}_contacts.log 2>&1

    if [ $? -ne 0 ]; then
        cat /tmp/${NAME}_contacts.log
        exit 1
    fi


    echo "===== DNA-LIG H-BONDS ====="

    printf "DNA\nLIG\n" |
    "$GMX" hbond \
      -s "$TPR" \
      -f "$XTC" \
      -n "$NDX" \
      -num "$ANALYSIS/results/hbond/${NAME}_DNA_LIG_hbond.xvg" \
      >/tmp/${NAME}_hbond.log 2>&1

    if [ $? -ne 0 ]; then
        cat /tmp/${NAME}_hbond.log
        exit 1
    fi


    echo "===== LIG RMSD AFTER DNA FIT ====="

    printf "DNA\nLIG\n" |
    "$GMX" rms \
      -s "$TPR" \
      -f "$XTC" \
      -n "$NDX" \
      -o "$ANALYSIS/results/ligand_rmsd/${NAME}_LIG_rmsd_DNAfit.xvg" \
      -tu ns \
      >/tmp/${NAME}_ligrmsd.log 2>&1

    if [ $? -ne 0 ]; then
        cat /tmp/${NAME}_ligrmsd.log
        exit 1
    fi


    echo "===== DNA RESIDUE RMSF ====="

    printf "DNA\n" |
    "$GMX" rmsf \
      -s "$TPR" \
      -f "$XTC" \
      -n "$NDX" \
      -o "$ANALYSIS/results/rmsf/${NAME}_DNA_residue_rmsf.xvg" \
      -res \
      -fit \
      >/tmp/${NAME}_rmsf.log 2>&1

    if [ $? -ne 0 ]; then
        cat /tmp/${NAME}_rmsf.log
        exit 1
    fi

    echo "[OK] $NAME core analyses completed"

done

echo
echo "============================================================"
echo "REP2 CORE SUMMARY"
echo "============================================================"

python3 <<'PY'
from pathlib import Path
import os
import numpy as np

root = Path(os.environ["ANALYSIS"])

def read_xvg(path):
    a=[]
    with open(path) as f:
        for line in f:
            if line.startswith(("#","@")):
                continue
            p=line.split()
            if len(p) >= 2:
                a.append([float(p[0]), float(p[1])])
    return np.asarray(a)

print(
    "system\treplicate\t"
    "DNA_RMSD_mean_nm\tDNA_Rg_mean_nm\t"
    "mindist_mean_nm\tcontact_fraction\t"
    "contact_count_mean\tHbond_mean\tHbond_fraction\t"
    "LIG_RMSD_mean_nm\tDNA_RMSF_mean_nm"
)

for name in ["APT3_rep2","APT4_rep2"]:

    system, rep = name.split("_")

    rmsd = read_xvg(root/"results/rmsd"/f"{name}_DNA_rmsd.xvg")[:,1]
    rg = read_xvg(root/"results/rg"/f"{name}_DNA_rg.xvg")[:,1]

    md = read_xvg(
        root/"results/mindist"/f"{name}_DNA_LIG_mindist.xvg"
    )[:,1]

    contacts = read_xvg(
        root/"results/contacts"/f"{name}_DNA_LIG_contacts.xvg"
    )[:,1]

    hb = read_xvg(
        root/"results/hbond"/f"{name}_DNA_LIG_hbond.xvg"
    )[:,1]

    lig = read_xvg(
        root/"results/ligand_rmsd"/f"{name}_LIG_rmsd_DNAfit.xvg"
    )[:,1]

    rmsf = read_xvg(
        root/"results/rmsf"/f"{name}_DNA_residue_rmsf.xvg"
    )[:,1]

    print(
        f"{system}\t{rep}\t"
        f"{rmsd.mean():.6f}\t"
        f"{rg.mean():.6f}\t"
        f"{md.mean():.6f}\t"
        f"{np.mean(md <= 0.35):.6f}\t"
        f"{contacts.mean():.6f}\t"
        f"{hb.mean():.6f}\t"
        f"{np.mean(hb >= 1):.6f}\t"
        f"{lig.mean():.6f}\t"
        f"{rmsf.mean():.6f}"
    )

print()
print("===== FRAME / RESIDUE COUNTS =====")

for name in ["APT3_rep2","APT4_rep2"]:
    rmsd = read_xvg(root/"results/rmsd"/f"{name}_DNA_rmsd.xvg")
    rmsf = read_xvg(root/"results/rmsf"/f"{name}_DNA_residue_rmsf.xvg")

    print(
        f"{name}: trajectory_frames={len(rmsd)}, "
        f"DNA_residues={len(rmsf)}"
    )
PY

