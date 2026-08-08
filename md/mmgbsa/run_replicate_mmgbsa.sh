#!/usr/bin/env bash

echo "Starting replicate MMGBSA..."
export GMX="${GMX:-$(command -v gmx || true)}"
BASE=~/aptamer_hcy_project

echo "aptamer,replicate,deltaG" > replicate_dg.csv

for APT in apt1-new apt3-new apt4-new
do
  for REP in rep1 rep2
  do

    DIR=$BASE/$APT/gmx/validation_10ns/$REP

    echo "Running $APT $REP"

    cd $DIR

    TPR=${APT}_${REP}_10ns.tpr
    XTC=traj_final.xtc

    gmx_MMPBSA -O \
    -i ../../../../mmpbsa.in \
    -cs $TPR \
    -ct $XTC \
    -ci index_clean.ndx \
    -cg 0 1 \
    -o FINAL_RESULTS.dat \
    -eo energy.csv
    DG=$(grep "DELTA TOTAL" FINAL_RESULTS.dat | awk '{print $3}')

    echo "$APT,$REP,$DG" >> $BASE/replicate_dg.csv

  done
done

echo "MMGBSA finished."

cd $BASE

python3 << 'PY'
import pandas as pd

df=pd.read_csv("replicate_dg.csv")

summary=df.groupby("aptamer")["deltaG"].agg(["mean","std"])

summary.to_csv("replicate_dg_summary.csv")

print("\nΔG summary:")
print(summary)
PY

echo ""
echo "Results saved:"
echo "replicate_dg.csv"
echo "replicate_dg_summary.csv"

