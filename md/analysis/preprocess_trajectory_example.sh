#!/usr/bin/env bash

ROOT="${APTAMER_PROJECT_ROOT:-$(pwd)}"
ANALYSIS=$ROOT/analysis_3x50ns_20260807
SRC=$ROOT/md_replicates_3x50ns_20260806/aptC/rep2
GMX="${GMX:-$(command -v gmx || true)}"
NAME=aptC_rep2

echo "===== 1. RAW LINKS ====="

ln -sfn "$SRC/md_50ns.tpr" "$ANALYSIS/raw_links/${NAME}.tpr"
ln -sfn "$SRC/md_50ns.xtc" "$ANALYSIS/raw_links/${NAME}.xtc"

echo
echo "===== 2. FULL TRAJECTORY ====="

"$GMX" check -f "$ANALYSIS/raw_links/${NAME}.xtc" 2>&1 |
grep -E '# Atoms|Last frame|Step|Time|Coords'

echo
echo "===== 3. FULL INDEX ====="

NDX="$ANALYSIS/index/${NAME}.ndx"

printf "1 | 2\nname 13 DNA_LIG\nq\n" |
"$GMX" make_ndx \
  -f "$ANALYSIS/raw_links/${NAME}.tpr" \
  -o "$NDX" \
  >/tmp/${NAME}_make_ndx.log 2>&1

echo "DNA / LIG / DNA_LIG:"
awk '
/^\[/ {
    if(group!="") printf "%-15s %8d\n",group,count
    group=$0
    gsub(/^\[[[:space:]]*|[[:space:]]*\]$/,"",group)
    count=0
    next
}
{for(i=1;i<=NF;i++) count++}
END {if(group!="") printf "%-15s %8d\n",group,count}
' "$NDX" |
grep -E '^(DNA|LIG|DNA_LIG)[[:space:]]'

echo
echo "===== 4. CENTER DNA + KEEP DNA_LIG ====="

OUT="$ANALYSIS/processed/${NAME}_DNA_LIG_centered.xtc"
REF="$ANALYSIS/processed/${NAME}_DNA_LIG_ref.gro"

printf "DNA\nDNA_LIG\n" |
"$GMX" trjconv \
  -s "$ANALYSIS/raw_links/${NAME}.tpr" \
  -f "$ANALYSIS/raw_links/${NAME}.xtc" \
  -n "$NDX" \
  -o "$OUT" \
  -b 0 -e 50000 \
  -pbc mol \
  -center \
  -ur compact

printf "DNA\nDNA_LIG\n" |
"$GMX" trjconv \
  -s "$ANALYSIS/raw_links/${NAME}.tpr" \
  -f "$ANALYSIS/raw_links/${NAME}.xtc" \
  -n "$NDX" \
  -o "$REF" \
  -dump 0 \
  -pbc mol \
  -center \
  -ur compact

echo
echo "===== 5. REDUCED TPR ====="

REDTPR="$ANALYSIS/processed/${NAME}_DNA_LIG.tpr"

printf "DNA_LIG\n" |
"$GMX" convert-tpr \
  -s "$ANALYSIS/raw_links/${NAME}.tpr" \
  -n "$NDX" \
  -o "$REDTPR"

echo
echo "===== 6. ANALYSIS INDEX ====="

ANDX="$ANALYSIS/analysis_index/${NAME}.ndx"

printf "1 | 2\nname 3 DNA_LIG\nq\n" |
"$GMX" make_ndx \
  -f "$REDTPR" \
  -o "$ANDX" \
  >/tmp/${NAME}_analysis_ndx.log 2>&1

echo
echo "===== 7. FINAL CHECK ====="

"$GMX" check -f "$OUT" 2>&1 |
grep -E '# Atoms|Last frame|Step|Time|Coords'

echo
"$GMX" dump -s "$REDTPR" 2>/dev/null |
grep -m1 'natoms'

echo
head -n 2 "$REF"

echo
echo "===== ANALYSIS GROUPS ====="

awk '
/^\[/ {
    if(group!="") printf "%-15s %8d\n",group,count
    group=$0
    gsub(/^\[[[:space:]]*|[[:space:]]*\]$/,"",group)
    count=0
    next
}
{for(i=1;i<=NF;i++) count++}
END {if(group!="") printf "%-15s %8d\n",group,count}
' "$ANDX" |
grep -E '^(System|DNA|LIG|DNA_LIG)[[:space:]]'

echo
echo "===== FILE SIZES ====="
ls -lh \
  "$OUT" \
  "$REF" \
  "$REDTPR"

