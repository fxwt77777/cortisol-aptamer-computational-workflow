#!/usr/bin/env bash
set -euo pipefail

CDHIT="${CDHIT:-$(command -v cd-hit-est || true)}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IN="${ROOT}/similarity_split/input/top5_extremes.fasta"
BASE="${ROOT}/similarity_split/cdhit"

if [[ ! -x "$CDHIT" ]]; then
    echo "[ERROR] cd-hit-est不存在: $CDHIT"
    exit 1
fi

if [[ ! -f "$IN" ]]; then
    echo "[ERROR] 输入FASTA不存在: $IN"
    exit 1
fi

mkdir -p "${BASE}/identity75" "${BASE}/identity70"

echo "[INFO] 开始75% identity聚类"

"$CDHIT" \
    -i "$IN" \
    -o "${BASE}/identity75/top5_id75" \
    -c 0.75 \
    -n 4 \
    -G 1 \
    -aS 0.8 \
    -g 1 \
    -d 0 \
    -T 0 \
    -M 8000 \
    > "${BASE}/identity75/run.log" 2>&1

echo "[INFO] 开始70% identity聚类"

"$CDHIT" \
    -i "$IN" \
    -o "${BASE}/identity70/top5_id70" \
    -c 0.70 \
    -n 3 \
    -G 1 \
    -aS 0.8 \
    -g 1 \
    -d 0 \
    -T 0 \
    -M 8000 \
    > "${BASE}/identity70/run.log" 2>&1

N75=$(grep -c '^>Cluster' "${BASE}/identity75/top5_id75.clstr")
N70=$(grep -c '^>Cluster' "${BASE}/identity70/top5_id70.clstr")

{
    echo "Similarity clustering summary"
    echo "============================="
    echo "Input sequences: 11438"
    echo "75% identity clusters: ${N75}"
    echo "70% identity clusters: ${N70}"
} | tee "${BASE}/lower_identity_summary.txt"

echo "[OK] 批量聚类完成"
