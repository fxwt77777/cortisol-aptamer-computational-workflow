#!/usr/bin/env bash
set -euo pipefail

MMSEQS="${MMSEQS:-$(command -v mmseqs || true)}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IN="${ROOT}/similarity_split/input/top5_extremes.fasta"
BASE="${ROOT}/similarity_split/mmseqs_linclust"
THREADS=8

if [[ ! -x "$MMSEQS" ]]; then
    echo "[ERROR] 找不到 MMseqs2: $MMSEQS"
    exit 1
fi

if [[ ! -s "$IN" ]]; then
    echo "[ERROR] 输入 FASTA 不存在或为空: $IN"
    exit 1
fi

run_one() {
    local label="$1"
    local identity="$2"

    local outdir="${BASE}/identity${label}"
    local prefix="${outdir}/top5_id${label}"
    local tmpdir="${outdir}/tmp"
    local tsv="${prefix}_cluster.tsv"

    rm -rf "$outdir"
    mkdir -p "$outdir" "$tmpdir"

    echo "[INFO] 开始 ${label}% identity Linclust"
    echo "[INFO] identity=${identity}, coverage=0.80, threads=${THREADS}"

    "$MMSEQS" easy-linclust \
        "$IN" \
        "$prefix" \
        "$tmpdir" \
        --min-seq-id "$identity" \
        -c 0.80 \
        --cov-mode 0 \
        --alignment-mode 3 \
        --kmer-per-seq 40 \
        --mask 0 \
        --threads "$THREADS" \
        > "${outdir}/run.log" 2>&1

    if [[ ! -s "$tsv" ]]; then
        echo "[ERROR] 未生成结果: $tsv"
        tail -n 80 "${outdir}/run.log" || true
        exit 1
    fi

    sequences=$(wc -l < "$tsv")

    clusters=$(
        cut -f1 "$tsv" |
        LC_ALL=C sort -u |
        wc -l
    )

    singleton_clusters=$(
        cut -f1 "$tsv" |
        LC_ALL=C sort |
        uniq -c |
        awk '$1 == 1 {n++} END {print n+0}'
    )

    multi_clusters=$(
        cut -f1 "$tsv" |
        LC_ALL=C sort |
        uniq -c |
        awk '$1 > 1 {n++} END {print n+0}'
    )

    largest_cluster=$(
        cut -f1 "$tsv" |
        LC_ALL=C sort |
        uniq -c |
        sort -nr |
        head -n 1 |
        awk '{print $1}'
    )

    {
        echo "threshold=${label}%"
        echo "minimum_identity=${identity}"
        echo "coverage=0.80"
        echo "coverage_mode=0"
        echo "input_sequences=${sequences}"
        echo "clusters=${clusters}"
        echo "singleton_clusters=${singleton_clusters}"
        echo "multi_member_clusters=${multi_clusters}"
        echo "largest_cluster_size=${largest_cluster}"
    } > "${outdir}/summary.txt"

    echo "[OK] ${label}% 完成"
    echo "[OK] clusters=${clusters}"
    echo "[OK] multi_member_clusters=${multi_clusters}"
    echo "[OK] largest_cluster_size=${largest_cluster}"
}

run_one 75 0.75
run_one 70 0.70

{
    echo "Sequence-similarity clustering summary"
    echo "======================================"
    echo
    echo "CD-HIT results"
    echo "85% identity: 11438 clusters"
    echo "80% identity: 11438 clusters"
    echo
    echo "MMseqs2 Linclust 75%"
    cat "${BASE}/identity75/summary.txt"
    echo
    echo "MMseqs2 Linclust 70%"
    cat "${BASE}/identity70/summary.txt"
} | tee "${BASE}/similarity_summary.txt"

echo "[OK] Linclust 批处理全部完成"
