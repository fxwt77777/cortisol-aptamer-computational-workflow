#!/usr/bin/env bash
set -euo pipefail

MMSEQS="${MMSEQS:-$(command -v mmseqs || true)}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IN="${ROOT}/similarity_split/input/top5_extremes.fasta"
BASE="${ROOT}/similarity_split/mmseqs"
THREADS="${THREADS:-32}"

if [[ ! -x "$MMSEQS" ]]; then
    echo "[ERROR] 找不到MMseqs2: $MMSEQS"
    exit 1
fi

if [[ ! -s "$IN" ]]; then
    echo "[ERROR] 输入FASTA不存在或为空: $IN"
    exit 1
fi

mkdir -p "$BASE"

run_cluster() {
    local name="$1"
    local identity="$2"

    local outdir="${BASE}/identity${name}"
    local prefix="${outdir}/top5_id${name}"
    local tmpdir="${outdir}/tmp"

    mkdir -p "$outdir"
    rm -rf "$tmpdir"
    mkdir -p "$tmpdir"

    echo "[INFO] 开始${name}% identity聚类"
    echo "[INFO] identity=${identity}, bidirectional coverage=0.80"

    "$MMSEQS" easy-cluster \
        "$IN" \
        "$prefix" \
        "$tmpdir" \
        --min-seq-id "$identity" \
        -c 0.80 \
        --cov-mode 0 \
        --threads "$THREADS" \
        > "${outdir}/run.log" 2>&1

    local tsv="${prefix}_cluster.tsv"

    if [[ ! -s "$tsv" ]]; then
        echo "[ERROR] 未生成cluster TSV: $tsv"
        tail -n 50 "${outdir}/run.log" || true
        exit 1
    fi

    local sequences
    local clusters
    local singleton_clusters
    local multi_clusters
    local largest_cluster

    sequences=$(wc -l < "$tsv")
    clusters=$(cut -f1 "$tsv" | sort -u | wc -l)

    singleton_clusters=$(
        cut -f1 "$tsv" |
        sort |
        uniq -c |
        awk '$1 == 1 {n++} END {print n+0}'
    )

    multi_clusters=$(
        cut -f1 "$tsv" |
        sort |
        uniq -c |
        awk '$1 > 1 {n++} END {print n+0}'
    )

    largest_cluster=$(
        cut -f1 "$tsv" |
        sort |
        uniq -c |
        sort -nr |
        head -n 1 |
        awk '{print $1}'
    )

    {
        echo "threshold_name=${name}"
        echo "minimum_identity=${identity}"
        echo "coverage=0.80"
        echo "coverage_mode=bidirectional"
        echo "sequences=${sequences}"
        echo "clusters=${clusters}"
        echo "singleton_clusters=${singleton_clusters}"
        echo "multi_member_clusters=${multi_clusters}"
        echo "largest_cluster_size=${largest_cluster}"
    } > "${outdir}/summary.txt"

    echo "[OK] ${name}%完成：${clusters} clusters"
}

run_cluster 75 0.75
run_cluster 70 0.70

{
    echo "Sequence-similarity clustering summary"
    echo "======================================"
    echo
    echo "Previous CD-HIT results"
    echo "85% identity: 11438 clusters"
    echo "80% identity: 11438 clusters"
    echo
    echo "MMseqs2 75% result"
    cat "${BASE}/identity75/summary.txt"
    echo
    echo "MMseqs2 70% result"
    cat "${BASE}/identity70/summary.txt"
} | tee "${BASE}/similarity_summary.txt"

echo "[OK] 全部相似性聚类完成"
