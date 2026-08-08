#!/usr/bin/env bash
set -euo pipefail

MMSEQS="${MMSEQS:-$(command -v mmseqs || true)}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IN="${ROOT}/similarity_split/input/top5_extremes.fasta"
BASE="${ROOT}/similarity_split/mmseqs_linclust"
THREADS=8

summarize_tsv() {
    local label="$1"
    local identity="$2"
    local tsv="$3"
    local summary="$4"

    if [[ ! -s "$tsv" ]]; then
        echo "[ERROR] 聚类结果不存在: $tsv"
        exit 1
    fi

    awk -F'\t' \
        -v label="$label" \
        -v identity="$identity" '
    {
        sequences++
        count[$1]++
    }
    END {
        clusters=0
        singleton=0
        multi=0
        largest=0

        for (cluster in count) {
            clusters++

            if (count[cluster] == 1)
                singleton++
            else
                multi++

            if (count[cluster] > largest)
                largest=count[cluster]
        }

        print "threshold=" label "%"
        print "minimum_identity=" identity
        print "coverage=0.80"
        print "coverage_mode=0"
        print "input_sequences=" sequences
        print "clusters=" clusters
        print "singleton_clusters=" singleton
        print "multi_member_clusters=" multi
        print "largest_cluster_size=" largest
    }
    ' "$tsv" > "$summary"
}

# 75%结果已完成，只重新统计
TSV75="${BASE}/identity75/top5_id75_cluster.tsv"
SUMMARY75="${BASE}/identity75/summary.txt"

echo "[INFO] 汇总已经完成的75%结果"
summarize_tsv 75 0.75 "$TSV75" "$SUMMARY75"
cat "$SUMMARY75"

# 运行70%
OUT70="${BASE}/identity70"
PREFIX70="${OUT70}/top5_id70"
TMP70="${OUT70}/tmp"
TSV70="${PREFIX70}_cluster.tsv"

rm -rf "$OUT70"
mkdir -p "$OUT70" "$TMP70"

echo
echo "[INFO] 开始70% identity Linclust"

"$MMSEQS" easy-linclust \
    "$IN" \
    "$PREFIX70" \
    "$TMP70" \
    --min-seq-id 0.70 \
    -c 0.80 \
    --cov-mode 0 \
    --alignment-mode 3 \
    --kmer-per-seq 40 \
    --mask 0 \
    --threads "$THREADS" \
    > "${OUT70}/run.log" 2>&1

echo "[OK] 70%聚类计算完成"

summarize_tsv \
    70 \
    0.70 \
    "$TSV70" \
    "${OUT70}/summary.txt"

cat "${OUT70}/summary.txt"

{
    echo "Sequence-similarity clustering summary"
    echo "======================================"
    echo
    echo "CD-HIT"
    echo "85% identity: 11438 clusters"
    echo "80% identity: 11438 clusters"
    echo
    echo "MMseqs2 Linclust 75%"
    cat "$SUMMARY75"
    echo
    echo "MMseqs2 Linclust 70%"
    cat "${OUT70}/summary.txt"
} > "${BASE}/similarity_summary.txt"

echo
echo "[OK] 最终汇总已生成："
echo "${BASE}/similarity_summary.txt"
