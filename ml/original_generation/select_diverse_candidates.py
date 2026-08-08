# src/select_diverse_candidates.py
import pandas as pd

def seq_identity(a: str, b: str) -> float:
    a = a.strip().upper()
    b = b.strip().upper()
    L = min(len(a), len(b))
    if L == 0:
        return 0.0
    same = sum(1 for i in range(L) if a[i] == b[i])
    return same / L

def main():
    in_path = "data/evolved_candidates_cls_constrained.csv"
    df = pd.read_csv(in_path)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    selected = []
    selected_rows = []

    max_candidates = 10    # 目标：至少/最多 10 条
    max_identity = 0.9     # 相似度上限 90%

    for _, row in df.iterrows():
        seq = row["sequence"]
        if not selected:
            selected.append(seq)
            selected_rows.append(row)
        else:
            idents = [seq_identity(seq, s) for s in selected]
            if max(idents) < max_identity:
                selected.append(seq)
                selected_rows.append(row)

        if len(selected) >= max_candidates:
            break

    out_df = pd.DataFrame(selected_rows)
    out_path = "data/evolved_selected_top10_cls.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Selected {len(out_df)} diverse candidates -> {out_path}")

if __name__ == "__main__":
    main()
