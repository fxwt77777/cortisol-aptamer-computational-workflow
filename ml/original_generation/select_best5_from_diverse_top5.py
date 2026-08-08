# src/select_best5_from_diverse_top5.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
IN_PATH  = DATA_DIR / "evolved_candidates_top5_cls_diverse10.csv"
OUT_PATH = DATA_DIR / "evolved_candidates_top5_cls_best5.csv"

def main():
    print(f"[INFO] 读取候选文件: {IN_PATH}")
    df = pd.read_csv(IN_PATH)
    print(f"[INFO] 总共有 {len(df)} 条多样性筛选后的候选。")

    # 先按 score 降序，再按 pred_strong_prob_cls 降序排序
    df = df.sort_values(
        by=["score", "pred_strong_prob_cls"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # 第一轮：严格筛选
    cond_strict = (
        (df["GC_content"] >= 0.35) &
        (df["GC_content"] <= 0.60) &
        (df["max_run"] <= 4)
    )
    df_strict = df[cond_strict].copy()
    print(f"[INFO] 严格条件下（0.35<=GC<=0.60, max_run<=4）候选数: {len(df_strict)}")

    selected = []

    if len(df_strict) >= 5:
        selected = df_strict.head(5)
        print("[INFO] 严格条件已经足够，直接取前 5 条。")
    else:
        # 严格条件不够 5 条，就先把严格的都拿上
        selected = df_strict.copy()
        need_more = 5 - len(selected)
        print(f"[INFO] 严格条件不足 5 条，还需要补充 {need_more} 条。")

        # 第二轮：放宽条件 max_run<=5, GC 0.35~0.65
        cond_relaxed = (
            (df["GC_content"] >= 0.35) &
            (df["GC_content"] <= 0.65) &
            (df["max_run"] <= 5)
        )
        df_relaxed = df[cond_relaxed].copy()

        # 把已经选过的去掉
        if not selected.empty:
            already = set(selected["sequence"].tolist())
            df_relaxed = df_relaxed[~df_relaxed["sequence"].isin(already)]

        # 再按排序顺序补足
        df_relaxed = df_relaxed.sort_values(
            by=["score", "pred_strong_prob_cls"],
            ascending=[False, False]
        )
        extra = df_relaxed.head(need_more)
        print(f"[INFO] 放宽条件下可补充 {len(extra)} 条。")

        selected = pd.concat([selected, extra], axis=0).reset_index(drop=True)

    print(f"[INFO] 最终选出 {len(selected)} 条作为 best5（目标是 5 条）。")

    selected.to_csv(OUT_PATH, index=False)
    print(f"[INFO] 已保存最终 5 条到: {OUT_PATH}")

    # 顺便打印一下结果概要
    print("\n[RESULT] 最终 5 条候选：")
    for i, row in selected.iterrows():
        print(f"#{i+1}: score={row['score']:.4f}, prob={row['pred_strong_prob_cls']:.4f}, "
              f"GC={row['GC_content']:.2f}, max_run={int(row['max_run'])}")
        print(f"     {row['sequence']}")
        print()

if __name__ == "__main__":
    main()
