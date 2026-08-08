# src/preprocess_topbottom_binary.py
import pandas as pd

Q_LOW = 0.2
Q_HIGH = 0.8

def make_topbottom_split(split_name: str):
    path = f"data/{split_name}.csv"
    df = pd.read_csv(path)
    print(f"{split_name}: 原始样本数 {len(df)}")

    if "y_log" not in df.columns:
        raise ValueError(f"{path} 中找不到 y_log 列，请确认预处理脚本已生成 y_log")

    q_low = df["y_log"].quantile(Q_LOW)
    q_high = df["y_log"].quantile(Q_HIGH)
    print(f"{split_name}: y_log q_low({Q_LOW})={q_low:.4f}, q_high({Q_HIGH})={q_high:.4f}")

    # 只保留 bottom20% 和 top20%
    low_df = df[df["y_log"] <= q_low].copy()
    high_df = df[df["y_log"] >= q_high].copy()

    low_df["class_topbin"] = 0
    high_df["class_topbin"] = 1

    out_df = pd.concat([low_df, high_df], ignore_index=True)
    print(f"{split_name}: 选出 bottom20% + top20% 后样本数 {len(out_df)}")

    print("class_topbin 计数：")
    print(out_df["class_topbin"].value_counts().sort_index())
    print("-" * 40)

    out_path = f"data/{split_name}_topbin.csv"
    out_df.to_csv(out_path, index=False)
    print(f"{split_name}: 已保存 {out_path}")

def main():
    for split in ["train", "val", "test"]:
        make_topbottom_split(split)

if __name__ == "__main__":
    main()
