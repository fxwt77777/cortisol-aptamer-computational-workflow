# src/evolve_cls.py
import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset_bin import encode_sequence   # 复用 encode_sequence
from model_cls import CNN_BiLSTM_Classifier
from config import NUC_VOCAB, SEED


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_trained_classifier(model_path="models/best_model_cls.pt", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")
    print(f"[DEBUG] Loading model from: {model_path}")
    model = CNN_BiLSTM_Classifier().to(device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print("[DEBUG] Model loaded.")
    return model, device


def predict_strong_prob(model, device, seq_list: List[str]):
    xs = np.stack([encode_sequence(s) for s in seq_list], axis=0)
    with torch.no_grad():
        x_tensor = torch.from_numpy(xs).to(device)
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=1)[:, 1]
    return probs.cpu().numpy()


def gc_content(seq: str) -> float:
    seq = seq.upper()
    if len(seq) == 0:
        return 0.0
    gc = sum(1 for ch in seq if ch in ["G", "C"])
    return gc / len(seq)


def max_homopolymer_run(seq: str) -> int:
    if not seq:
        return 0
    max_run = 1
    curr_run = 1
    prev = seq[0]
    for ch in seq[1:]:
        if ch == prev:
            curr_run += 1
            max_run = max(max_run, curr_run)
        else:
            prev = ch
            curr_run = 1
    return max_run


def constraint_penalty(seq: str,
                       gc_min=0.3,
                       gc_max=0.7,
                       max_run_allowed=6) -> float:
    seq = seq.upper()

    gc = gc_content(seq)
    penalty_gc = 0.0
    if gc < gc_min:
        penalty_gc = (gc_min - gc) * 10.0
    elif gc > gc_max:
        penalty_gc = (gc - gc_max) * 10.0

    run = max_homopolymer_run(seq)
    penalty_run = 0.0
    if run > max_run_allowed:
        penalty_run = (run - max_run_allowed) * 1.0

    return penalty_gc + penalty_run


def random_mutate(seq: str, n_mut: int = 1) -> str:
    seq = list(str(seq).upper())
    L = len(seq)
    n_mut = min(max(n_mut, 1), L)
    idxs = np.random.choice(L, size=n_mut, replace=False)
    for i in idxs:
        old = seq[i]
        choices = [n for n in NUC_VOCAB if n != old]
        seq[i] = np.random.choice(choices)
    return "".join(seq)


def evolve(
    init_seqs: List[str],
    model,
    device,
    n_generations=20,
    offspring_per_seq=5,
    top_k=20,
    max_mut_per_gen=3,
    gc_min=0.35,
    gc_max=0.65,
    max_run_allowed=5,
    penalty_weight=1.0,
):
    print(f"[DEBUG] Start evolve with {len(init_seqs)} seeds.")
    current_pool = init_seqs
    history = []

    for gen in range(1, n_generations + 1):
        candidates = []

        for s in current_pool:
            candidates.append(s)
            for _ in range(offspring_per_seq):
                n_mut = np.random.randint(1, max_mut_per_gen + 1)
                child = random_mutate(s, n_mut=n_mut)
                candidates.append(child)

        candidates = list(dict.fromkeys(candidates))
        print(f"[DEBUG] Gen {gen}: candidates={len(candidates)}")

        probs = predict_strong_prob(model, device, candidates)

        scores = []
        for seq, p in zip(candidates, probs):
            pen = constraint_penalty(
                seq,
                gc_min=gc_min,
                gc_max=gc_max,
                max_run_allowed=max_run_allowed,
            )
            score = p - penalty_weight * pen
            scores.append(score)
        scores = np.array(scores)

        idx_sorted = np.argsort(-scores)
        candidates_sorted = [candidates[i] for i in idx_sorted]
        scores_sorted = scores[idx_sorted]
        probs_sorted = probs[idx_sorted]

        best_seq = candidates_sorted[0]
        best_score = scores_sorted[0]
        best_prob = probs_sorted[0]
        best_gc = gc_content(best_seq)
        best_run = max_homopolymer_run(best_seq)

        print(
            f"Gen {gen}: score={best_score:.4f}, prob_cls={best_prob:.4f}, "
            f"GC={best_gc:.2f}, max_run={best_run}, seq={best_seq[:25]}..."
        )

        history.append((gen, best_seq, best_score, best_prob, best_gc, best_run))

        current_pool = candidates_sorted[:top_k]

    return history, current_pool, scores_sorted[:top_k], probs_sorted[:top_k]


def main():
    print("[DEBUG] evolve_cls.py main() start")
    set_seed()

    # 1) 加载 best_model_cls.pt
    model, device = load_trained_classifier(
        model_path="models/best_model_cls.pt"
    )

    # 2) 从 train_topbin.csv 选 strong 种子（top20%）
    print("[DEBUG] Loading data/train_topbin.csv")
    train_top = pd.read_csv("data/train_topbin.csv")
    strong_df = train_top[train_top["class_topbin"] == 1].copy()
    print(f"[DEBUG] strong samples (top20%): {len(strong_df)}")

    if "binding_affinity_likelihood" in strong_df.columns:
        strong_df = strong_df.sort_values(
            "binding_affinity_likelihood", ascending=False
        )

    N_SEEDS = 30
    init_seqs = strong_df["dna_sequence"].head(N_SEEDS).tolist()
    print(f"[DEBUG] Using {len(init_seqs)} seeds.")

    # 3) 进化
    history, final_pool, final_scores, final_probs = evolve(
        init_seqs,
        model,
        device,
        n_generations=20,
        offspring_per_seq=5,
        top_k=20,
        max_mut_per_gen=3,
        gc_min=0.35,
        gc_max=0.65,
        max_run_allowed=5,
        penalty_weight=1.0,
    )

    # 4) 保存结果
    out_path = "data/evolved_candidates_cls_constrained.csv"
    out_df = pd.DataFrame({
        "sequence": final_pool,
        "score": final_scores,
        "pred_strong_prob_cls": final_probs,
        "GC_content": [gc_content(s) for s in final_pool],
        "max_run": [max_homopolymer_run(s) for s in final_pool],
    })
    out_df.to_csv(out_path, index=False)
    print(f"[DEBUG] Saved evolved candidates to {out_path}")


if __name__ == "__main__":
    main()
