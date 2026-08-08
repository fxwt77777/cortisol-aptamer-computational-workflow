# src/train_cls.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

from config import BATCH_SIZE, LR, WEIGHT_DECAY, N_EPOCHS, EARLY_STOP_PATIENCE, SEED
from dataset_bin import get_dataloaders_bin
from model_cls import CNN_BiLSTM_Classifier


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def eval_epoch(model, loader, device):
    model.eval()
    ys_cls = []
    preds_cls = []
    probs_pos = []

    with torch.no_grad():
        for x, y_cls in loader:
            x = x.to(device)
            y_cls = y_cls.to(device)

            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]  # 正类概率

            preds = logits.argmax(dim=1)

            ys_cls.append(y_cls.cpu().numpy())
            preds_cls.append(preds.cpu().numpy())
            probs_pos.append(prob.cpu().numpy())

    ys_cls = np.concatenate(ys_cls)
    preds_cls = np.concatenate(preds_cls)
    probs_pos = np.concatenate(probs_pos)

    acc = accuracy_score(ys_cls, preds_cls)

    # 可以看一下“标签 vs 概率”的 Pearson（只是参考，不是主要指标）
    try:
        pear = pearsonr(ys_cls, probs_pos)[0]
    except Exception:
        pear = 0.0

    return acc, pear


def main():
    set_seed()
    os.makedirs("models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_dataloaders_bin(BATCH_SIZE)
    model = CNN_BiLSTM_Classifier().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        train_losses = []

        for x, y_cls in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = x.to(device)
            y_cls = y_cls.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y_cls)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_acc, val_pear = eval_epoch(model, val_loader, device)

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.3f} pear(y,prob)={val_pear:.3f}"
        )

        # Early stopping based on val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    torch.save(model.state_dict(), "models/best_model_cls.pt")
    print("Best classifier saved to models/best_model_cls.pt")

    # 测试集评估
    test_acc, test_pear = eval_epoch(model, test_loader, device)
    print(
        f"Test (binary strong vs weak): Acc={test_acc:.3f}, "
        f"Pearson(y,prob)={test_pear:.3f}"
    )


if __name__ == "__main__":
    main()
