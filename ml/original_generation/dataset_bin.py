# src/dataset_bin.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import L_MAX, NUC_VOCAB, BATCH_SIZE

NUC2IDX = {n: i for i, n in enumerate(NUC_VOCAB)}

def encode_sequence(seq: str, L: int = L_MAX):
    seq = str(seq).upper()
    arr = np.zeros((L, len(NUC_VOCAB)), dtype=np.float32)

    max_len = min(len(seq), L)
    for i in range(max_len):
        ch = seq[i]
        if ch in NUC2IDX:
            arr[i, NUC2IDX[ch]] = 1.0
    return arr

class AptamerBinDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = row["dna_sequence"]
        x = encode_sequence(seq)

        y_cls = int(row["class_bin"])  # 0 or 1

        x_tensor = torch.from_numpy(x)                      # [L,4]
        y_cls_tensor = torch.tensor(y_cls, dtype=torch.long)  # 标准CE用long

        return x_tensor, y_cls_tensor

def get_dataloaders_bin(batch_size: int = BATCH_SIZE):
    train_ds = AptamerBinDataset("data/train_bin.csv")
    val_ds   = AptamerBinDataset("data/val_bin.csv")
    test_ds  = AptamerBinDataset("data/test_bin.csv")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader
