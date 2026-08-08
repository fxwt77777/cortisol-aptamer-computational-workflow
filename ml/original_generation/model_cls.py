# src/model_cls.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUC_VOCAB, CONV_FILTERS, KERNEL_SIZES, HIDDEN_DIM, DROPOUT

class CNN_BiLSTM_Classifier(nn.Module):
    """
    CNN + BiLSTM 二分类：
    输出是 logits，使用 CrossEntropyLoss（标签 0/1）
    """
    def __init__(self):
        super().__init__()
        input_dim = len(NUC_VOCAB)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=CONV_FILTERS,
                kernel_size=k
            )
            for k in KERNEL_SIZES
        ])

        self.lstm = nn.LSTM(
            input_size=CONV_FILTERS * len(KERNEL_SIZES),
            hidden_size=HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM * 2, 2)  # 二分类 -> 2 维 logits

    def forward(self, x):
        # x: [batch, L, 4]
        x = x.transpose(1, 2)  # [batch, 4, L]

        conv_features = []
        lengths = []

        for conv in self.convs:
            c = F.relu(conv(x))    # [batch, F, L']
            c = c.transpose(1, 2)  # [batch, L', F]
            conv_features.append(c)
            lengths.append(c.size(1))

        min_len = min(lengths)
        conv_features = [cf[:, :min_len, :] for cf in conv_features]

        feats = torch.cat(conv_features, dim=2)   # [batch, L', F * num_kernels]

        lstm_out, _ = self.lstm(feats)            # [batch, L', 2*H]

        h = lstm_out.mean(dim=1)                  # [batch, 2*H]
        h = self.dropout(h)

        logits = self.fc(h)                       # [batch, 2]
        return logits
