#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RNN / LSTM model definitions —— for Shakespeare dataset"""

import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_dim=256, num_layers=2):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out
