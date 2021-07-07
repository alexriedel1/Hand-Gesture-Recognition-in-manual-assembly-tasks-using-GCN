import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.adj_mat
from config import CFG


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, batch_size, num_layers ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = batch_size

        self.hidden2label = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(int(hidden_dim/2), output_dim),
        )
    
    def forward(self, x):
        x = x.reshape(self.batch_size, self.seq_len, CFG.num_feats*21)
        x, (h, c) = self.lstm(x)
        x = x[:, -1, :]
        x = self.hidden2label(x)
        return x