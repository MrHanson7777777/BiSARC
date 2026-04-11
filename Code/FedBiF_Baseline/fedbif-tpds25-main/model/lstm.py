import torch
import torch.nn as nn

class CharLSTM(nn.Module):

    def __init__(self, num_chars: int = 80, embed_dim: int = 8,
                 hidden_dim: int = 256, num_layers: int = 2):
        super(CharLSTM, self).__init__()
        self.num_chars = num_chars
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(num_chars, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_chars)

    def forward(self, x):

        emb = self.embed(x)
        out, _ = self.lstm(emb)

        logits = self.classifier(out[:, -1, :]) 
        return logits