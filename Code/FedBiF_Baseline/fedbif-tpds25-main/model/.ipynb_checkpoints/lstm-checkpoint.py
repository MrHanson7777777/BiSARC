import torch
import torch.nn as nn

class CharLSTM(nn.Module):
    """
    统一后的字符级 LSTM 语言模型。
    参数量与 Ours 保持绝对一致，但适配了 FedBiF 的单步预测 Loss。
    """
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
        # 统一使用带有 bias 的 Linear 层
        self.classifier = nn.Linear(hidden_dim, num_chars)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        
        # 【核心修复】：因为 LEAF 数据的 target 是单字符 (batch_size)
        # 我们只截取 LSTM 最后一个时间步的隐藏状态来做分类
        # 这样输出维度变为 (batch_size, num_chars)，完美契合 [32] 的标签
        logits = self.classifier(out[:, -1, :]) 
        return logits