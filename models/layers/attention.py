# attention機構を実装するよ
# config: 
# input: d_model, d_k, embeddingが終わった直後の行列x
# output: attention


import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_k, bias=False) # 入力×出力
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x):
        # x: embedded_vectors (Batch, SeqLen, d_model)
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        scores = torch.matmul(Q, K.transpose(1, 2))
        attention_weights = torch.softmax(scores, dim=1)
        attention = torch.matmul(attention_weights, V)
        return attention