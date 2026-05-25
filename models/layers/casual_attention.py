# attention機構を実装するよ
# config: 
# input: d_model, d_k, embeddingが終わった直後の行列x
# output: attention


import torch
import torch.nn as nn
import config
import math

class CasualAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False) # 入力×出力
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # attentionくっつけたあとの最終調整の重み
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: embedded_vectors (Batch, SeqLen, d_model)
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        batch_size = x.size(0)
        seq_len = x.size(1)

        # マルチヘッドにする。ヘッドを分割。
        Q_multi = Q.view(batch_size, seq_len, config.num_head, config.d_k)
        K_multi = K.view(batch_size, seq_len, config.num_head, config.d_k)
        V_multi = V.view(batch_size, seq_len, config.num_head, config.d_k)

        # 行列計算する部分を最後に持ってきたいので入れ替え
        Q_multi = Q_multi.transpose(1, 2)
        K_multi = K_multi.transpose(1, 2)
        V_multi = V_multi.transpose(1, 2)

        scores = torch.matmul(Q_multi, K_multi.transpose(2, 3))
        # √d_kで割るよ
        scores = scores / math.sqrt(config.d_k)

        # === ここが普通のself attentionとの違い
        # マスク行列を作成
        mask = torch.tril(torch.ones(seq_len, seq_len, device = x.device))
        # マスクがかかった部分を小さい値に
        scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=3)
        attention = torch.matmul(attention_weights, V_multi)

        # くっつけたいので次元戻す
        attention = attention.transpose(1, 2)
        attention = attention.reshape(batch_size, seq_len, config.d_model)

        attention = self.w_o(attention)

        return attention