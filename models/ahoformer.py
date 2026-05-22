# パーツを組み立てるよ
# config: d_model, d_k
# input: num_embeddings(単語の種類), 単語idの並びx
# output: attentionベクトル, 素の単語ベクトル

import torch.nn as nn
import config
from .layers import SelfAttention

class Ahoformer(nn.Module): # modelを呼び出すと、initで定義してforwardで実行までを自動でやってくれる。
    def __init__(self, num_embeddings):
        super().__init__()
        # 埋め込み層
        self.embedding_layer = nn.Embedding(num_embeddings, config.d_model)
        # self-attention
        self.attention = SelfAttention(config.d_model, config.d_k)
    def forward(self, x): # 学習のさい、input側だからinput_vectors
        input_vectors = self.embedding_layer(x).detach()
        attention_out = self.attention(input_vectors)
        return attention_out, input_vectors