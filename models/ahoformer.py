# パーツを組み立てるよ
# config: d_model, d_k
# input: num_embeddings(単語の種類), 単語idの並びx
# output: attentionベクトル, 素の単語ベクトル

import torch.nn as nn
import config
from .layers import SelfAttention
from .layers import FFN

class Ahoformer(nn.Module): # modelを呼び出すと、initで定義してforwardで実行までを自動でやってくれる。
    def __init__(self, num_embeddings):
        super().__init__()
        # 埋め込み層
        self.embedding_layer = nn.Embedding(num_embeddings, config.d_model)
        # self-attention
        self.attention = SelfAttention(config.d_model, config.d_k)

        # 普通のffn
        self.ffn = FFN(config.d_model, config.d_ff)
        
        # 8文字を2値に分類するFFN
        self.classifier = nn.Linear(config.num_digits * config.d_model, 1)

    def forward(self, x): # 学習のさい、input側だからinput_vectors
        input_vectors = self.embedding_layer(x)
        out = self.attention(input_vectors)
        out = self.ffn(out)
        
        # フラット化して全結合層に入力し、ロジット (logits) を計算
        flat_out = out.view(out.size(0), -1)  # 形状: (Batch, 8 * d_model)
        logits = self.classifier(flat_out)  # 形状: (Batch, 1)

        return logits, input_vectors