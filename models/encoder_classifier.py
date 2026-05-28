# パーツを組み立てるよ
# config: d_model, d_k
# input: num_embeddings(単語の種類), 単語idの並びx
# output: attentionベクトル, 素の単語ベクトル

import torch.nn as nn
import config
from .layers import SelfAttention
from .layers import FFN
from .layers import PositionalEncoding
from .layers import Encoder

class AhoformerEncoder(nn.Module): # modelを呼び出すと、initで定義してforwardで実行までを自動でやってくれる。
    def __init__(self, num_embeddings):
        super().__init__()
        # 埋め込み層
        self.embedding_layer = nn.Embedding(num_embeddings, config.d_model)

         # 位置エンコーディング層を初期化する
        self.pos_encoder = PositionalEncoding(config.d_model, config.num_digits + 1)
        
        # encoderのまとまり
        self.encoder = Encoder()
        
        # 2値分類するFFN
        self.classifier = nn.Linear(config.d_model, 1)

    def forward(self, x): # 学習のさい、input側だからinput_vectors
        input_vectors = self.embedding_layer(x)

        # 埋め込みベクトルの直後に、位置情報を加算する
        pos_vectors = self.pos_encoder(input_vectors)

        # encoderに通す
        out = self.encoder(pos_vectors)
        
        # [CLS]トークンを含むすべてのトークン出力の平均（Mean Pooling）を計算
        mean_output = out.mean(dim=1)
        logits = self.classifier(mean_output)

        return logits, input_vectors