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
import torch

class AhoformerEncoder(nn.Module): # modelを呼び出すと、initで定義してforwardで実行までを自動でやってくれる。
    def __init__(self, num_embeddings):
        super().__init__()
        # 埋め込み層
        self.embedding_layer = nn.Embedding(num_embeddings, config.d_model)

        # 位置エンコーディング層を初期化する
        self.pos_encoder = PositionalEncoding(config.d_model, config.num_digits + 1)
        
        # encoderのまとまり
        self.encoder = Encoder()

        self.ffn1 = nn.Linear((config.num_digits + 1) * config.d_model, (config.num_digits + 1) * config.d_model)
        self.ffn2 = nn.Linear((config.num_digits + 1) * config.d_model, (config.num_digits + 1) * config.d_model)
        
        # 2値分類するFFN
        # フラットアウトver
        self.classifier = nn.Linear((config.num_digits + 1) * config.d_model, 1)
        # プーリングver
        # self.classifier = nn.Linear(config.d_model, 1)

    def forward(self, x): # 学習のさい、input側だからinput_vectors
        input_vectors = self.embedding_layer(x)

        # 埋め込みベクトルの直後に、位置情報を加算する
        pos_vectors = self.pos_encoder(input_vectors)

        # encoderに通す
        out = self.encoder(pos_vectors)

        # フラット化して全結合層に入力し、ロジット (logits) を計算
        flat_out = out.view(out.size(0), -1)  # 形状: (Batch, 8 * d_model)
        flat_out = self.ffn1(flat_out)
        flat_out = torch.relu(flat_out)
        flat_out = self.ffn2(flat_out)
        flat_out = torch.relu(flat_out)
        logits = self.classifier(flat_out)  # 形状: (Batch, 1)
        
        # [CLS]トークンを含むすべてのトークン出力の平均（Mean Pooling）を計算
        #mean_output = out.mean(dim=1)
        #logits = self.classifier(mean_output)

        return logits, input_vectors