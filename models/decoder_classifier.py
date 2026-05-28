# パーツを組み立てるよ
# config: d_model, d_k
# input: num_embeddings(単語の種類), 単語idの並びx
# output: attentionベクトル, 素の単語ベクトル

import torch.nn as nn
import config
from .layers import SelfAttention
from .layers import FFN
from .layers import PositionalEncoding
from .layers import Decoder

class AhoformerDecoder(nn.Module): # modelを呼び出すと、initで定義してforwardで実行までを自動でやってくれる。
    def __init__(self, num_embeddings):
        super().__init__()
        # 埋め込み層
        self.embedding_layer = nn.Embedding(num_embeddings, config.d_model)

         # 位置エンコーディング層を初期化する
        self.pos_encoder = PositionalEncoding(config.d_model, 128)
        
        # decoderのまとまり
        self.decoder = Decoder()
        
        # 出力層
        self.output_linear = nn.Linear(config.d_model, num_embeddings)

    def forward(self, x): # 学習のさい、input側だからinput_vectors
        input_vectors = self.embedding_layer(x)

        # 埋め込みベクトルの直後に、位置情報を加算する
        pos_vectors = self.pos_encoder(input_vectors)

        # decoderに通す
        out = self.decoder(pos_vectors)
        
        # フラット化して全結合層に入力し、ロジット (logits) を計算
        # flat_out = out.view(out.size(0), -1)  # 形状: (Batch, 8 * d_model)
        # logits = self.classifier(flat_out)  # 形状: (Batch, 1)

        logits = self.output_linear(out)

        return logits