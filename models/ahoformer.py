# パーツを組み立てるよ
# config: d_model, d_k
# input: num_embeddings(単語の種類), 単語idの並びsrc, tgt
# output: logits

import torch
import torch.nn as nn
import config
from .layers import Encoder  # models/layers/encoder_layer.py から (1層エンコーダ)
from .layers import Decoder  # models/layers/decoder_layer.py から (1層デコーダ)
from .layers import PositionalEncoding

class Ahoformer(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        # エンコーダとデコーダで共有する単語埋め込み層
        self.embedding = nn.Embedding(num_embeddings, config.d_model)

        # 位置エンコーディング層を初期化する
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=config.num_digits)
        
        # 1層のエンコーダブロック
        self.encoder = Encoder()

        # 1層のデコーダブロック
        self.decoder = Decoder()
        
        # 最終的に文字の予測ID（ボキャブラリ）の確率分布を出すための線形層
        self.output_linear = nn.Linear(config.d_model, num_embeddings)

    def forward(self, src, tgt):
        # src: 入力の数字ID系列 (Batch, SeqLen_src)
        # tgt: ターゲット文字ID系列 (Batch, SeqLen_tgt)
        
        # 1. エンコーダ処理
        src_embedded = self.embedding(src)
        src_pos = self.pos_encoder(src_embedded)
        encoder_outputs = self.encoder(src_pos)  # (Batch, SeqLen_src, d_model)
        
        # 2. デコーダ処理 (エンコーダの出力をクロスアテンションで参照)
        tgt_embedded = self.embedding(tgt)
        tgt_pos = self.pos_encoder(tgt_embedded)
        decoder_outputs = self.decoder(tgt_pos, encoder_outputs)  # (Batch, SeqLen_tgt, d_model)
        
        # 3. 最終出力 (各文字位置におけるボキャブラリのロジット)
        logits = self.output_linear(decoder_outputs)  # (Batch, SeqLen_tgt, num_embeddings)
        
        return logits