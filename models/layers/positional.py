# 位置エンコーディングをするよ
# あんまり理解してない。ここはAIに書かせた

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module): # modelを呼び出すと、initで定義してforwardで実行までを自動でやってくれる。
    def __init__(self, d_model, max_len = 512):
        super().__init__()

         # (max_len, d_model) の形状でテーブルの初期値（ゼロ）を作成
        pe = torch.zeros(max_len, d_model)
        
        # 各位置 (0, 1, 2, ..., max_len-1) のテンソルを作成
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        
        # 周波数の計算用分母 (div_term)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        
        # 偶数次元にはsin、奇数次元にはcosを適用して位置情報を埋め込む
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (Batch, SeqLen, d_model) にブロードキャストできるよう、
        # 先頭にサイズ1のバッチ次元を追加して (1, max_len, d_model) に変換
        pe = pe.unsqueeze(0)
        
        # パラメータ（勾配計算対象）ではなく、固定のバッファとして登録
        self.register_buffer("pe", pe)

    def forward(self, x): 
        # x の形状: (Batch, SeqLen, d_model)
        seq_len = x.size(1)
        
        # 現在のシーケンス長 (seq_len) の分だけ切り出して入力に加算する
        return x + self.pe[:, :seq_len, :]