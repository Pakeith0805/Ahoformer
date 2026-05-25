# encoderをまとめるよ

import torch.nn as nn
import config
from .self_attention import SelfAttention
from .ffn import FFN

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self-attention
        self.attention = SelfAttention(config.d_model, config.d_k)

        # 層正規化
        self.layernorm1 = nn.LayerNorm(normalized_shape = config.d_model)
        self.layernorm2 = nn.LayerNorm(normalized_shape = config.d_model)

        # 普通のffn
        self.ffn = FFN(config.d_model, config.d_ff)

    def forward(self, x): # 位置エンコーディングまで終わったテンソルを受け取る
        # 残差接続
        out = x + self.attention(x)
        out = self.layernorm1(out)

        # 残差接続
        out = out + self.ffn(out)
        out = self.layernorm2(out)

        return out