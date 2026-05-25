# decoderを実装するよ

import torch.nn as nn
import config
from .casual_attention import CasualAttention
from .cross_attention import CrossAttention
from .ffn import FFN

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # casual-attention
        self.casual_attention = CasualAttention(config.d_model)

        # cross-attention
        self.cross_attention = CrossAttention(config.d_model)

        # 層正規化
        self.layernorm1 = nn.LayerNorm(normalized_shape = config.d_model)
        # self.layernorm2 = nn.LayerNorm(normalized_shape = config.d_model)
        self.layernorm3 = nn.LayerNorm(normalized_shape = config.d_model)

        # 普通のffn
        self.ffn = FFN(config.d_model, config.d_ff)

    def forward(self, x):# , encoder_outputs): # 位置エンコーディングまで終わったテンソルと、encoderを通り抜けたテンソルを受け取る
        # casual-attention
        out = x + self.casual_attention(x)
        out = self.layernorm1(out)

        # cross-attention
        # out = out + self.cross_attention(out, encoder_outputs)
        # out = self.layernorm2(out)

        # FFN
        out = out + self.ffn(out)
        out = self.layernorm3(out)

        return out