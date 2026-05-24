# ffnを作るよ

import torch.nn as nn

class FFN(nn.Module): # modelを呼び出すと、initで定義してforwardで実行までを自動でやってくれる。
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.ffn_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.ffn_2 = nn.Linear(d_ff, d_model)

    def forward(self, x): # 
        mid_in = self.ffn_1(x)
        mid_out = self.relu(mid_in)
        out = self.ffn_2(mid_out)

        return out