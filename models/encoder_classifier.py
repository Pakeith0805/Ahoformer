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
        # self.classifier = nn.Linear((config.num_digits + 1) * config.d_model, 1)
        # プーリングver
        self.classifier = nn.Linear(config.d_model, 1)

    def forward(self, x): # 学習のさい、input側だからinput_vectors
        input_vectors = self.embedding_layer(x)

        # 埋め込みベクトルの直後に、位置情報を加算する
        pos_vectors = self.pos_encoder(input_vectors)

        # encoderに通す
        out = self.encoder(pos_vectors)

        # フラット化して全結合層に入力し、ロジット (logits) を計算
        # flat_out = out.view(out.size(0), -1)  # 形状: (Batch, 8 * d_model)
        # flat_out = self.ffn1(flat_out)
        # flat_out = torch.relu(flat_out)
        # flat_out = self.ffn2(flat_out)
        # flat_out = torch.relu(flat_out)
        # logits = self.classifier(flat_out)  # 形状: (Batch, 1)
        
        # [CLS]トークンを含むすべてのトークン出力の平均（Mean Pooling）を計算
        mean_output = out.mean(dim=1)
        logits = self.classifier(mean_output)

        return logits, input_vectors


class AhoformerSpectralEncoder(nn.Module):
    """
    Transformer Encoder model designed specifically for 1D continuous spectral signals (e.g. NIR spectra).
    It projects the continuous 1D signals into a sequence using a 1D Conv layer, adds positional encoding,
    passes it through the Transformer Encoder, and outputs a regression scalar (moisture content).
    """
    def __init__(self):
        super().__init__()
        # Conv1D to map the continuous 1D spectral signal to sequence embeddings
        # input shape: (Batch, 2, 1555) -> output shape: (Batch, d_model, 129)
        self.embedding_layer = nn.Conv1d(
            in_channels=2, 
            out_channels=config.d_model, 
            kernel_size=16, 
            stride=12
        )

        # Positional encoding for sequence length 129
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=512)
        
        # Encoder module (1-layer stack to prevent overfitting)
        self.encoder = nn.Sequential(
            Encoder()
        )

        # Regression head to map pooled sequence representations to moisture content scalar
        self.regressor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1)
        )

    def forward(self, x, return_features=False):
        # x: shape (Batch, 2, 1555)
        
        # Project 1D spectral signal to sequence embeddings
        out = self.embedding_layer(x) # (Batch, d_model, 129)
        out = out.transpose(1, 2)     # (Batch, 129, d_model)

        # Add positional encoding
        out = self.pos_encoder(out)

        # Pass through Transformer Encoder
        out = self.encoder(out)

        # Mean pooling over the sequence dimension
        mean_output = out.mean(dim=1)  # (Batch, d_model)

        if return_features:
            return mean_output

        # Map to regression output
        preds = self.regressor(mean_output) # (Batch, 1)
        
        return preds