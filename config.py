# ハイパーパラメータや設定値を管理するよ。あと使いたいパーツの順序を指定するよ。

# --- ハイパーパラメータの設定 ---
num_head = 4
d_model = 128 # 単語の次元数
d_k = d_model // num_head
epochs = 60
lr = 0.001
num_digits = 8 # 何桁の数字まで扱えるようにするか
d_ff = 256 # FFNの中間層の隠れ次元数
dropout = 0.05
weight_decay = 1e-4
num_train_data = 40000
num_test_data = 10000
# --- スイッチの設定 ---
USE_RESIDUAL = False   # 残差接続を使うか
USE_LAYERNORM = False  # レイヤー正規化を使うか
# --- 組み立てたいレイヤーの順番 ---
# コメントアウト（#）するだけで、そのレイヤーを排除して実験できます
MODEL_PIPELINE = [
    "embedding",
    "positional",
    "attention",
    "ffn",
    "classifier"
]