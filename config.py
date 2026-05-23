# ハイパーパラメータや設定値を管理するよ。あと使いたいパーツの順序を指定するよ。

# --- ハイパーパラメータの設定 ---
num_head = 1
d_model = 4
d_k = d_model // num_head
epochs = 1000
lr = 0.01
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