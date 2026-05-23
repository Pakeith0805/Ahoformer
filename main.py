import csv
import torch
import torch.nn as nn # embeddingに使う
import torch.optim as optim # optimizerを使う
import config
import dataset
from models import Ahoformer

# モデルと損失関数の作成
model = Ahoformer(dataset.num_embeddings) # num_embeddingsは単語の種類数。
# 系列変換モデル用
# criterion = nn.MSELoss() # 損失関数。これは平均二乗誤差
# 2値分類用
criterion = nn.BCEWithLogitsLoss()

# === 学習の設定
optimizer = optim.Adam(model.parameters(), lr=config.lr)

# === 訓練データ
input_tensor = dataset.input_tensor
target_tensor = dataset.target_tensor_bin

# === 学習開始
print("--- 学習開始 ---")

for epoch in range (config.epochs):
    # 順伝播。系列
    # attention, input_vectors = model(input_tensor)
    # 順伝播。2値分類
    logits, _ = model(input_tensor)

    # 2値分類では不要
    # target_vectors = model.embedding_layer(target_tensor).detach()

    # 損失計算。系列
    # loss = criterion(attention, target_vectors)
    # 損失計算。2値分類
    loss = criterion(logits, target_tensor)

    # 誤差逆伝播
    optimizer.zero_grad() # 勾配の初期化
    loss.backward() # 誤差逆伝播。どう動かせばいいか学習する
    optimizer.step() # 重みの更新。backwardで計算した結果をもとに実際に更新する

    # 10エポックごとに損失を表示
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{config.epochs} | Loss: {loss.item():.6f}")

# === 単語を予測
model.eval() # 単語を予測モードに
with torch.no_grad(): # withは、自動で後処理を実行してくれる文。
    # 系列
    # attention, _ = model(input_tensor) # attentionだけ取り出せればいい
    # 2値
    logits, _ = model(input_tensor)

    # 系列での予測
    # distances = torch.cdist(attention, model.embedding_layer.weight) # attentionの出力と単語ベクトルを照らし合わせ、全部の距離を計算している。
    # predicted_ids = torch.argmin(distances, dim=-1)
    # 2値
    probs = torch.sigmoid(logits)
    predicted_ids = (probs >= 0.5).long()

# 結果を表示。系列
"""
print("\n--- 学習後の予測結果 (前半15件) ---")
for i in range(100):
    num = dataset.numbers[i]
    orig_word = dataset.outputs[i]
    pred_chars = [dataset.id_to_char[idx.item()] for idx in predicted_ids[i]]
    pred_word = "".join(pred_chars).strip()
    print(f"Num: {num} | 元の単語: {orig_word:4s} -> 予測された単語: {pred_word}")
"""
 
# 結果を表示 2値
print("\n--- 学習後の予測結果 (前半15件) ---")
for i in range(200):
    num = dataset.numbers[i]  # 元の数字 (文字列)
    orig_label = int(dataset.outputs[i])  # 正解ラベル (0 または 1)
    pred_label = predicted_ids[i].item()  # 予測ラベル (0 または 1)
    
    # 1のときは "Aho"、0のときは元の数字を表示用にする
    orig_display = "Aho" if orig_label == 1 else num
    pred_display = "Aho" if pred_label == 1 else num
    
    print(f"Num: {num:>3s} | 正解: {orig_display:4s} -> 予測: {pred_display}")