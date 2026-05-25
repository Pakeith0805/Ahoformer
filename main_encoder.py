import torch
import torch.nn as nn # embeddingに使う
import torch.optim as optim # optimizerを使う
import config
import dataset
from models import AhoformerEncoder

# モデルと損失関数の作成
model = AhoformerEncoder(dataset.num_embeddings) # num_embeddingsは単語の種類数。
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
for i in range(1000):
    num = dataset.numbers[i]  # 元の数字 (文字列)
    orig_label = int(dataset.outputs[i])  # 正解ラベル (0 または 1)
    pred_label = predicted_ids[i].item()  # 予測ラベル (0 または 1)
    
    # 1のときは "Aho"、0のときは元の数字を表示用にする
    orig_display = "Aho" if orig_label == 1 else num
    pred_display = "Aho" if pred_label == 1 else num
    
    print(f"Num: {num:>3s} | 正解: {orig_display:4s} -> 予測: {pred_display}")



# ==========================================
# === 未知のデータ (101〜150) でのテスト推論
# ==========================================

test_numbers = list(range(2001, 3000))  # 学習時に見せていないデータ

# 1. 追加した関数でテストデータをテンソルに変換
test_input_tensor = dataset.encode_numbers(test_numbers)

# 2. モデルを評価モードにして推論を実行
model.eval()
with torch.no_grad():
    logits, _ = model(test_input_tensor)
    probs = torch.sigmoid(logits)
    predicted_labels = (probs >= 0.5).long()

# 3. テストデータの正解ラベル (0 or 1) をプログラム側で計算しておく
test_targets = []
for n in test_numbers:
    is_multiple_of_3 = (n % 3 == 0)
    contains_3 = ('3' in str(n))
    test_targets.append(1 if (is_multiple_of_3 or contains_3) else 0)

# 4. 結果の表示と正答率の計算
print("\n--- 未知のデータ (101〜150) での予測結果 ---")
correct_count = 0

for i, num in enumerate(test_numbers):
    orig_label = test_targets[i]
    pred_label = predicted_labels[i].item()
    
    # 1のときは "Aho"、0のときは元の数字を表示用に整形
    orig_display = "Aho" if orig_label == 1 else str(num)
    pred_display = "Aho" if pred_label == 1 else str(num)
    
    # 正解・不正解のマーク判定
    status = "◯" if orig_label == pred_label else "×"
    if orig_label == pred_label:
        correct_count += 1
        
    print(f"Num: {num:>3d} | 正解: {orig_display:4s} -> 予測: {pred_display:4s} | {status}")

# 全体の正答率を表示
accuracy = (correct_count / len(test_numbers)) * 100
print(f"\n未知データでの正解率 (Accuracy): {accuracy:.1f}% ({correct_count}/{len(test_numbers)})")