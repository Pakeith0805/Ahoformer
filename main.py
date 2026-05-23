import csv
import torch
import torch.nn as nn # embeddingに使う
import torch.optim as optim # optimizerを使う

num_head = 1 # ヘッドの数
d_model = 4 # トークンの次元。単語ベクトルの次元みたいな
d_k = d_model // num_head

# ================== embedding

# === csvファイルを読み取る
csv_file = "aho_dataset_standard.csv"
numbers = []
outputs = []
len_seq = 8 # リストの長さ。扱える数字の桁数の最大

with open(csv_file, mode="r", encoding="utf-8-sig") as f:
    # DictReaderを使って列名でアクセス
    reader = csv.DictReader(f)
    for row in reader:
        numbers.append(row["number"])  # "1", "2", "3", ... が入る。intに変換しない
        outputs.append(row["output"])  # "1", "2", "Aho", ... が入る

# 右詰めで8文字にする。空白で埋める。
numbers_split = [list(f"{word:>8}") for word in numbers]
outputs_split = [list(f"{word:>8}") for word in outputs]

# ユニークな文字を抽出。0～9とAhoになるはず。
all_chars = set()
for seq in numbers_split + outputs_split:
    for char in seq:
        all_chars.add(char)

# idと文字の対応付け
unique_chars = sorted(list(all_chars))
char_to_id = {char: idx for idx, char in enumerate(unique_chars)}
id_to_char = {idx: char for char, idx in char_to_id.items()}

num_embeddings = len(char_to_id)  # 単語の種類数（行数）
embedding_dim = d_model           # ベクトルの次元数（列数）


# === テキストをidに変換し、tensorにする
input_ids = [[char_to_id[char] for char in seq] for seq in numbers_split] # 右詰めのリストをidに変換している
input_tensor = torch.tensor(input_ids, dtype=torch.long)  # それをテンソルにしている。形状: (N, 8)
# 系列の場合のtarget
# target_ids = [[char_to_id[char] for char in seq] for seq in outputs_split]
# target_tensor = torch.tensor(target_ids, dtype=torch.long)  # 形状: (N, 8)
# 2値分類の場合のtarget tensor
target_ids = [int(word) for word in outputs]
target_tensor = torch.tensor(target_ids, dtype=torch.float32).unsqueeze(1) # 出力とtargetのデータ型が一致している必要があるためfloatに

# === ただのtensorとして重み行列を作成
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)

# === idを対応するベクトルにする。てか行列。このままQ, K, Vにできる。
embedded_vectors = embedding_layer(input_tensor).detach()
# 2値分類では不要
# target_vectors = embedding_layer(target_tensor).detach()     # 正解ベクトル (N, 8, 4)

# 確認表示
print(f"辞書（文字数: {num_embeddings}）: {char_to_id}")
print(f"入力テンソルの形状: {input_tensor.shape}")       # 例: torch.Size([9999, 8])
print(f"埋め込みテンソルの形状: {embedded_vectors.shape}") # 例: torch.Size([9999, 8, 4])
# 最初の3件のデータを確認
for i in range(3):
    orig = outputs[i]
    split = outputs_split[i]
    ids = input_ids[i]
    print(f"元の単語: {orig:6s} -> 分割: {split} -> ID: {ids}")

# === Q,K,Vを生成
w_q = nn.Linear(d_model, d_k, bias=False)  # 重み行列 W_Q
w_k = nn.Linear(d_model, d_k, bias=False)  # 重み行列 W_K
w_v = nn.Linear(d_model, d_k, bias=False)  # 重み行列 W_V

# 8文字を2値に分類するFFN
classifier = nn.Linear(8 * d_k, 1)

# === 学習の設定
optimizer = optim.Adam( # 学習対象を設定
    list(w_q.parameters()) + list(w_k.parameters()) + list(w_v.parameters()) + list(classifier.parameters()), # 学習対象となる重みを連結
    lr = 0.01 # 学習率
)
# 系列変換モデル用
# criterion = nn.MSELoss() # 損失関数。これは平均二乗誤差
# 2値分類用
criterion = nn.BCEWithLogitsLoss()

epochs = 1000

# === 学習開始
print("--- 学習開始 ---")

for epoch in range (epochs):
    # 順伝播。系列
    # attention = ...
    # 順伝播。2値分類
    Q = w_q(embedded_vectors)
    K = w_k(embedded_vectors)
    V = w_v(embedded_vectors)

    # attentionを算出
    # Q,K: (単語ベクトルの本数, 数字の桁数, d_k)
    scores = torch.matmul(Q, K.transpose(1, 2)) # この場合、単語ベクトルの本数部分に関して、行列計算は行わない。そこは固定
    attention_weights = torch.softmax(scores, dim=1)
    attention = torch.matmul(attention_weights, V)

    # フラット化して全結合層に入力し、ロジット (logits) を計算
    flat_out = attention.view(attention.size(0), -1)  # 形状: (Batch, 8 * d_k)
    logits = classifier(flat_out)  # 形状: (Batch, 1)

    # 2値分類では不要
    # target_vectors = ...

    # 損失の計算。系列
    # loss = criterion(attention, target_vectors)
    # 損失の計算。2値分類
    loss = criterion(logits, target_tensor)

    # 誤差逆伝播
    optimizer.zero_grad() # 勾配の初期化
    loss.backward() # 誤差逆伝播。どう動かせばいいか学習する
    optimizer.step() # 重みの更新。backwardで計算した結果をもとに実際に更新する

    # 10エポックごとに損失を表示
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.6f}")

# === 単語を予測

with torch.no_grad(): # withは、自動で後処理を実行してくれる文。
    # 系列
    # attention = ...
    # 2値
    Q = w_q(embedded_vectors)
    K = w_k(embedded_vectors)
    V = w_v(embedded_vectors)

    scores = torch.matmul(Q, K.transpose(1, 2))
    attention_weights = torch.softmax(scores, dim=1)
    attention = torch.matmul(attention_weights, V)

    flat_out = attention.view(attention.size(0), -1)
    logits = classifier(flat_out)

    # 系列での予測
    # distances = torch.cdist(attention, embedding_layer.weight) # attentionの出力と単語ベクトルを照らし合わせ、全部の距離を計算している。
    # predicted_ids = torch.argmin(distances, dim=-1)
    # 2値
    probs = torch.sigmoid(logits)
    predicted_ids = (probs >= 0.5).long()

# 結果を表示。系列
"""
print("\n--- 学習後の予測結果 (前半15件) ---")
for i in range(100):
    num = numbers[i]
    orig_word = outputs[i]
    pred_chars = [id_to_char[idx.item()] for idx in predicted_ids[i]]
    pred_word = "".join(pred_chars).strip()
    print(f"Num: {num} | 元の単語: {orig_word:4s} -> 予測された単語: {pred_word}")
"""

# 結果を表示 2値
print("\n--- 学習後の予測結果 (前半15件) ---")
for i in range(100):
    num = numbers[i]  # 元の数字 (文字列)
    orig_label = int(outputs[i])  # 正解ラベル (0 または 1)
    pred_label = predicted_ids[i].item()  # 予測ラベル (0 または 1)

    # 1のときは "Aho"、0のときは元の数字を表示用にする
    orig_display = "Aho" if orig_label == 1 else num
    pred_display = "Aho" if pred_label == 1 else num

    print(f"Num: {num:>3s} | 正解: {orig_display:4s} -> 予測: {pred_display}")