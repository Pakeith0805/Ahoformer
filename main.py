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

with open(csv_file, mode="r", encoding="utf-8-sig") as f:
    # DictReaderを使って列名でアクセス
    reader = csv.DictReader(f)
    for row in reader:
        numbers.append(int(row["number"]))
        outputs.append(row["output"])  # "1", "2", "Aho", ... が入る

# === 単語の辞書を作る

unique_words = list(set(outputs))
unique_words.sort()

word_to_id = {word: idx for idx, word in enumerate(unique_words)}

num_embeddings = len(word_to_id)  # 単語の種類数（行数）
embedding_dim = d_model                 # ベクトルの次元数（列数）

# === ただのテンソルとして重み行列を作成

embedding_layer = nn.Embedding(num_embeddings, embedding_dim)

# === テキストをidに変換し、pytorchで扱えるテンソルにする
input_ids = [word_to_id[word] for word in outputs] # テキストをidに変換
input_tensor = torch.tensor(input_ids, dtype=torch.long) # テンソルにする

# === idを対応するベクトルにする。てか行列。このままQ, K, Vにできる。
embedded_vectors = embedding_layer(input_tensor).detach()

# === 結果の確認
print(f"Embedding層の構造: {embedding_layer}")
print("--- nn.Embedding を使った結果 (前半15件) ---")
for i in range(15):
    word = outputs[i]
    vector = embedded_vectors[i].tolist()  # リスト型に変換
    print(f"Num: {numbers[i]:2d} | Text: {word:4s} -> Vector: {[round(v, 4) for v in vector]}")

# === Q,K,Vを生成
w_q = nn.Linear(d_model, d_k, bias=False)  # 重み行列 W_Q
w_k = nn.Linear(d_model, d_k, bias=False)  # 重み行列 W_K
w_v = nn.Linear(d_model, d_k, bias=False)  # 重み行列 W_V

# === 学習の設定
optimizer = optim.Adam( # 学習対象を設定
    list(w_q.parameters()) + list(w_k.parameters()) + list(w_v.parameters()), # 学習対象となる重みを連結
    lr = 0.01 # 学習率
)
criterion = nn.MSELoss() # 損失関数。これは平均二乗誤差

epochs = 100

# === 学習開始
print("--- 学習開始 ---")

for epoch in range (epochs):
    # 順伝播
    Q = w_q(embedded_vectors)
    K = w_k(embedded_vectors)
    V = w_v(embedded_vectors)

    # attentionを算出
    attention = torch.softmax(Q @ K.T, dim=1) @ V

    # 損失の計算
    loss = criterion(attention, embedded_vectors) # (input(予測値), target(正解))

    # 誤差逆伝播
    optimizer.zero_grad() # 勾配の初期化
    loss.backward() # 誤差逆伝播。どう動かせばいいか学習する
    optimizer.step() # 重みの更新。backwardで計算した結果をもとに実際に更新する

    # 10エポックごとに損失を表示
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.6f}")

# === 単語を予測

with torch.no_grad(): # withは、自動で後処理を実行してくれる文。
    Q = w_q(embedded_vectors)
    K = w_k(embedded_vectors)
    V = w_v(embedded_vectors)

    attention = torch.softmax(Q @ K.T, dim=1) @ V

    # 平均二乗誤差使うと、足しちゃうからダメ
    distances = torch.cdist(attention, embedding_layer.weight) # attentionの出力と単語ベクトルを照らし合わせ、全部の距離を計算している。
    predicted_ids = torch.argmin(distances, dim=-1)

# 結果を表示
print("\n--- 学習後の予測結果 (前半15件) ---")
for i in range(15):
    num = numbers[i]
    orig_word = outputs[i]
    pred_id = predicted_ids[i].item()
    pred_word = unique_words[pred_id]
    print(f"Num: {num:2d} | 元の単語: {orig_word:4s} -> 予測された単語: {pred_word}")