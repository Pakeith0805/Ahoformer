import csv
import torch
import torch.nn as nn # embeddingに使う

num_head = 2 # ヘッドの数
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
embedded_vectors = embedding_layer(input_tensor)

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

Q = w_q(embedded_vectors)
K = w_k(embedded_vectors)
V = w_v(embedded_vectors)

torch.softmax(Q @ K.T, dim=1) @ V