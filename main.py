import csv
import torch
import torch.nn as nn # embeddingに使う
import torch.optim as optim # optimizerを使う

num_head = 4 # ヘッドの数
d_model = 64 # トークンの次元。単語ベクトルの次元みたいな
d_k = d_model // num_head
dim_feedforward = 128 # 隠れ層の次元
num_layers = 2 # encoderを何層重ねるか

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




# 2値分類では不要
# target_vectors = embedding_layer(target_tensor).detach()     # 正解ベクトル (N, 8, 4)

# 確認表示
print(f"辞書（文字数: {num_embeddings}）: {char_to_id}")
print(f"入力テンソルの形状: {input_tensor.shape}")       # 例: torch.Size([9999, 8])
# print(f"埋め込みテンソルの形状: {embedded_vectors.shape}") # 例: torch.Size([9999, 8, 4])
# 最初の3件のデータを確認
for i in range(3):
    orig = outputs[i]
    split = outputs_split[i]
    ids = input_ids[i]
    print(f"元の単語: {orig:6s} -> 分割: {split} -> ID: {ids}")

# =============transformerEncoder使う場合

class Ahoformer(nn.Module):
    def __init__(self, num_head, d_model, d_k, dim_feedforward, num_layers):
        super().__init__()
        # === ただのtensorとして重み行列を作成
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_head, 
            dim_feedforward=dim_feedforward,
            batch_first=True  # データの形状を (batch, seq, feature) に指定
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # 8文字を2値に分類するFFN
        self.classifier = nn.Linear(8 * d_model, 1)
    
    def forward(self, input_tensor):
        # === idを対応するベクトルにする。てか行列。このままQ, K, Vにできる。
        embedded_vectors = self.embedding_layer(input_tensor)#.detach()
        attention_out = self.transformer_encoder(embedded_vectors)

        # フラット化して全結合層に入力し、ロジット (logits) を計算
        flat_out = attention_out.view(attention_out.size(0), -1)  # 形状: (Batch, 8 * d_k)
        logits = self.classifier(flat_out)  # 形状: (Batch, 1)

        return logits, embedded_vectors

model = Ahoformer(num_head = num_head, d_model = d_model, d_k = d_k, dim_feedforward = dim_feedforward, num_layers = num_layers)

# === 学習の設定
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 系列変換モデル用
# criterion = nn.MSELoss() # 損失関数。これは平均二乗誤差
# 2値分類用
criterion = nn.BCEWithLogitsLoss()

epochs = 1000

# === 学習開始
print("--- 学習開始 ---")

for epoch in range (epochs):
    model.train()
    # 順伝播。2値分類
    logits, _ = model(input_tensor)
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

model.eval()
with torch.no_grad(): # withは、自動で後処理を実行してくれる文。
    # 順伝播
    logits, _ = model(input_tensor)

    # 損失計算
    probs = torch.sigmoid(logits)
    predicted_ids = (probs >= 0.5).long()

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









# === 未知データの変換

def encode_numbers(number_list, max_len=8):
    """
    任意の数字リストをモデル入力用のテンソルに変換する
    例: [101, 102] -> 右詰め8文字にしてID化したテンソル
    """
    # 数値を文字列にし、右詰め8文字のリストにする
    padded_list = [list(f"{str(num):>{max_len}}") for num in number_list]
    
    # 登録されている文字辞書を使ってIDに変換
    input_ids = []
    for seq in padded_list:
        row_ids = []
        for char in seq:
            # 万が一、辞書にない文字が含まれていた場合は空白 ' ' に置き換える安全策
            row_ids.append(char_to_id.get(char, char_to_id[' ']))
        input_ids.append(row_ids)
        
    return torch.tensor(input_ids, dtype=torch.long)



# ==========================================
# === 未知のデータ (101〜150) でのテスト推論
# ==========================================
test_numbers = list(range(2001, 3000))  # 学習時に見せていないデータ

# 1. 追加した関数でテストデータをテンソルに変換
test_input_tensor = encode_numbers(test_numbers)

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