# データセットを読み解くよ
import csv
import torch

csv_file = "aho_dataset_standard.csv"
numbers = []
outputs = []

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

# === テキストをidに変換し、tensorにする
input_ids = [[char_to_id[char] for char in seq] for seq in numbers_split] # 右詰めのリストをidに変換している
input_tensor = torch.tensor(input_ids, dtype=torch.long)  # それをテンソルにしている。形状: (N, 8)
target_ids = [[char_to_id[char] for char in seq] for seq in outputs_split]
target_tensor = torch.tensor(target_ids, dtype=torch.long)  # 形状: (N, 8)