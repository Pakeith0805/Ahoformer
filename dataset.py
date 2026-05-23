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

# 系列を出力する場合のtarget tensor
target_ids_seq = [[char_to_id[char] for char in seq] for seq in outputs_split]
target_tensor_seq = torch.tensor(target_ids_seq, dtype=torch.long)  # 形状: (N, 8)

# 2値分類の場合のtarget tensor
target_ids_bin = [int(word) for word in outputs]
target_tensor_bin = torch.tensor(target_ids_bin, dtype=torch.float32).unsqueeze(1) # 出力とtargetのデータ型が一致している必要があるためfloatに



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