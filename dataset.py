# データセットを読み解くよ
import csv
import torch
import config

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
numbers_split = [list(f"{word:>{config.num_digits}}") for word in numbers]
outputs_split = [list(f"{word:>{config.num_digits}}") for word in outputs]

# ユニークな文字を抽出。0～9とAhoになるはず。(系列)
# 2値分類なら数字になる。あと空白
all_chars = set()
for seq in numbers_split + outputs_split:
    for char in seq:
        all_chars.add(char)

# idと文字の対応付け
unique_chars = sorted(list(all_chars))
if '[CLS]' not in unique_chars:
    unique_chars.append('[CLS]')
char_to_id = {char: idx for idx, char in enumerate(unique_chars)}
id_to_char = {idx: char for char, idx in char_to_id.items()}

num_embeddings = len(char_to_id)  # 単語の種類数（行数）

# === テキストをidに変換し、tensorにする
cls_id = char_to_id['[CLS]']
input_ids = [[cls_id] + [char_to_id[char] for char in seq] for seq in numbers_split] # 先頭に[CLS]を追加し、右詰めのリストをidに変換している
input_tensor = torch.tensor(input_ids, dtype=torch.long)  # それをテンソルにしている。形状: (単語数, 9)

# 系列を出力する場合のtarget tensor
target_ids_seq = [[char_to_id[char] for char in seq] for seq in outputs_split]
target_tensor_seq = torch.tensor(target_ids_seq, dtype=torch.long)  # 形状: (単語数, 8)

# 2値分類の場合のtarget tensor
target_ids_bin = [1 if word == "Aho" else 0 for word in outputs]
target_tensor_bin = torch.tensor(target_ids_bin, dtype=torch.float32).unsqueeze(1) # 出力とtargetのデータ型が一致している必要があるためfloatに



# === 未知データの変換

def encode_numbers(number_list, max_len=config.num_digits):
    """
    任意の数字リストをモデル入力用のテンソルに変換する
    例: [101, 102] -> 先頭にCLSトークンを追加し、右詰め8文字にしてID化したテンソル
    """
    if not number_list:
        return torch.empty((0, max_len + 1), dtype=torch.long)

    # 数値を文字列にし、右詰め8文字のリストにする
    padded_list = [list(f"{str(num):>{max_len}}") for num in number_list]
    
    # 登録されている文字辞書を使ってIDに変換
    input_ids = []
    cls_id = char_to_id['[CLS]']
    for seq in padded_list:
        row_ids = [cls_id]
        for char in seq:
            # 万が一、辞書にない文字が含まれていた場合は空白 ' ' に置き換える安全策
            row_ids.append(char_to_id.get(char, char_to_id[' ']))
        input_ids.append(row_ids)
        
    return torch.tensor(input_ids, dtype=torch.long)