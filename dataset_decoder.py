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
# decoder追加要素
all_chars.add('[SEP]')
all_chars.add('[MASK]')

# idと文字の対応付け
unique_chars = sorted(list(all_chars))
char_to_id = {char: idx for idx, char in enumerate(unique_chars)}
id_to_char = {idx: char for char, idx in char_to_id.items()}

num_embeddings = len(char_to_id)  # 単語の種類数（行数）

# === テキストをidに変換し、tensorにする
input_ids = [] # 右詰めのリストをidに変換している

# === decoderで追加
for word in numbers:
    # 特殊トークン2つ分のスペースを空けるため、(config.num_digits - 2) 桁で右詰め
    padded_word = f"{word:>{config.num_digits - 2}}"
    # 末尾に [SEP] と [MASK] を結合
    seq = list(padded_word) + ['[SEP]', '[MASK]']
    row_ids = [char_to_id[char] for char in seq]
    input_ids.append(row_ids)

input_tensor = torch.tensor(input_ids, dtype=torch.long)  # それをテンソルにしている。形状: (単語数, 8)

# 系列を出力する場合のtarget tensor
target_ids_seq = [[char_to_id[char] for char in seq] for seq in outputs_split]
target_tensor_seq = torch.tensor(target_ids_seq, dtype=torch.long)  # 形状: (単語数, 8)

# 2値分類の場合のtarget tensor (CrossEntropy用にするため、'0' または '1' の文字IDに変更)
target_ids_bin = [char_to_id[word] for word in outputs] # outputs は "0" または "1"
target_tensor_bin = torch.tensor(target_ids_bin, dtype=torch.long) # 形状: (単語数,)


# === 未知データの変換関数の修正（推論テスト用）
def encode_numbers(number_list, max_len=config.num_digits):
    """
    任意の数字リストをデコーダ入力用のテンソルに変換する（末尾に[SEP],[MASK]）
    """
    input_ids = []
    for num in number_list:
        padded_word = f"{str(num):>{max_len - 2}}"
        seq = list(padded_word) + ['[SEP]', '[MASK]']
        
        row_ids = []
        for char in seq:
            row_ids.append(char_to_id.get(char, char_to_id[' ']))
        input_ids.append(row_ids)
        
    return torch.tensor(input_ids, dtype=torch.long)
