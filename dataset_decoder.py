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

# ユニークな文字を抽出。
all_chars = set()
for seq in numbers_split + outputs_split:
    for char in seq:
        all_chars.add(char)

# decoder追加要素
all_chars.add('[SEP]')
all_chars.add('[EOS]')

# idと文字の対応付け
unique_chars = sorted(list(all_chars))
char_to_id = {char: idx for idx, char in enumerate(unique_chars)}
id_to_char = {idx: char for char, idx in char_to_id.items()}

num_embeddings = len(char_to_id)  # 単語の種類数（行数）

max_seq_len = 12

# === テキストをidに変換し、tensorにする
input_ids = []
target_ids = []

for num, out in zip(numbers, outputs):
    # プロンプト部: 5文字の右詰め + [SEP] (長さ 6)
    prompt = list(f"{num:>5}") + ['[SEP]']
    # 応答部: 出力文字列 + [EOS]
    response = list(out) + ['[EOS]']
    
    # 結合して固定長にパディング
    seq = prompt + response
    padding_len = max_seq_len - len(seq)
    if padding_len > 0:
        seq = seq + [' '] * padding_len
    
    # インプットのID化。辞書を参照してidをとってくる
    row_ids = [char_to_id[char] for char in seq]
    input_ids.append(row_ids)
    
    # ターゲットのID化 (Causal LM用: 次トークン予測、プロンプト/パディングは -100 にマスク)
    row_targets = []
    for i in range(max_seq_len - 1):
        if i < 5:  # プロンプト予測位置はマスク (i=4 は [SEP] 予測)
            row_targets.append(-100)
        else:
            target_char = seq[i + 1]
            if target_char == ' ':  # パディング部分はマスク
                row_targets.append(-100)
            else:
                row_targets.append(char_to_id[target_char])
    row_targets.append(-100)  # 末尾の次の予測はマスク
    target_ids.append(row_targets)

input_tensor = torch.tensor(input_ids, dtype=torch.long)  # 形状: (単語数, max_seq_len)
target_tensor = torch.tensor(target_ids, dtype=torch.long)  # 形状: (単語数, max_seq_len)


# === 未知データの変換関数（プロンプト生成用）
def encode_numbers(number_list):
    """
    任意の数字リストをデコーダ入力用のプロンプトテンソルに変換する（[SEP]まで）
    """
    input_ids = []
    for num in number_list:
        prompt = list(f"{num:>5}") + ['[SEP]']
        row_ids = [char_to_id.get(char, char_to_id[' ']) for char in prompt]
        input_ids.append(row_ids)
        
    return torch.tensor(input_ids, dtype=torch.long)
