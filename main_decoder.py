import torch
import torch.nn as nn
import torch.optim as optim
import config
import dataset_decoder
from models import AhoformerDecoder

# モデルと損失関数の作成
model = AhoformerDecoder(dataset_decoder.num_embeddings)
criterion = nn.CrossEntropyLoss(ignore_index=-100)  # マスクされたトークン(-100)を無視する

# === 学習の設定
optimizer = optim.Adam(model.parameters(), lr=config.lr)

# === 訓練データ
input_tensor = dataset_decoder.input_tensor     # 形状: (Batch, max_seq_len)
target_tensor = dataset_decoder.target_tensor   # 形状: (Batch, max_seq_len)

# === 学習開始
print("--- 学習開始 ---")

for epoch in range(config.epochs):
    model.train()
    # 順伝播
    logits = model(input_tensor)  # 形状: (Batch, max_seq_len, num_embeddings)

    # 損失計算 (Batch*max_seq_len, num_embeddings) と (Batch*max_seq_len) にフラット化して計算
    loss = criterion(logits.view(-1, dataset_decoder.num_embeddings), target_tensor.view(-1))

    # 誤差逆伝播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 10エポックごとに損失を表示
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{config.epochs} | Loss: {loss.item():.6f}")


# === 自己回帰生成関数の定義
def generate_autoregressive(model, number, char_to_id, id_to_char, max_len=12):
    model.eval()
    with torch.no_grad():
        # 1. プロンプトを作成: 例 "  123[SEP]" (長さ 6)
        prompt_str = f"{number:>5}"
        prompt_chars = list(prompt_str) + ['[SEP]']
        input_ids = [char_to_id[c] for c in prompt_chars]
        
        # デコーダーループ
        for _ in range(max_len - len(prompt_chars)):
            # テンソル化してバッチ次元を追加 (1, seq_len)
            input_t = torch.tensor([input_ids], dtype=torch.long, device=next(model.parameters()).device)
            
            # 推論
            logits = model(input_t)  # (1, seq_len, num_embeddings)
            
            # 最後のトークンの出力ロジットを取得
            last_logit = logits[0, -1, :]  # (num_embeddings,)
            
            # 最も確率の高いIDを取得
            pred_id = last_logit.argmax(dim=-1).item()
            
            # 終了トークンならループを抜ける
            if pred_id == char_to_id['[EOS]']:
                break
                
            # 出力系列に追加
            input_ids.append(pred_id)
            
        # [SEP] より後ろの部分を取り出して文字列に戻す
        sep_idx = prompt_chars.index('[SEP]')
        generated_ids = input_ids[sep_idx + 1:]
        generated_chars = [id_to_char[idx] for idx in generated_ids]
        return "".join(generated_chars)


# 結果を表示 (前半15件)
print("\n--- 学習後の予測結果 (前半15件) ---")
for i in range(15):
    num = dataset_decoder.numbers[i]
    orig_label = dataset_decoder.outputs[i]
    pred_label = generate_autoregressive(
        model, num, dataset_decoder.char_to_id, dataset_decoder.id_to_char
    )
    
    status = "◯" if orig_label == pred_label else "×"
    print(f"Num: {num:>3s} | 正解: {orig_label:4s} -> 予測: {pred_label:4s} | {status}")


# ==========================================
# === 未知のデータ (2001〜2100) でのテスト推論
# ==========================================
test_numbers = list(range(2001, 2101))

print("\n--- 未知のデータ (2001〜2100) での予測結果 (前半15件) ---")
correct_count = 0

for i, num in enumerate(test_numbers):
    is_multiple_of_3 = (num % 3 == 0)
    contains_3 = ('3' in str(num))
    orig_label = "Aho" if (is_multiple_of_3 or contains_3) else str(num)
    
    # 自己回帰生成
    pred_label = generate_autoregressive(
        model, num, dataset_decoder.char_to_id, dataset_decoder.id_to_char
    )
    
    status = "◯" if orig_label == pred_label else "×"
    if orig_label == pred_label:
        correct_count += 1
        
    if i < 15:  # 前半15件を表示
        print(f"Num: {num:>4d} | 正解: {orig_label:4s} -> 予測: {pred_label:4s} | {status}")

# 全体の正答率を表示
accuracy = (correct_count / len(test_numbers)) * 100
print(f"\n未知データでの正解率 (Accuracy): {accuracy:.1f}% ({correct_count}/{len(test_numbers)})")

# === 対話テストモード (ターミナルからの入力受け取り)
print("\n" + "="*40)
print("--- 対話テストモード (終了するには Enter キーのみ、または Ctrl+C) ---")
try:
    while True:
        user_input = input("数字を入力してください: ").strip()
        if not user_input:
            print("終了します。")
            break
        if not user_input.isdigit():
            print("エラー: 半角数字のみ入力可能です。")
            continue
            
        num = int(user_input)
        # 自己回帰生成を実行
        pred_label = generate_autoregressive(
            model, num, dataset_decoder.char_to_id, dataset_decoder.id_to_char
        )
        print(f"入力: {num:>5d} -> 生成（予測）結果: {pred_label}\n")
except KeyboardInterrupt:
    print("\n終了します。")