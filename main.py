import torch
import torch.nn as nn
import torch.optim as optim
import config
import dataset
from models import Ahoformer

# デバイスの設定 (GPUがあれば使用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# モデルと損失関数の作成
model = Ahoformer(dataset.num_embeddings).to(device)

# 多クラス分類用の交差エントロピー
criterion = nn.CrossEntropyLoss()

# 学習の設定 (Transformerの学習率としては0.001が安定します)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練データ
input_tensor = dataset.input_tensor.to(device)         # (Batch, 8)
target_tensor = dataset.target_tensor_seq.to(device)   # (Batch, 8)

# デコーダの入力 (tgt) の準備
# Teacher Forcing用のターゲット。先頭に開始トークン（スペースのID）を追加し、最後の1文字を削る
batch_size = input_tensor.size(0)
sos_id = dataset.char_to_id[' ']
sos_tokens = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
decoder_input = torch.cat([sos_tokens, target_tensor[:, :-1]], dim=1) # (Batch, 8)

# 学習開始
print("--- 学習開始 ---")

for epoch in range(config.epochs):
    model.train()
    
    # 順伝播
    logits = model(input_tensor, decoder_input)  # (Batch, 8, num_embeddings)
    
    # 損失計算 (Batch * 8, num_embeddings) と (Batch * 8) に平坦化して計算する
    loss = criterion(logits.view(-1, dataset.num_embeddings), target_tensor.view(-1))
    
    # 誤差逆伝播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 50エポックごとに損失を表示
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{config.epochs} | Loss: {loss.item():.6f}")

# === 単語を予測 (自己回帰生成)
model.eval()

def generate_sequences(src_tensor):
    """
    自己回帰（Autoregressive）で1文字ずつ生成する
    src_tensor: (Batch, 8)
    """
    eval_batch_size = src_tensor.size(0)
    # 開始トークンのみで初期化 (長さ1)
    tgt_eval = torch.full((eval_batch_size, 1), sos_id, dtype=torch.long, device=device)
    
    with torch.no_grad():
        # エンコーダの出力を一回だけ計算して保持
        src_embedded = model.embedding(src_tensor)
        src_pos = model.pos_encoder(src_embedded)
        encoder_outputs = model.encoder(src_pos)
        
        # 8回ループして、合計9文字 (開始トークン + 8文字の生成結果) にする
        for _ in range(config.num_digits):
            tgt_embedded = model.embedding(tgt_eval)
            tgt_pos = model.pos_encoder(tgt_embedded)
            decoder_outputs = model.decoder(tgt_pos, encoder_outputs)
            logits = model.output_linear(decoder_outputs)  # (Batch, current_len, num_embeddings)
            
            # 最後の時間ステップの予測値を取得
            next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (Batch, 1)
            # 生成結果を結合
            tgt_eval = torch.cat([tgt_eval, next_tokens], dim=1)
            
    return tgt_eval

# 訓練データでの予測とデコード
predicted_ids = generate_sequences(input_tensor)

# 結果を表示
print("\n--- 学習後の予測結果 (前半15件) ---")
for i in range(15):
    num = dataset.numbers[i]
    orig_label_raw = dataset.outputs[i] # "0" または "1"
    
    # 生成されたIDリストを文字に変換 (インデックス0の開始トークンはスキップ)
    pred_chars = [dataset.id_to_char[idx.item()] for idx in predicted_ids[i][1:]]
    pred_label_raw = "".join(pred_chars).strip() # "0" または "1"
    
    # 表示用に整形 (1ならAho、0なら元の数字)
    orig_display = "Aho" if orig_label_raw == "1" else num
    pred_display = "Aho" if pred_label_raw == "1" else num
    
    status = "◯" if orig_label_raw == pred_label_raw else "×"
    print(f"Num: {num:>3s} | 正解: {orig_display:4s} -> 予測: {pred_display:4s} | {status}")


# ==========================================
# === 未知のデータ (2001〜2100) でのテスト推論
# ==========================================
test_numbers = list(range(2001, 2100))

# 1. テストデータをテンソルに変換
test_input_tensor = dataset.encode_numbers(test_numbers).to(device)

# 2. 推論実行
test_predicted_ids = generate_sequences(test_input_tensor)

# 3. テストデータの正解ラベル ("0" または "1") を作成
test_targets_raw = []
for n in test_numbers:
    is_multiple_of_3 = (n % 3 == 0)
    contains_3 = ('3' in str(n))
    test_targets_raw.append("1" if (is_multiple_of_3 or contains_3) else "0")

# 4. 結果の表示と正答率の計算
print("\n--- 未知のデータ (2001〜2100) での予測結果 (前半15件) ---")
correct_count = 0

for i, num in enumerate(test_numbers):
    orig_label_raw = test_targets_raw[i]
    
    # 生成結果からデコード (インデックス0はスキップ)
    pred_chars = [dataset.id_to_char[idx.item()] for idx in test_predicted_ids[i][1:]]
    pred_label_raw = "".join(pred_chars).strip()
    
    # 表示用に整形
    orig_display = "Aho" if orig_label_raw == "1" else str(num)
    pred_display = "Aho" if pred_label_raw == "1" else str(num)
    
    # 正解・不正解のマーク判定
    status = "◯" if orig_label_raw == pred_label_raw else "×"
    if orig_label_raw == pred_label_raw:
        correct_count += 1
        
    if i < 15:
        print(f"Num: {num:>3d} | 正解: {orig_display:4s} -> 予測: {pred_display:4s} | {status}")

# 全体の正答率を表示
accuracy = (correct_count / len(test_numbers)) * 100
print(f"\n未知データでの正解率 (Accuracy): {accuracy:.1f}% ({correct_count}/{len(test_numbers)})")