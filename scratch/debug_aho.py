import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("c:\\Users\\katis\\Projects\\Kurabayashi\\Ahoformer")
import config
import dataset
from models import Ahoformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 学習率を少し下げて安定させる
lr = 0.001
epochs = 500

model = Ahoformer(dataset.num_embeddings).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

input_tensor = dataset.input_tensor.to(device)
target_tensor = dataset.target_tensor_seq.to(device)

batch_size = input_tensor.size(0)
sos_id = dataset.char_to_id[' ']
sos_tokens = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
decoder_input = torch.cat([sos_tokens, target_tensor[:, :-1]], dim=1)

print("--- Start Debug Training ---")
for epoch in range(epochs):
    model.train()
    logits = model(input_tensor, decoder_input)
    loss = criterion(logits.view(-1, dataset.num_embeddings), target_tensor.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.6f}")

model.eval()

# 1. 教師強制（Teacher Forcing）での予測結果を確認
print("\n=== Teacher Forcing Predictions ===")
with torch.no_grad():
    train_logits = model(input_tensor, decoder_input)
    train_preds = train_logits.argmax(dim=-1) # (Batch, 8)

for i in range(10):
    num = dataset.numbers[i]
    orig_word = dataset.outputs[i]
    pred_ids = train_preds[i].tolist()
    pred_chars = [dataset.id_to_char[idx] for idx in pred_ids]
    pred_word = "".join(pred_chars)
    print(f"Num: {num} | Target IDs: {target_tensor[i].tolist()} | Pred IDs: {pred_ids} | Pred: '{pred_word}'")

# 2. 自己回帰（Autoregressive）での予測結果を確認
print("\n=== Autoregressive Predictions ===")
def generate_sequences(src_tensor):
    eval_batch_size = src_tensor.size(0)
    tgt_eval = torch.full((eval_batch_size, 1), sos_id, dtype=torch.long, device=device)
    
    with torch.no_grad():
        src_embedded = model.embedding(src_tensor)
        src_pos = model.pos_encoder(src_embedded)
        encoder_outputs = model.encoder(src_pos)
        
        for step in range(config.num_digits - 1):
            tgt_embedded = model.embedding(tgt_eval)
            tgt_pos = model.pos_encoder(tgt_embedded)
            decoder_outputs = model.decoder(tgt_pos, encoder_outputs)
            logits = model.output_linear(decoder_outputs)
            
            next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt_eval = torch.cat([tgt_eval, next_tokens], dim=1)
    return tgt_eval

autoreg_preds = generate_sequences(input_tensor)
for i in range(10):
    num = dataset.numbers[i]
    pred_ids = autoreg_preds[i].tolist()
    pred_chars = [dataset.id_to_char[idx] for idx in pred_ids]
    pred_word = "".join(pred_chars)
    print(f"Num: {num} | Autoreg Pred IDs: {pred_ids} | Pred: '{pred_word}'")
