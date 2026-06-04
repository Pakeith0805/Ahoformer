import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
import config
from models import AhoformerSpectralEncoder

# Set seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Raw Data
train_df = pd.read_csv("train.csv", encoding="cp932")
y = train_df.iloc[:, 3].values
X_raw = train_df.iloc[:, 4:].values
species_ids = train_df["species number"].values

test_df = pd.read_csv("test.csv", encoding="cp932")
X_test_raw = test_df.iloc[:, 3:].values

print(f"Loaded {len(y)} training samples. Number of unique species: {len(np.unique(species_ids))}")
print(f"Loaded {len(X_test_raw)} test samples for unsupervised domain adaptation.")

# Preprocessing helpers
def apply_snv(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)

def get_2ch_features(X_raw):
    feats = []
    # Channel 0: 1st Derivative + SG(15) + SNV
    x0 = savgol_filter(X_raw, window_length=15, polyorder=2, deriv=1, axis=-1)
    x0 = apply_snv(x0)
    feats.append(x0)
    
    # Channel 1: 2nd Derivative + SG(15) + SNV
    x1 = savgol_filter(X_raw, window_length=15, polyorder=2, deriv=2, axis=-1)
    x1 = apply_snv(x1)
    feats.append(x1)
    
    return np.stack(feats, axis=1) # (N, 2, 1555)

X_prep = get_2ch_features(X_raw)
y_log = np.log1p(y)

X_test_prep = get_2ch_features(X_test_raw)

# Mixup function for training data
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

class WoodDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None
            
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]

class UnlabeledDataset(Dataset):
    def __init__(self, x):
        self.x = torch.tensor(x, dtype=torch.float32)
            
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx]

# 2. Define Domain-Adaptive Autoencoder Ahoformer
class DAAWoodAhoformer(nn.Module):
    def __init__(self, base_encoder, out_features=1555, out_channels=2):
        super().__init__()
        self.embedding_layer = base_encoder.embedding_layer
        self.pos_encoder = base_encoder.pos_encoder
        self.encoder = base_encoder.encoder
        self.regressor = base_encoder.regressor
        
        # Decoder to reconstruct prep features
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model * 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, out_features * out_channels)
        )
        self.out_features = out_features
        self.out_channels = out_channels
        
    def forward(self, x):
        # x: shape (Batch, 2, 1555)
        out = self.embedding_layer(x) # (Batch, d_model, L_seq)
        out = out.transpose(1, 2)     # (Batch, L_seq, d_model)
        out = self.pos_encoder(out)
        out = self.encoder(out)
        mean_output = out.mean(dim=1)  # (Batch, d_model)
        
        # Moisture prediction
        preds_moisture = self.regressor(mean_output)
        
        # Reconstruction of input features (N, 2, 1555)
        recon = self.decoder(mean_output)
        recon = recon.view(-1, self.out_channels, self.out_features)
        
        return preds_moisture, recon

# --- PHASE 1: Find Optimal Epoch Count using 5-Fold GroupKFold ---
print("\n--- PHASE 1: Running GroupKFold to find optimal epoch count ---")
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_prep, y_log, groups=species_ids))

max_epochs = 100
batch_size = 32
learning_rate = 0.0003
weight_decay = 0.001
mixup_prob = 0.8
alpha_mixup = 0.4
gamma_recon = 1.0 # Reconstruction loss weight

fold_epoch_losses = np.zeros((5, max_epochs))

for fold in range(5):
    print(f"Evaluating Fold {fold+1}/5...")
    train_idx, val_idx = folds[fold]
    
    train_dataset = WoodDataset(X_prep[train_idx], y_log[train_idx])
    val_dataset = WoodDataset(X_prep[val_idx], y_log[val_idx])
    test_dataset = UnlabeledDataset(X_test_prep)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    base_model = AhoformerSpectralEncoder(num_layers=2).to(device)
    base_model.embedding_layer = nn.Conv1d(
        in_channels=2, out_channels=config.d_model, kernel_size=16, stride=12
    ).to(device)
    
    model = DAAWoodAhoformer(base_model).to(device)
    
    criterion_moisture = nn.HuberLoss(delta=0.1)
    criterion_recon = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    
    for epoch in range(max_epochs):
        model.train()
        test_loader_iter = iter(test_loader)
        
        for bx_train, by_train in train_loader:
            bx_train, by_train = bx_train.to(device), by_train.to(device)
            
            try:
                bx_test = next(test_loader_iter).to(device)
            except StopIteration:
                test_loader_iter = iter(test_loader)
                bx_test = next(test_loader_iter).to(device)
            
            # Apply mixup on training data
            if np.random.rand() < mixup_prob:
                bx_train_mix, by_train_mix = mixup_data(bx_train, by_train, alpha=alpha_mixup)
                preds_m, recon_train = model(bx_train_mix)
                loss_m = criterion_moisture(preds_m, by_train_mix)
                loss_recon_train = criterion_recon(recon_train, bx_train_mix)
            else:
                preds_m, recon_train = model(bx_train)
                loss_m = criterion_moisture(preds_m, by_train)
                loss_recon_train = criterion_recon(recon_train, bx_train)
                
            # Unsupervised test data reconstruction
            _, recon_test = model(bx_test)
            loss_recon_test = criterion_recon(recon_test, bx_test)
            
            loss = loss_m + gamma_recon * (loss_recon_train + loss_recon_test)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # Evaluate on validation fold
        model.eval()
        val_preds_log_list = []
        with torch.no_grad():
            for batch_data in val_loader:
                bx_val = batch_data[0].to(device)
                preds_m, _ = model(bx_val)
                val_preds_log_list.append(preds_m.cpu().numpy())
        val_preds_log = np.concatenate(val_preds_log_list, axis=0).squeeze()
        
        val_loss_log = np.mean((val_preds_log - y_log[val_idx]) ** 2)
        fold_epoch_losses[fold, epoch] = val_loss_log

# Find optimal epoch
mean_epoch_losses = np.mean(fold_epoch_losses, axis=0)
optimal_epoch = int(np.argmin(mean_epoch_losses)) + 1
print(f"\n==========================================")
print(f"Optimal Epoch for unseen species: {optimal_epoch} (Val Log MSE: {mean_epoch_losses[optimal_epoch-1]:.4f})")
print(f"==========================================")


# --- PHASE 2: Train a Single Model on 100% of Training Data ---
print(f"\n--- PHASE 2: Training a SINGLE unified model on 100% of train data for {optimal_epoch} epochs ---")
set_seed(42)

full_train_dataset = WoodDataset(X_prep, y_log)
test_dataset = UnlabeledDataset(X_test_prep)

full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

base_model = AhoformerSpectralEncoder(num_layers=2).to(device)
base_model.embedding_layer = nn.Conv1d(
    in_channels=2, out_channels=config.d_model, kernel_size=16, stride=12
).to(device)

final_model = DAAWoodAhoformer(base_model).to(device)

criterion_moisture = nn.HuberLoss(delta=0.1)
criterion_recon = nn.MSELoss()

optimizer = optim.AdamW(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimal_epoch, eta_min=1e-6)

for epoch in range(optimal_epoch):
    final_model.train()
    test_loader_iter = iter(test_loader)
    
    train_loss_m = 0.0
    train_loss_rec_tr = 0.0
    train_loss_rec_te = 0.0
    total_samples = 0
    
    for bx_train, by_train in full_train_loader:
        bx_train, by_train = bx_train.to(device), by_train.to(device)
        
        try:
            bx_test = next(test_loader_iter).to(device)
        except StopIteration:
            test_loader_iter = iter(test_loader)
            bx_test = next(test_loader_iter).to(device)
            
        if np.random.rand() < mixup_prob:
            bx_train_mix, by_train_mix = mixup_data(bx_train, by_train, alpha=alpha_mixup)
            preds_m, recon_train = final_model(bx_train_mix)
            loss_m = criterion_moisture(preds_m, by_train_mix)
            loss_recon_train = criterion_recon(recon_train, bx_train_mix)
        else:
            preds_m, recon_train = final_model(bx_train)
            loss_m = criterion_moisture(preds_m, by_train)
            loss_recon_train = criterion_recon(recon_train, bx_train)
            
        _, recon_test = final_model(bx_test)
        loss_recon_test = criterion_recon(recon_test, bx_test)
        
        loss = loss_m + gamma_recon * (loss_recon_train + loss_recon_test)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_size_actual = bx_train.size(0)
        train_loss_m += loss_m.item() * batch_size_actual
        train_loss_rec_tr += loss_recon_train.item() * batch_size_actual
        train_loss_rec_te += loss_recon_test.item() * batch_size_actual
        total_samples += batch_size_actual
        
    scheduler.step()
    
    train_loss_m /= total_samples
    train_loss_rec_tr /= total_samples
    train_loss_rec_te /= total_samples
    
    if (epoch + 1) % 10 == 0 or epoch == optimal_epoch - 1:
        print(f"Epoch {epoch+1:2d}/{optimal_epoch} | Moisture Loss: {train_loss_m:.4f} | Recon Train: {train_loss_rec_tr:.4f} | Recon Test: {train_loss_rec_te:.4f}")

# --- PHASE 3: Inference on test.csv ---
print("\n--- PHASE 3: Running inference on test.csv using the single model ---")
test_df = pd.read_csv("test.csv", encoding="cp932")
test_sample_numbers = test_df.iloc[:, 0].values
X_test_raw = test_df.iloc[:, 3:].values
X_test_prep = get_2ch_features(X_test_raw)

test_dataset = WoodDataset(X_test_prep)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

final_model.eval()
test_preds_log_list = []
with torch.no_grad():
    for batch_data in test_loader:
        bx = batch_data.to(device) if isinstance(batch_data, torch.Tensor) else batch_data[0].to(device)
        preds_m, _ = final_model(bx)
        test_preds_log_list.append(preds_m.cpu().numpy())

test_preds_log = np.concatenate(test_preds_log_list, axis=0).squeeze()
test_preds_orig = np.expm1(test_preds_log)
test_preds_orig = np.clip(test_preds_orig, 0.01, None)

pd.DataFrame({
    "sample_number": test_sample_numbers,
    "moisture_content": test_preds_orig
}).to_csv("submission_daa.csv", index=False, header=False)

print("Single DAA Model predictions saved to submission_daa.csv")
