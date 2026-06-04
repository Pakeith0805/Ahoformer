import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.signal import savgol_filter

# Override config variables dynamically
import config
config.dropout = 0.10

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

# 1. Load raw data
train_df = pd.read_csv("train.csv", encoding="cp932")
y = train_df.iloc[:, 3].values
X_raw = train_df.iloc[:, 4:].values

# Preprocessing helpers
def apply_snv(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)

def get_2ch_features(X_raw):
    feats = []
    # Channel 0: 1st Derivative + SG(21) + SNV
    x0 = savgol_filter(X_raw, window_length=21, polyorder=2, deriv=1, axis=-1)
    x0 = apply_snv(x0)
    feats.append(x0)
    
    # Channel 1: 2nd Derivative + SG(21) + SNV
    x1 = savgol_filter(X_raw, window_length=21, polyorder=2, deriv=2, axis=-1)
    x1 = apply_snv(x1)
    feats.append(x1)
    
    return np.stack(feats, axis=1) # (N, 2, 1555)

X_prep = get_2ch_features(X_raw)
y_log = np.log1p(y)

# Mixup function
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

class AugmentWoodDataset(Dataset):
    def __init__(self, x, y=None, augment=False):
        self.x = torch.tensor(x, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None
        self.augment = augment
            
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        x = self.x[idx].clone() # (C, num_features)
        
        if self.augment:
            # 1. Add random Gaussian noise to all channels (std=0.01)
            noise = torch.randn_like(x) * 0.01
            x = x + noise
            
            # 2. Channel-wise scaling (std=0.02)
            for c in range(x.size(0)):
                scale = 1.0 + torch.randn(1) * 0.02
                x[c] = x[c] * scale
                
        if self.y is not None:
            return x, self.y[idx]
        return x

def get_kfold_indices(n, k=10, seed=42):
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        val_indices = indices[current:current + fold_size]
        train_indices = np.concatenate([indices[:current], indices[current + fold_size:]])
        folds.append((train_indices, val_indices))
        current += fold_size
    return folds

K = 10
epochs = 100
batch_size = 32
learning_rate = 0.0003
weight_decay = 0.005 # Stiffer weight decay
mixup_prob = 0.8
alpha = 0.4
num_layers = 2

print(f"Starting Experiment 7: Full-Spectrum 2-Channel + Log Checkpoint + Augmentation (10-Fold Random CV)...")
folds = get_kfold_indices(len(X_prep), k=K, seed=42)
fold_val_rmses = []
predictions_all = []

# Load test data
test_df = pd.read_csv("test.csv", encoding="cp932")
test_sample_numbers = test_df.iloc[:, 0].values
X_test_raw = test_df.iloc[:, 3:].values
X_test_prep = get_2ch_features(X_test_raw)

test_dataset = AugmentWoodDataset(X_test_prep, augment=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for fold in range(K):
    print(f"\n--- Fold {fold+1}/{K} ---")
    train_idx, val_idx = folds[fold]
    
    train_dataset = AugmentWoodDataset(X_prep[train_idx], y_log[train_idx], augment=True)
    val_dataset = AugmentWoodDataset(X_prep[val_idx], y_log[val_idx], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = AhoformerSpectralEncoder(num_layers=num_layers).to(device)
    # Configure Conv1d for 2 channels and sequence length 129 (kernel=16, stride=12)
    model.embedding_layer = nn.Conv1d(
        in_channels=2, 
        out_channels=config.d_model, 
        kernel_size=16, 
        stride=12
    ).to(device)
        
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_loss_log = float('inf')
    best_weights_path = f"best_model_exp7_fold_{fold}.pth"
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            
            if np.random.rand() < mixup_prob:
                bx_mixed, by_mixed = mixup_data(bx, by, alpha=alpha)
                loss = criterion(model(bx_mixed), by_mixed)
            else:
                loss = criterion(model(bx), by)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * bx.size(0)
            
        scheduler.step()
        train_loss /= len(train_idx)
        
        # Validation
        model.eval()
        val_preds_log_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                preds_log = model(bx)
                val_preds_log_list.append(preds_log.cpu().numpy())
                
        val_preds_log = np.concatenate(val_preds_log_list, axis=0).squeeze()
        
        # Calculate Validation Loss on Log Scale for Checkpoint saving
        val_loss_log = np.mean((val_preds_log - y_log[val_idx]) ** 2)
        
        # Also compute RMSE on original scale for logging purposes
        val_preds_orig = np.expm1(val_preds_log)
        val_preds_orig = np.clip(val_preds_orig, 0.01, None)
        val_rmse = np.sqrt(np.mean((val_preds_orig - y[val_idx]) ** 2))
        
        if val_loss_log < best_val_loss_log:
            best_val_loss_log = val_loss_log
            torch.save(model.state_dict(), best_weights_path)
            
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1:2d}/{epochs} | Train Loss (Log MSE): {train_loss:.4f} | Val Log MSE: {val_loss_log:.4f} | Val RMSE: {val_rmse:.4f}")
            
    # Evaluate best model on validation set
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()
    val_preds_log_list = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            preds_log = model(bx)
            val_preds_log_list.append(preds_log.cpu().numpy())
    val_preds_log = np.concatenate(val_preds_log_list, axis=0).squeeze()
    val_preds_orig = np.expm1(val_preds_log)
    val_preds_orig = np.clip(val_preds_orig, 0.01, None)
    best_val_rmse = np.sqrt(np.mean((val_preds_orig - y[val_idx]) ** 2))
    
    print(f"  Fold {fold+1} Best Val Log MSE: {best_val_loss_log:.4f} | Best Val RMSE: {best_val_rmse:.4f}")
    fold_val_rmses.append(best_val_rmse)
    
    fold_preds_log_list = []
    with torch.no_grad():
        for bx in test_loader:
            bx = bx.to(device)
            preds_log = model(bx)
            fold_preds_log_list.append(preds_log.cpu().numpy())
            
    fold_preds_log = np.concatenate(fold_preds_log_list, axis=0).squeeze()
    predictions_all.append(fold_preds_log)
    
    try:
        os.remove(best_weights_path)
    except OSError:
        pass

# Summary
mean_rmse = np.mean(fold_val_rmses)
std_rmse = np.std(fold_val_rmses)
print(f"\n==========================================")
print(f"Experiment 7 CV Summary:")
print(f"Overall Estimation RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"==========================================")

# Ensemble average
avg_predictions_log = np.mean(predictions_all, axis=0)
avg_predictions = np.expm1(avg_predictions_log)
avg_predictions = np.clip(avg_predictions, 0.01, None)

pd.DataFrame({
    "sample_number": test_sample_numbers,
    "moisture_content": avg_predictions
}).to_csv("submission_exp7.csv", index=False, header=False)
print("Saved predictions to submission_exp7.csv")
