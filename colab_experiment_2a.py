import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import config
import dataset
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

# 1. Load 2-Channel Data
X, y, sample_ids, species_ids = dataset.load_train_data("train.csv", use_savgol=True, use_snv=True, use_msc=False, num_channels=2)
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
weight_decay = 0.001
mixup_prob = 0.8
alpha = 0.4

print(f"Starting Experiment 2A: 2-Channel + Stronger Mixup (10-Fold CV)...")
folds = get_kfold_indices(len(X), k=K, seed=42)
fold_val_rmses = []
predictions_all = []

# Load test data
X_test, test_sample_numbers, test_species_ids = dataset.load_test_data("test.csv", use_savgol=True, use_snv=True, use_msc=False, num_channels=2)
test_dataset = WoodDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for fold in range(K):
    print(f"\n--- Fold {fold+1}/{K} ---")
    train_idx, val_idx = folds[fold]
    
    train_dataset = WoodDataset(X[train_idx], y_log[train_idx])
    val_dataset = WoodDataset(X[val_idx], y_log[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = AhoformerSpectralEncoder().to(device)
    model.embedding_layer = nn.Conv1d(
        in_channels=2, 
        out_channels=config.d_model, 
        kernel_size=16, 
        stride=12
    ).to(device)
        
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_rmse = float('inf')
    best_weights_path = f"best_model_exp2a_fold_{fold}.pth"
    
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
        val_preds_orig = np.expm1(val_preds_log)
        val_preds_orig = np.clip(val_preds_orig, 0.01, None)
        
        val_rmse = np.sqrt(np.mean((val_preds_orig - y[val_idx]) ** 2))
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_weights_path)
            
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1:2d}/{epochs} | Train Loss (Log MSE): {train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Best Val: {best_val_rmse:.4f}")
            
    print(f"  Fold {fold+1} Best Val RMSE: {best_val_rmse:.4f}")
    fold_val_rmses.append(best_val_rmse)
    
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()
    
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
print(f"Experiment 2A (Stronger Mixup) CV Summary:")
print(f"Overall Estimation RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"==========================================")

# Ensemble average
avg_predictions_log = np.mean(predictions_all, axis=0)
avg_predictions = np.expm1(avg_predictions_log)
avg_predictions = np.clip(avg_predictions, 0.01, None)

pd.DataFrame({
    "sample_number": test_sample_numbers,
    "moisture_content": avg_predictions
}).to_csv("submission_exp2a.csv", index=False, header=False)
print("Saved predictions to submission_exp2a.csv")
