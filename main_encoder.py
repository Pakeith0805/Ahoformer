import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
import dataset
from models import AhoformerSpectralEncoder


# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
torch.set_num_threads(4)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# K-Fold split generator (pure numpy, 100% robust Random K-Fold)
def get_kfold_indices(n, k=5, seed=42):
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

# 1. Load the training data (applies Savitzky-Golay 1st derivative + SNV scaling -> 1 channel)
print("Loading train.csv...")
X, y, sample_ids, species_ids = dataset.load_train_data("train.csv", use_savgol=True, use_snv=True)
num_samples = X.shape[0]
num_features = X.shape[2]
print(f"Loaded {num_samples} samples with {num_features} wavelengths (1 channel).")

# Create log-transformed targets for Stage 1 training
y_log = np.log1p(y)

# Hyperparameters
K = 10
epochs = config.epochs
batch_size = 32
learning_rate = config.lr

# Learning rate scheduler multiplier for Cosine Annealing with Warmup
def get_lr_multiplier(epoch, warmup_epochs=5, total_epochs=60):
    if epoch < warmup_epochs:
        return 0.2 + 0.8 * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

print(f"Starting {K}-Fold Cross-Validation (Random K-Fold)...")
folds = get_kfold_indices(num_samples, k=K, seed=42)
fold_val_rmses = []
fold_val_rmses = []

for fold in range(K):
    print(f"\n--- Fold {fold+1}/{K} ---")
    train_idx, val_idx = folds[fold]
    
    # ----------------------------------------------------------------
    # === Stage 1: Train Transformer Feature Extractor ===
    # ----------------------------------------------------------------
    print(f"  Training Transformer on log-scale targets with MSE loss...")
    
    train_dataset = dataset.WoodSpectralDataset(X[train_idx], y_log[train_idx], augment=True)
    val_dataset = dataset.WoodSpectralDataset(X[val_idx], y_log[val_idx], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = AhoformerSpectralEncoder().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda ep: get_lr_multiplier(ep, warmup_epochs=5, total_epochs=epochs)
    )
    
    best_val_rmse = float('inf')
    best_weights_path = f"best_model_fold_{fold}.pth"
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y_log in train_loader:
            batch_x, batch_y_log = batch_x.to(device), batch_y_log.to(device)
            
            preds_log = model(batch_x)
            loss = criterion(preds_log, batch_y_log)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        scheduler.step()
        train_loss /= len(train_idx)
        
        # Validation
        model.eval()
        val_preds_log_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                preds_log = model(batch_x)
                val_preds_log_list.append(preds_log.cpu().numpy())
        
        val_preds_log = np.concatenate(val_preds_log_list, axis=0).squeeze()
        val_preds_orig = np.expm1(val_preds_log)
        val_y_orig = y[val_idx]
        
        val_rmse = np.sqrt(np.mean((val_preds_orig - val_y_orig) ** 2))
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_weights_path)
            
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:3d}/{epochs} | LR: {current_lr:.6f} | Train Loss (Log MSE): {train_loss:.4f} | Val RMSE (Orig): {val_rmse:.4f} | Best Val RMSE: {best_val_rmse:.4f}")
            
    print(f"  Fold {fold+1} Best Val RMSE: {best_val_rmse:.4f}")
    fold_val_rmses.append(best_val_rmse)

# Summary of pipeline
mean_rmse = np.mean(fold_val_rmses)
std_rmse = np.std(fold_val_rmses)
print(f"\n==========================================")
print(f"Transformer Validation Summary:")
for f, rmse in enumerate(fold_val_rmses):
    print(f"  Fold {f+1}: RMSE = {rmse:.4f}")
print(f"Overall Estimation RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"==========================================")


# ==========================================
# === Inference on test.csv
# ==========================================
print("\nLoading test.csv...")
X_test, test_sample_numbers, test_species_ids = dataset.load_test_data("test.csv", use_savgol=True, use_snv=True)
test_dataset = dataset.WoodSpectralDataset(X_test, augment=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Running inference with ensembled fold models (Transformer End-to-End)...")
predictions_all = []

for fold in range(K):
    # Load pretrained Transformer
    model = AhoformerSpectralEncoder().to(device)
    best_weights_path = f"best_model_fold_{fold}.pth"
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()
    
    # Predict directly with Transformer (log-scale)
    fold_preds_log_list = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            preds_log = model(batch_x)
            fold_preds_log_list.append(preds_log.cpu().numpy())
            
    fold_preds_log = np.concatenate(fold_preds_log_list, axis=0).squeeze()
    predictions_all.append(fold_preds_log)
    
    # Cleanup weight file
    try:
        os.remove(best_weights_path)
    except OSError:
        pass

# Average predictions in log-space (Geometric Mean ensembling) and invert
avg_predictions_log = np.mean(predictions_all, axis=0)
avg_predictions = np.expm1(avg_predictions_log)

# Ensure no negative predictions (moisture must be > 0)
avg_predictions = np.clip(avg_predictions, a_min=0.01, a_max=None)

# Create submission dataframe matching sample_submit.csv format
submission_df = pd.DataFrame({
    "sample_number": test_sample_numbers,
    "moisture_content": avg_predictions
})

# Save to submission.csv
submission_df.to_csv("submission.csv", index=False, header=False)
print("Saved final ensembled predictions to submission.csv.")
print(f"Predictions stats: Min={avg_predictions.min():.2f}, Max={avg_predictions.max():.2f}, Mean={avg_predictions.mean():.2f}")