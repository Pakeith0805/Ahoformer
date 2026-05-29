import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
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

# 1. Load the training data (applies Savitzky-Golay 1st & 2nd derivative + SNV scaling -> 2 channels)
print("Loading train.csv...")
X, y, sample_ids, species_ids = dataset.load_train_data("train.csv", use_savgol=True, use_snv=True)
num_samples = X.shape[0]
num_features = X.shape[2]
print(f"Loaded {num_samples} samples with {num_features} wavelengths (2 channels).")

# Hyperparameters
K = 5
epochs = config.epochs
batch_size = 32
learning_rate = config.lr

# Learning rate scheduler multiplier for Cosine Annealing with Warmup
def get_lr_multiplier(epoch, warmup_epochs=5, total_epochs=60):
    if epoch < warmup_epochs:
        # Linear warmup from 0.2x to 1.0x of base learning rate
        return 0.2 + 0.8 * (epoch / warmup_epochs)
    else:
        # Cosine decay down to 0
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

print(f"Starting {K}-Fold Group K-Fold Cross-Validation (grouped by species number)...")
gkf = GroupKFold(n_splits=K)
fold_val_rmses = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=species_ids)):
    print(f"\n--- Fold {fold+1}/{K} ---")
    train_species = np.unique(species_ids[train_idx])
    val_species = np.unique(species_ids[val_idx])
    print(f"  Training on species: {train_species}")
    print(f"  Validating on species (unseen in training): {val_species}")
    
    # Create datasets (inputs shape: (N, 2, 1555), targets shape: (N, 1))
    # Enable training-time augmentations only for training splits
    train_dataset = dataset.WoodSpectralDataset(X[train_idx], y[train_idx], augment=True)
    val_dataset = dataset.WoodSpectralDataset(X[val_idx], y[val_idx], augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = AhoformerSpectralEncoder().to(device)
    
    # Loss and optimizer (AdamW + Weight Decay)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Cosine Annealing with Warmup
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda ep: get_lr_multiplier(ep, warmup_epochs=5, total_epochs=epochs)
    )
    
    best_val_rmse = float('inf')
    best_weights_path = f"best_model_fold_{fold}.pth"
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        scheduler.step()
        train_loss /= len(train_idx)
        
        # Validation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                preds = model(batch_x)
                val_preds_list.append(preds.cpu().numpy())
        
        val_preds_orig = np.concatenate(val_preds_list, axis=0).squeeze()
        val_y_orig = y[val_idx]
        
        # Calculate RMSE on original scale
        val_rmse = np.sqrt(np.mean((val_preds_orig - val_y_orig) ** 2))
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_weights_path)
            
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Best Val RMSE: {best_val_rmse:.4f}")
            
    print(f"Fold {fold+1} Best Val RMSE: {best_val_rmse:.4f}")
    fold_val_rmses.append(best_val_rmse)

# Calculate estimation accuracy (overall cross-validation accuracy)
mean_rmse = np.mean(fold_val_rmses)
std_rmse = np.std(fold_val_rmses)
print(f"\n==========================================")
print(f"Group K-Fold Cross-Validation Summary:")
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

print("Running inference with ensembled fold models...")
predictions_all = []

for fold in range(K):
    # Initialize model
    model = AhoformerSpectralEncoder().to(device)
    # Load best weights
    best_weights_path = f"best_model_fold_{fold}.pth"
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()
    
    fold_preds = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            fold_preds.append(preds.cpu().numpy())
            
    fold_preds = np.concatenate(fold_preds, axis=0).squeeze()
    predictions_all.append(fold_preds)
    
    # Cleanup weight file
    try:
        os.remove(best_weights_path)
    except OSError:
        pass

# Average the predictions (ensembling)
avg_predictions = np.mean(predictions_all, axis=0)

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