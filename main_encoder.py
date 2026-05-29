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

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load the training data
print("Loading train.csv...")
X, y, sample_ids, species_ids = dataset.load_train_data("train.csv", use_snv=True)
num_samples = X.shape[0]
num_features = X.shape[1]
print(f"Loaded {num_samples} samples with {num_features} features.")

# K-Fold split generator (pure numpy, 100% robust)
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

# Hyperparameters
K = 5
epochs = 100
batch_size = 32
learning_rate = 0.001

print(f"Starting {K}-Fold Cross-Validation...")
folds = get_kfold_indices(num_samples, k=K, seed=42)
fold_val_rmses = []
fold_models = []

for fold in range(K):
    print(f"\n--- Fold {fold+1}/{K} ---")
    train_idx, val_idx = folds[fold]
    
    # Create datasets
    train_dataset = dataset.WoodSpectralDataset(X[train_idx], y[train_idx])
    val_dataset = dataset.WoodSpectralDataset(X[val_idx], y[val_idx])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = AhoformerSpectralEncoder().to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
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
        val_mse = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_mse += loss.item() * batch_x.size(0)
        
        val_mse /= len(val_idx)
        val_rmse = np.sqrt(val_mse)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_weights_path)
            
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Best Val RMSE: {best_val_rmse:.4f}")
            
    print(f"Fold {fold+1} Best Val RMSE: {best_val_rmse:.4f}")
    fold_val_rmses.append(best_val_rmse)

# Calculate estimation accuracy (overall cross-validation accuracy)
mean_rmse = np.mean(fold_val_rmses)
std_rmse = np.std(fold_val_rmses)
print(f"\n==========================================")
print(f"Cross-Validation Summary:")
for f, rmse in enumerate(fold_val_rmses):
    print(f"  Fold {f+1}: RMSE = {rmse:.4f}")
print(f"Overall Estimation RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"==========================================")


# ==========================================
# === Inference on test.csv
# ==========================================
print("\nLoading test.csv...")
X_test, test_sample_numbers, test_species_ids = dataset.load_test_data("test.csv", use_snv=True)
test_dataset = dataset.WoodSpectralDataset(X_test)
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

# Average the predictions across all folds
avg_predictions = np.mean(predictions_all, axis=0)

# Create submission dataframe matching sample_submit.csv format (no header, sample ID, prediction)
submission_df = pd.DataFrame({
    "sample_number": test_sample_numbers,
    "moisture_content": avg_predictions
})

# Save to submission.csv
submission_df.to_csv("submission.csv", index=False, header=False)
print("Saved final ensembled predictions to submission.csv.")
print(f"Predictions stats: Min={avg_predictions.min():.2f}, Max={avg_predictions.max():.2f}, Mean={avg_predictions.mean():.2f}")