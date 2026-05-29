import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.cross_decomposition import PLSRegression
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

# Custom RMSE Loss with clamped log predictions to prevent gradient explosion
class CustomRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds_log, targets_orig):
        # Clamp log predictions between -0.01 and 5.7 (approx. 300% moisture content)
        # This mathematically prevents exponential gradient explosion from torch.expm1
        preds_log_clamped = torch.clamp(preds_log, min=-0.01, max=5.7)
        preds_orig = torch.expm1(preds_log_clamped)
        # Calculate RMSE directly against original scale targets
        return torch.sqrt(torch.mean((preds_orig - targets_orig) ** 2) + 1e-8)

# 1. Load the training data (applies Savitzky-Golay 1st & 2nd derivative + SNV scaling -> 2 channels)
print("Loading train.csv...")
X, y, sample_ids, species_ids = dataset.load_train_data("train.csv", use_savgol=True, use_snv=True)
num_samples = X.shape[0]
num_features = X.shape[2]
print(f"Loaded {num_samples} samples with {num_features} wavelengths (2 channels).")

# Create log-transformed targets for Stage 1 training
y_log = np.log1p(y)

# Hyperparameters
K = 5
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

print(f"Starting {K}-Fold Group K-Fold Cross-Validation (grouped by species number)...")
gkf = GroupKFold(n_splits=K)
fold_val_rmses = []
fold_pls_models = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=species_ids)):
    print(f"\n--- Fold {fold+1}/{K} ---")
    train_species = np.unique(species_ids[train_idx])
    val_species = np.unique(species_ids[val_idx])
    print(f"  Training on species: {train_species}")
    print(f"  Validating on species (unseen in training): {val_species}")
    
    # ----------------------------------------------------------------
    # === Stage 1: Pretrain Transformer Feature Extractor ===
    # ----------------------------------------------------------------
    print(f"  [Stage 1] Pretraining Transformer on log-scale targets with Custom RMSE loss...")
    
    # Feed y_log for dataset targets (to match log predictions of Transformer)
    train_dataset = dataset.WoodSpectralDataset(X[train_idx], y_log[train_idx], augment=True)
    val_dataset = dataset.WoodSpectralDataset(X[val_idx], y_log[val_idx], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = AhoformerSpectralEncoder().to(device)
    
    # Custom RMSE loss that evaluates RMSE on original scale directly
    criterion = CustomRMSELoss()
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
            # Reconstruct original scale targets for loss computation
            batch_y_orig = torch.expm1(batch_y_log)
            
            preds_log = model(batch_x)
            loss = criterion(preds_log, batch_y_orig)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        scheduler.step()
        train_loss /= len(train_idx)
        
        # Validation for Stage 1
        model.eval()
        val_preds_log_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                preds_log = model(batch_x)
                val_preds_log_list.append(preds_log.cpu().numpy())
        
        val_preds_log = np.concatenate(val_preds_log_list, axis=0).squeeze()
        # Apply clamp to validation predictions as well for consistency
        val_preds_log = np.clip(val_preds_log, a_min=-0.01, a_max=5.7)
        val_preds_orig = np.expm1(val_preds_log)
        val_y_orig = y[val_idx]
        
        val_rmse = np.sqrt(np.mean((val_preds_orig - val_y_orig) ** 2))
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_weights_path)
            
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:3d}/{epochs} | LR: {current_lr:.6f} | Train Loss (RMSE): {train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Best Val RMSE: {best_val_rmse:.4f}")
            
    print(f"  [Stage 1] Pretraining finished. Best Transformer Val RMSE: {best_val_rmse:.4f}")
    
    # ----------------------------------------------------------------
    # === Stage 2: Extract Features & Fit PLS Regressor ===
    # ----------------------------------------------------------------
    print(f"  [Stage 2] Loading best Transformer, extracting features, and fitting PLS...")
    
    # Load pretrained best weights
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()
    
    # Helper to extract 128-dimensional features
    def extract_features(loader):
        features_all = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                feats = model(batch_x, return_features=True)
                features_all.append(feats.cpu().numpy())
        return np.concatenate(features_all, axis=0)
    
    # Features from Stage 1 are derived from non-augmented inputs for stability
    train_dataset_eval = dataset.WoodSpectralDataset(X[train_idx], y[train_idx], augment=False)
    train_loader_eval = DataLoader(train_dataset_eval, batch_size=batch_size, shuffle=False)
    
    X_train_feats = extract_features(train_loader_eval)
    X_val_feats = extract_features(val_loader)
    
    y_train_orig = y[train_idx]
    y_val_orig = y[val_idx]
    
    # Search for the best number of PLS latent components (1 to 15)
    best_pls_rmse = float('inf')
    best_n_components = 2
    best_pls_model = None
    
    for n_comp in range(1, 16):
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_train_feats, y_train_orig)
        
        # Predict validation set
        val_preds_pls = pls.predict(X_val_feats).squeeze()
        # Clip negative predictions
        val_preds_pls = np.clip(val_preds_pls, a_min=0.01, a_max=None)
        
        pls_rmse = np.sqrt(np.mean((val_preds_pls - y_val_orig) ** 2))
        if pls_rmse < best_pls_rmse:
            best_pls_rmse = pls_rmse
            best_n_components = n_comp
            best_pls_model = pls
            
    print(f"  [Stage 2] Fold {fold+1} Optimal PLS Components: {best_n_components} | Validation RMSE: {best_pls_rmse:.4f}")
    
    fold_pls_models.append(best_pls_model)
    fold_val_rmses.append(best_pls_rmse)

# Summary of hybrid pipeline
mean_rmse = np.mean(fold_val_rmses)
std_rmse = np.std(fold_val_rmses)
print(f"\n==========================================")
print(f"Hybrid Transformer-PLS Validation Summary:")
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

print("Running inference with ensembled fold models (Transformer + PLS)...")
predictions_all = []

for fold in range(K):
    # Load pretrained Transformer
    model = AhoformerSpectralEncoder().to(device)
    best_weights_path = f"best_model_fold_{fold}.pth"
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()
    
    # Extract 128-dimensional features for test set
    test_features_list = []
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            feats = model(batch_x, return_features=True)
            test_features_list.append(feats.cpu().numpy())
    test_features = np.concatenate(test_features_list, axis=0)
    
    # Predict using the fold's trained PLS regression
    pls_model = fold_pls_models[fold]
    fold_preds = pls_model.predict(test_features).squeeze()
    predictions_all.append(fold_preds)
    
    # Cleanup weight file
    try:
        os.remove(best_weights_path)
    except OSError:
        pass

# Average the predictions (ensembling) using Median to suppress outliers
avg_predictions = np.median(predictions_all, axis=0)

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