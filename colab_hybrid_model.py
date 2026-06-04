import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

print("=== Starting Hybrid Calibration Model Pipeline ===")

# 1. Load Data
train_df = pd.read_csv("train.csv", encoding="cp932")
test_df = pd.read_csv("test.csv", encoding="cp932")

y_train = train_df.iloc[:, 3].values  # Original scale moisture content
species_ids = train_df["species number"].values

# Dynamically match spectral columns
spectral_cols = [c for c in train_df.columns[4:] if c in test_df.columns]
print(f"Matched {len(spectral_cols)} wavelength columns.")

X_train_raw = train_df[spectral_cols].values
X_test_raw = test_df[spectral_cols].values
test_sample_numbers = test_df.iloc[:, 0].values

# 2. Preprocessing (1D + 2D Derivatives + SNV)
def apply_snv(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)

print("Applying SG filtering and SNV standardization...")
# Train features
X_train_1d = savgol_filter(X_train_raw, window_length=15, polyorder=2, deriv=1, axis=-1)
X_train_2d = savgol_filter(X_train_raw, window_length=15, polyorder=2, deriv=2, axis=-1)
X_train_1d_snv = apply_snv(X_train_1d)
X_train_2d_snv = apply_snv(X_train_2d)
X_train_concat = np.hstack([X_train_1d_snv, X_train_2d_snv])

# Test features
X_test_1d = savgol_filter(X_test_raw, window_length=15, polyorder=2, deriv=1, axis=-1)
X_test_2d = savgol_filter(X_test_raw, window_length=15, polyorder=2, deriv=2, axis=-1)
X_test_1d_snv = apply_snv(X_test_1d)
X_test_2d_snv = apply_snv(X_test_2d)
X_test_concat = np.hstack([X_test_1d_snv, X_test_2d_snv])

# Limit predictions to physical bounds [0, 300]%
M_MAX = 300.0

# ==========================================
# APPROACH 1: SVR Global (Original Scale)
# ==========================================
print("\n--- Training SVR Global Model ---")
svr_model = SVR(C=100.0, epsilon=0.1)
svr_model.fit(X_train_concat, y_train)

preds_svr = svr_model.predict(X_test_concat)
preds_svr = np.clip(preds_svr, 0.01, M_MAX)

pd.DataFrame({
    "sample_number": test_sample_numbers,
    "moisture_content": preds_svr
}).to_csv("submission_svr_global.csv", index=False, header=False)
print("SVR Global predictions saved to submission_svr_global.csv")


# ==========================================
# APPROACH 2: SWR Local (Similarity-Weighted Regression)
# ==========================================
print("\n--- Training SWR Local Models ---")
unique_species = sorted(np.unique(species_ids))

# Train local models and compute centroids
local_models = {}
centroids = {}

for sp in unique_species:
    mask = (species_ids == sp)
    # Using Ridge for local calibration (highly stable on local data)
    model = Ridge(alpha=10.0)
    model.fit(X_train_concat[mask], y_train[mask])
    local_models[sp] = model
    
    # Species centroid in the concatenated feature space
    centroids[sp] = X_train_concat[mask].mean(axis=0)

print("Running transductive similarity-weighted inference on test set...")
preds_swr = []
temp = 0.05  # Temperature parameter to control weight sharpness (low value favors closest match)

for x_test in X_test_concat:
    similarities = {}
    for sp, centroid in centroids.items():
        # Cosine similarity between test sample and training species centroid
        cos_sim = np.dot(x_test, centroid) / (np.linalg.norm(x_test) * np.linalg.norm(centroid) + 1e-8)
        similarities[sp] = cos_sim
        
    # Convert to Softmax weights
    sim_array = np.array([similarities[sp] for sp in unique_species])
    weights = np.exp(sim_array / temp)
    weights /= np.sum(weights)
    
    # Aggregate local predictions
    pred_weighted = 0.0
    for idx, sp in enumerate(unique_species):
        pred_local = local_models[sp].predict(x_test.reshape(1, -1))[0]
        pred_weighted += weights[idx] * pred_local
        
    preds_swr.append(pred_weighted)

preds_swr = np.array(preds_swr)
preds_swr = np.clip(preds_swr, 0.01, M_MAX)

pd.DataFrame({
    "sample_number": test_sample_numbers,
    "moisture_content": preds_swr
}).to_csv("submission_swr_local.csv", index=False, header=False)
print("SWR Local predictions saved to submission_swr_local.csv")


# ==========================================
# APPROACH 3: Ensemble Blend (50% SVR + 50% SWR)
# ==========================================
print("\n--- Blending Predictions (50% SVR + 50% SWR) ---")
preds_blend = 0.5 * preds_svr + 0.5 * preds_swr

pd.DataFrame({
    "sample_number": test_sample_numbers,
    "moisture_content": preds_blend
}).to_csv("submission_blend.csv", index=False, header=False)
print("Ensemble Blend predictions saved to submission_blend.csv")

print("\nAll pipeline tasks finished successfully!")
