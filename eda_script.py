import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16

print("Starting complete data analysis...")

# 1. Load Data
train_df = pd.read_csv("train.csv", encoding="cp932")
test_df = pd.read_csv("test.csv", encoding="cp932")

# Extract columns
# Train columns: sample_number, species, species number, moisture_content, wavelengths...
# Test columns: sample_number, species, species number, wavelengths... (Wait, test does not have species number at test time, but test.csv might have columns? Let's check headers)
# Let's inspect test headers programmatically inside the script.
train_moisture = train_df.iloc[:, 3].values
train_spectra_raw = train_df.iloc[:, 4:].values
train_species_ids = train_df["species number"].values

# Dynamically find spectral columns for test
# Test has 3 columns before spectra: sample_number, species, species number (usually filled with NaN or omitted)
# Let's align wavelength columns by finding columns that match between train and test
spectral_cols = [c for c in train_df.columns[4:] if c in test_df.columns]
print(f"Matched {len(spectral_cols)} wavelength columns between train and test.")

X_train_raw = train_df[spectral_cols].values
X_test_raw = test_df[spectral_cols].values

# Extract wavelengths as floats if possible
wavelengths = np.array([float(c) for c in spectral_cols])

# 2. Write summary to text file
with open("eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("=== WOOD MOISTURE DATASET EDA SUMMARY ===\n\n")
    f.write(f"Number of training samples: {len(train_df)}\n")
    f.write(f"Number of test samples: {len(test_df)}\n")
    f.write(f"Number of spectral bands: {len(spectral_cols)}\n")
    f.write(f"Wavelength range: {wavelengths[0]} nm to {wavelengths[-1]} nm\n\n")
    
    # Moisture content statistics
    f.write("--- Moisture Content Statistics (Train) ---\n")
    moisture_series = pd.Series(train_moisture)
    f.write(moisture_series.describe().to_string())
    f.write("\n\n")
    
    # Species statistics in train
    f.write("--- Species Distribution and Moisture Stats ---\n")
    moisture_col = train_df.columns[3]
    species_summary = train_df.groupby("species number")[moisture_col].agg(["count", "mean", "std", "min", "max"])
    f.write(species_summary.to_string())
    f.write("\n\n")

print("Created eda_summary.txt with basic statistics.")

# Preprocessing helpers
def apply_snv(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)

# 3. Spectral Profiling (Raw vs. Derivatives)
# Compute SG derivatives
X_train_1d = savgol_filter(X_train_raw, window_length=15, polyorder=2, deriv=1, axis=-1)
X_train_2d = savgol_filter(X_train_raw, window_length=15, polyorder=2, deriv=2, axis=-1)
X_test_1d = savgol_filter(X_test_raw, window_length=15, polyorder=2, deriv=1, axis=-1)
X_test_2d = savgol_filter(X_test_raw, window_length=15, polyorder=2, deriv=2, axis=-1)

# Apply SNV
X_train_snv = apply_snv(X_train_raw)
X_test_snv = apply_snv(X_test_raw)
X_train_1d_snv = apply_snv(X_train_1d)
X_test_1d_snv = apply_snv(X_test_1d)
X_train_2d_snv = apply_snv(X_train_2d)
X_test_2d_snv = apply_snv(X_test_2d)

# Plot spectral profiles (Raw, SNV, 2nd Derivative)
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# Raw spectra average by species
unique_species = sorted(np.unique(train_species_ids))
colors = sns.color_palette("tab20", len(unique_species))

for i, sp in enumerate(unique_species):
    idx = (train_species_ids == sp)
    axes[0].plot(wavelengths, X_train_raw[idx].mean(axis=0), label=f"Sp {sp}", color=colors[i], alpha=0.8)
axes[0].plot(wavelengths, X_test_raw.mean(axis=0), label="TEST Mean", color="black", linewidth=2.5, linestyle="--")
axes[0].set_title("Raw Spectra: Species Mean vs Test Mean")
axes[0].set_ylabel("Absorbance")
axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)

# SNV spectra average
for i, sp in enumerate(unique_species):
    idx = (train_species_ids == sp)
    axes[1].plot(wavelengths, X_train_snv[idx].mean(axis=0), color=colors[i], alpha=0.8)
axes[1].plot(wavelengths, X_test_snv.mean(axis=0), color="black", linewidth=2.5, linestyle="--")
axes[1].set_title("SNV Standardized Spectra")
axes[1].set_ylabel("Standardized Absorbance")

# 2nd Derivative (SG 15) average
for i, sp in enumerate(unique_species):
    idx = (train_species_ids == sp)
    axes[2].plot(wavelengths, X_train_2d[idx].mean(axis=0), color=colors[i], alpha=0.8)
axes[2].plot(wavelengths, X_test_2d.mean(axis=0), color="black", linewidth=2.5, linestyle="--")
axes[2].set_title("2nd Derivative (SG Window 15)")
axes[2].set_ylabel("2nd Deriv Value")
axes[2].set_xlabel("Wavelength (nm)")

plt.tight_layout()
plt.savefig("eda_spectral_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eda_spectral_profiles.png")

# 4. Correlation Analysis: Moisture vs Wavelengths
# Calculate Pearson correlation coefficient for raw, SNV, and 2nd derivative spectra
corr_raw = np.array([np.corrcoef(X_train_raw[:, w], train_moisture)[0, 1] for w in range(len(wavelengths))])
corr_snv = np.array([np.corrcoef(X_train_snv[:, w], train_moisture)[0, 1] for w in range(len(wavelengths))])
corr_2d = np.array([np.corrcoef(X_train_2d[:, w], train_moisture)[0, 1] for w in range(len(wavelengths))])

plt.figure(figsize=(12, 6))
plt.plot(wavelengths, corr_raw, label="Raw Spectra Correlation", alpha=0.8)
plt.plot(wavelengths, corr_snv, label="SNV Spectra Correlation", alpha=0.8)
plt.plot(wavelengths, corr_2d, label="2nd Derivative Correlation", alpha=0.8)
plt.axhline(0, color="grey", linestyle="--")
plt.title("Correlation between Spectral Features and Moisture Content")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Pearson Correlation (r)")
plt.legend()
plt.savefig("eda_moisture_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eda_moisture_correlations.png")

# Write top correlating wavelengths to summary
top_raw_idx = np.argsort(np.abs(corr_raw))[::-1][:5]
top_snv_idx = np.argsort(np.abs(corr_snv))[::-1][:5]
top_2d_idx = np.argsort(np.abs(corr_2d))[::-1][:5]

with open("eda_summary.txt", "a", encoding="utf-8") as f:
    f.write("--- Top 5 Correlating Wavelengths with Moisture ---\n")
    f.write("Raw Spectra:\n")
    for idx in top_raw_idx:
        f.write(f"  {wavelengths[idx]:.1f} nm: r = {corr_raw[idx]:.4f}\n")
    f.write("SNV Spectra:\n")
    for idx in top_snv_idx:
        f.write(f"  {wavelengths[idx]:.1f} nm: r = {corr_snv[idx]:.4f}\n")
    f.write("2nd Derivative:\n")
    for idx in top_2d_idx:
        f.write(f"  {wavelengths[idx]:.1f} nm: r = {corr_2d[idx]:.4f}\n")
    f.write("\n")

# 5. Dimensionality Reduction & Domain Shift (PCA)
# Run PCA on SNV-preprocessed combined dataset to analyze domain shift
scaler = StandardScaler()
X_combined = np.vstack([X_train_snv, X_test_snv])
X_combined_scaled = scaler.fit_transform(X_combined)

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_combined_scaled)

X_train_pca = X_pca[:len(X_train_raw)]
X_test_pca = X_pca[len(X_train_raw):]

# Write PCA variance explained
with open("eda_summary.txt", "a", encoding="utf-8") as f:
    f.write("--- PCA Explained Variance (Combined SNV Spectra) ---\n")
    for comp_idx, var in enumerate(pca.explained_variance_ratio_):
        f.write(f"  PC{comp_idx+1}: {var*100:.2f}%\n")
    f.write(f"  Total explained variance (5 components): {np.sum(pca.explained_variance_ratio_)*100:.2f}%\n\n")

# Plot PCA
plt.figure(figsize=(10, 8))
# Plot train samples colored by species
for i, sp in enumerate(unique_species):
    idx = (train_species_ids == sp)
    plt.scatter(X_train_pca[idx, 0], X_train_pca[idx, 1], label=f"Sp {sp}", color=colors[i], alpha=0.6, edgecolors="none", s=25)
# Plot test samples
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], label="TEST (Unseen)", color="black", marker="x", alpha=0.8, s=35)
plt.title("PCA Space (PC1 vs PC2) demonstrating Domain Shift")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
plt.savefig("eda_pca_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved eda_pca_distribution.png")

# 6. Quantify Test-to-Train Species Similarity (Cosine Similarity in PCA Space)
# Calculate mean centroid for each species in train PCA space (using top 5 PCs)
centroids = {}
for sp in unique_species:
    idx = (train_species_ids == sp)
    centroids[sp] = X_train_pca[idx, :5].mean(axis=0)

# Calculate similarity of each test sample to each train species centroid
similarity_records = []
for test_idx in range(len(X_test_pca)):
    test_vec = X_test_pca[test_idx, :5]
    record = {"test_sample_index": test_idx}
    for sp in unique_species:
        centroid = centroids[sp]
        # Cosine similarity
        cos_sim = np.dot(test_vec, centroid) / (np.linalg.norm(test_vec) * np.linalg.norm(centroid) + 1e-8)
        record[f"sim_to_sp_{sp}"] = cos_sim
    similarity_records.append(record)

sim_df = pd.DataFrame(similarity_records)
sim_df.to_csv("eda_species_similarity.csv", index=False)
print("Saved eda_species_similarity.csv")

# Compute overall average similarity of test set to each train species
avg_similarities = sim_df.mean(axis=0).drop("test_sample_index")
with open("eda_summary.txt", "a", encoding="utf-8") as f:
    f.write("--- Average Cosine Similarity of Test Set to Train Species (PCA Space) ---\n")
    for k, v in avg_similarities.items():
        f.write(f"  {k}: similarity = {v:.4f}\n")
    f.write("\n")

print("EDA script execution completed successfully!")
