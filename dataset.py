import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter

class WoodSpectralDataset(Dataset):
    """
    PyTorch Dataset for Wood Near-Infrared (NIR) Spectra.
    Each spectrum is shape (C, num_features) to represent a 2-channel 1D sequence:
      - Channel 0: 1st Derivative (SG smoothed, SNV normalized)
      - Channel 1: 2nd Derivative (SG smoothed, SNV normalized)
    """
    def __init__(self, features, targets=None, augment=False):
        # features: np.ndarray of shape (N, C, num_features)
        # targets: np.ndarray of shape (N,)
        self.features = torch.tensor(features, dtype=torch.float32) # Shape: (N, C, num_features)
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1) # Shape: (N, 1)
        else:
            self.targets = None
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx] # Shape: (C, num_features)
        
        if self.augment:
            # Apply training-time data augmentation on the cloned tensor
            x = x.clone()
            
            # 1. Add random Gaussian noise to all channels (simulates sensor noise)
            noise = torch.randn_like(x) * 0.015
            x = x + noise
            
            # 2. Apply random scale variations per channel (simulates light scattering differences)
            # Since derivatives have a zero baseline, we only scale the signal amplitude
            for c in range(x.size(0)):
                scale = 1.0 + torch.randn(1) * 0.02
                x[c] = x[c] * scale
                
        if self.targets is not None:
            return x, self.targets[idx]
        return x


def apply_savgol_derivative(x, window_length=21, polyorder=2, deriv=1):
    """
    Apply Savitzky-Golay filtering and calculate the derivative.
    Helps resolve overlapping peaks and removes baseline offsets.
    Window length 21 is selected to smooth out high-frequency noise in the 2nd derivative.
    """
    return savgol_filter(x, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=-1)


def apply_snv(x):
    """
    Apply Standard Normal Variate (SNV) preprocessing to NIR spectra.
    SNV centers and scales each individual spectrum to have mean 0 and standard deviation 1.
    """
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)


# Global reference spectrum for Multiplicative Scatter Correction (MSC)
_reference_spectrum = None

def apply_msc(X, reference_spectrum=None):
    """
    Apply Multiplicative Scatter Correction (MSC) to NIR spectra.
    Each spectrum is regressed against the reference_spectrum.
    """
    global _reference_spectrum
    if reference_spectrum is None:
        if _reference_spectrum is None:
            # Fallback to mean of input if reference is not set yet
            _reference_spectrum = X.mean(axis=0)
        reference_spectrum = _reference_spectrum
        
    X_msc = np.zeros_like(X)
    for i in range(len(X)):
        # Linear regression: X[i] = slope * reference + intercept
        fit = np.polyfit(reference_spectrum, X[i], 1)
        slope, intercept = fit[0], fit[1]
        X_msc[i] = (X[i] - intercept) / (slope + 1e-8)
    return X_msc


def get_wavenumber_mask(column_names):
    """
    Generate boolean mask for water absorption bands from column names.
    Water absorption bands:
      - 7300-6300 cm-1 (includes O-H stretching overtone and hydrogen bonding)
      - 5300-4700 cm-1 (expanded combination bands)
    """
    wavenumbers = np.array([float(col) for col in column_names])
    mask1 = (wavenumbers >= 6300) & (wavenumbers <= 7300)
    mask2 = (wavenumbers >= 4700) & (wavenumbers <= 5300)
    return mask1 | mask2


def get_multichannel_features(X, use_savgol=True, use_snv=True, use_msc=False, wavenumber_mask=None, num_channels=1):
    """
    Extract multi-channel representation of NIR spectra:
      - Channel 0: 1st Derivative (window=15, poly=2, deriv=1, SNV normalized)
      - Channel 1 (optional): 2nd Derivative (window=15, poly=2, deriv=2, SNV normalized)
    Returns: np.ndarray of shape (N, C, num_features)
    """
    feats = []
    
    # Channel 0: 1st derivative
    x1 = apply_savgol_derivative(X, window_length=15, polyorder=2, deriv=1)
    if use_snv:
        x1 = apply_snv(x1)
    if wavenumber_mask is not None:
        x1 = x1[:, wavenumber_mask]
    feats.append(x1)
    
    # Channel 1: 2nd derivative
    if num_channels == 2:
        x2 = apply_savgol_derivative(X, window_length=15, polyorder=2, deriv=2)
        if use_snv:
            x2 = apply_snv(x2)
        if wavenumber_mask is not None:
            x2 = x2[:, wavenumber_mask]
        feats.append(x2)
        
    return np.stack(feats, axis=1)


def load_train_data(file_path="train.csv", use_savgol=True, use_snv=True, use_msc=False, num_channels=1):
    """
    Load training data, extract multi-channel features and targets.
    """
    global _reference_spectrum
    df = pd.read_csv(file_path, encoding="cp932")
    
    y = df.iloc[:, 3].values
    X = df.iloc[:, 4:].values
    
    # Compute and save the training reference spectrum for MSC (if ever needed)
    _reference_spectrum = X.mean(axis=0)
    
    X_multi = get_multichannel_features(
        X, use_savgol=use_savgol, use_snv=use_snv, use_msc=use_msc, wavenumber_mask=None, num_channels=num_channels
    )
        
    return X_multi, y, df["sample number"].values, df["species number"].values


def load_test_data(file_path="test.csv", use_savgol=True, use_snv=True, use_msc=False, num_channels=1):
    """
    Load test data, extract multi-channel features.
    """
    df = pd.read_csv(file_path, encoding="cp932")
    
    sample_numbers = df.iloc[:, 0].values
    X = df.iloc[:, 3:].values
    
    X_multi = get_multichannel_features(
        X, use_savgol=use_savgol, use_snv=use_snv, use_msc=use_msc, wavenumber_mask=None, num_channels=num_channels
    )
        
    return X_multi, sample_numbers, df["species number"].values