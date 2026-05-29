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


def get_multichannel_features(X, use_savgol=True, use_snv=True):
    """
    Extract 2-channel derivative-only representations of NIR spectra:
      - Channel 0: 1st Derivative (window=21, poly=2, deriv=1, SNV normalized)
      - Channel 1: 2nd Derivative (window=21, poly=2, deriv=2, SNV normalized)
    Returns: np.ndarray of shape (N, 2, 1555)
    """
    feats = []
    
    # Channel 0: 1st Derivative
    x1 = apply_savgol_derivative(X, window_length=21, polyorder=2, deriv=1)
    if use_snv:
        x1 = apply_snv(x1)
    feats.append(x1)
    
    # Channel 1: 2nd Derivative
    x2 = apply_savgol_derivative(X, window_length=21, polyorder=2, deriv=2)
    if use_snv:
        x2 = apply_snv(x2)
    feats.append(x2)
        
    return np.stack(feats, axis=1)


def load_train_data(file_path="train.csv", use_savgol=True, use_snv=True):
    """
    Load training data, extract multi-channel features and targets.
    """
    df = pd.read_csv(file_path, encoding="cp932")
    
    y = df.iloc[:, 3].values
    X = df.iloc[:, 4:].values
    
    X_multi = get_multichannel_features(X, use_savgol=use_savgol, use_snv=use_snv)
        
    return X_multi, y, df["sample number"].values, df["species number"].values


def load_test_data(file_path="test.csv", use_savgol=True, use_snv=True):
    """
    Load test data, extract multi-channel features.
    """
    df = pd.read_csv(file_path, encoding="cp932")
    
    sample_numbers = df.iloc[:, 0].values
    X = df.iloc[:, 3:].values
    
    X_multi = get_multichannel_features(X, use_savgol=use_savgol, use_snv=use_snv)
        
    return X_multi, sample_numbers, df["species number"].values