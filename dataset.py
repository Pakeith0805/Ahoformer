import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter

class WoodSpectralDataset(Dataset):
    """
    PyTorch Dataset for Wood Near-Infrared (NIR) Spectra.
    Each spectrum is shape (1, num_features) to represent a 1D sequence of length num_features with 1 channel.
    """
    def __init__(self, features, targets=None):
        # features: np.ndarray of shape (N, num_features)
        # targets: np.ndarray of shape (N,)
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1) # Shape: (N, 1, num_features)
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1) # Shape: (N, 1)
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]


def apply_savgol_derivative(x, window_length=15, polyorder=2, deriv=1):
    """
    Apply Savitzky-Golay filtering and calculate the derivative.
    Helps resolve overlapping peaks and removes baseline offsets.
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


def load_train_data(file_path="train.csv", use_savgol=True, use_snv=True):
    """
    Load training data, extract features and targets, and apply optional Savitzky-Golay and SNV normalization.
    """
    df = pd.read_csv(file_path, encoding="cp932")
    
    y = df.iloc[:, 3].values
    X = df.iloc[:, 4:].values
    
    if use_savgol:
        X = apply_savgol_derivative(X)
    if use_snv:
        X = apply_snv(X)
        
    return X, y, df["sample number"].values, df["species number"].values


def load_test_data(file_path="test.csv", use_savgol=True, use_snv=True):
    """
    Load test data, extract features, and apply optional Savitzky-Golay and SNV normalization.
    """
    df = pd.read_csv(file_path, encoding="cp932")
    
    sample_numbers = df.iloc[:, 0].values
    X = df.iloc[:, 3:].values
    
    if use_savgol:
        X = apply_savgol_derivative(X)
    if use_snv:
        X = apply_snv(X)
        
    return X, sample_numbers, df["species number"].values