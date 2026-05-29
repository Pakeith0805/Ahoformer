import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

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


def apply_snv(x):
    """
    Apply Standard Normal Variate (SNV) preprocessing to NIR spectra.
    SNV centers and scales each individual spectrum to have mean 0 and standard deviation 1.
    """
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)


def load_train_data(file_path="train.csv", use_snv=True):
    """
    Load training data, extract features and targets, and apply optional SNV normalization.
    """
    df = pd.read_csv(file_path, encoding="cp932")
    
    # Target variable '含水率' (moisture content) is the 4th column (index 3)
    # The columns: 
    # Col 0: sample number
    # Col 1: species number
    # Col 2: 樹種 (species name)
    # Col 3: 含水率 (moisture content)
    # Col 4 to end: wavelengths (1555 features)
    
    y = df.iloc[:, 3].values
    X = df.iloc[:, 4:].values
    
    if use_snv:
        X = apply_snv(X)
        
    return X, y, df["sample number"].values, df["species number"].values


def load_test_data(file_path="test.csv", use_snv=True):
    """
    Load test data, extract features, and apply optional SNV normalization.
    """
    df = pd.read_csv(file_path, encoding="cp932")
    
    # Test dataset does NOT contain '含水率' column (it has 1558 columns total)
    # Col 0: sample number
    # Col 1: species number
    # Col 2: 樹種 (species name)
    # Col 3 to end: wavelengths (1555 features)
    
    sample_numbers = df.iloc[:, 0].values
    X = df.iloc[:, 3:].values
    
    if use_snv:
        X = apply_snv(X)
        
    return X, sample_numbers, df["species number"].values