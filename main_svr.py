import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from scipy.signal import savgol_filter

def preprocess(X_raw, window=15, poly=2):
    """
    Savitzky-Golay 1st derivative + Standard Normal Variate (SNV) normalization.
    """
    X_deriv = savgol_filter(X_raw, window_length=window, polyorder=poly, deriv=1, axis=-1)
    mean = X_deriv.mean(axis=1, keepdims=True)
    std = X_deriv.std(axis=1, keepdims=True)
    return (X_deriv - mean) / (std + 1e-8)

def main():
    print("Loading data...")
    train_df = pd.read_csv("train.csv", encoding="cp932")
    y_train = train_df.iloc[:, 3].values
    X_train_raw = train_df.iloc[:, 4:].values
    train_species = train_df["species number"].values

    test_df = pd.read_csv("test.csv", encoding="cp932")
    test_sample_numbers = test_df.iloc[:, 0].values
    X_test_raw = test_df.iloc[:, 3:].values

    print("Preprocessing spectra (SG 1st derivative + SNV)...")
    X_train = preprocess(X_train_raw)
    X_test = preprocess(X_test_raw)

    print("Training Leave-One-Species-Out SVR Ensemble (C=1.2, epsilon=0.05, kernel='rbf')...")
    y_train_log = np.log1p(y_train)
    unique_species = np.unique(train_species)
    predictions_all = []

    for sp in unique_species:
        train_idx = (train_species != sp)
        
        # Train fold model
        fold_model = SVR(C=1.2, gamma='auto', epsilon=0.05, kernel='rbf')
        fold_model.fit(X_train[train_idx], y_train_log[train_idx])
        
        # Predict on test set
        fold_preds_log = fold_model.predict(X_test)
        fold_preds = np.expm1(fold_preds_log)
        predictions_all.append(fold_preds)

    # Average the predictions
    avg_predictions = np.mean(predictions_all, axis=0)
    avg_predictions = np.clip(avg_predictions, 0.01, None)

    # Save to submission.csv
    submission_df = pd.DataFrame({
        "sample_number": test_sample_numbers,
        "moisture_content": avg_predictions
    })
    
    submission_df.to_csv("submission.csv", index=False, header=False)
    print("Saved SVR Ensemble predictions to submission.csv.")
    print(f"Predictions stats: Min={avg_predictions.min():.2f}, Max={avg_predictions.max():.2f}, Mean={avg_predictions.mean():.2f}")

if __name__ == "__main__":
    main()
