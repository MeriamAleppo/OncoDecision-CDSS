"""
Modul Data Ingestion & Preprocessing sesuai Guyon et al. (2002)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, RANDOM_STATE


def generate_high_dimensional_medical_data(n_noise_genes: int = 70) -> pd.DataFrame:
    """
    Memuat dataset klinis Breast Cancer Wisconsin (30 fitur morfologi asli)
    dan menambahkan 70 fitur ekspresi gen sintetis berdimensi tinggi:
    - 15 gen kolinear redundan (salinan ber-noise dari 1 fitur tunggal yang mengecoh ranking univariat Golub,
      persis seperti ilustrasi teoretis Guyon et al., 2002 Bagian 6.2.1).
    - 55 gen Gaussian noise mikroarray.
    Label: 1 = Malignant (Kanker/Positif), 0 = Benign (Jinak/Negatif).
    """
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH)

    data = load_breast_cancer()
    df_real = pd.DataFrame(data.data, columns=data.feature_names)
    y_clinical = np.where(data.target == 0, 1, 0)

    rng = np.random.default_rng(RANDOM_STATE)
    n_samples = df_real.shape[0]

    noise_data = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_noise_genes))
    # 15 gen redundan kolinear dari satu fitur lemah namun tampak korelatif secara univariat
    base_col = (df_real["mean radius"].values - df_real["mean radius"].mean()) / df_real["mean radius"].std()
    for j in range(15):
        noise_data[:, j] = base_col + rng.normal(0, 0.25, size=n_samples)

    noise_cols = [f"GENE_EXPR_{i:04d}" for i in range(1, n_noise_genes + 1)]
    df_noise = pd.DataFrame(noise_data, columns=noise_cols)

    df_full = pd.concat([df_real, df_noise], axis=1)
    df_full["target_malignant"] = y_clinical

    df_full.to_csv(RAW_DATA_PATH, index=False)
    return df_full


def guyon_preprocessing(X_train: np.ndarray, X_test: np.ndarray, apply_squashing: bool = False, c: float = 2.0):
    """
    Implementasi Preprocessing Guyon et al. (2002) Section 5.1 & 5.2:
    1. Kurangi mean dan bagi dengan standar deviasi pada setiap fitur.
    2. Opsional: Fungsi squashing f(x) = c * arctan(x / c) untuk meredam outlier.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    if apply_squashing:
        X_train_norm = c * np.arctan(X_train_norm / c)
        X_test_norm = c * np.arctan(X_test_norm / c)

    return X_train_norm, X_test_norm, mean, std


def get_train_test_data(test_size: float = 0.2, n_noise_genes: int = 70):
    df = generate_high_dimensional_medical_data(n_noise_genes=n_noise_genes)
    feature_names = [col for col in df.columns if col != "target_malignant"]
    X = df[feature_names].values
    y = df["target_malignant"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, feature_names
