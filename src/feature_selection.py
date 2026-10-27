"""
Implementasi Algoritma Seleksi Fitur:
1. SVM-RFE (Guyon, Weston, Barnhill, & Vapnik, 2002 - Section 3.2)
2. Baseline Correlation Ranking (Golub et al., 1999 - Equation 2)
"""

import numpy as np
from sklearn.svm import SVC


class GolubBaselineSelector:
    """
    Metode Baseline Golub et al. (1999) sesuai Persamaan (2) di paper Guyon (2002):
    w_i = (mu_i(+) - mu_i(-)) / (sigma_i(+) + sigma_i(-))
    """
    def __init__(self, n_features_to_select: int = 12):
        self.n_features_to_select = n_features_to_select
        self.weights_ = None
        self.selected_indices_ = None
        self.ranked_indices_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        pos_mask = (y == 1)
        neg_mask = (y == 0)

        mu_pos = np.mean(X[pos_mask], axis=0)
        mu_neg = np.mean(X[neg_mask], axis=0)
        sigma_pos = np.std(X[pos_mask], axis=0)
        sigma_neg = np.std(X[neg_mask], axis=0)

        self.weights_ = (mu_pos - mu_neg) / (sigma_pos + sigma_neg + 1e-8)
        self.ranked_indices_ = np.argsort(np.abs(self.weights_))[::-1]
        self.selected_indices_ = self.ranked_indices_[:self.n_features_to_select]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]


class SVMRFEGuyonSelector:
    """
    Implementasi eksak Algoritma SVM-RFE dari Guyon et al. (2002) Bagian 3.2:
    - Melatih Linear SVM
    - Menghitung vektor bobot w = sum(alpha_k * y_k * x_k)
    - Menghitung kriteria ranking c_i = (w_i)^2
    - Mengeliminasi fitur dengan c_i terkecil secara iteratif hingga mencapai subset optimal.
    """
    def __init__(self, n_features_to_select: int = 12, C: float = 1.0, step: int = 1):
        self.n_features_to_select = n_features_to_select
        self.C = C
        self.step = step
        self.ranked_list_ = []
        self.selected_indices_ = None
        self.final_weights_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        s = list(range(n_features))
        r = []

        while len(s) > 0:
            X_restricted = X[:, s]
            svm = SVC(kernel="linear", C=self.C, class_weight="balanced")
            svm.fit(X_restricted, y)

            w = svm.coef_[0]
            c = w ** 2

            n_remove = min(self.step, len(s))
            smallest_idx_in_s = np.argsort(c)[:n_remove]

            for idx in smallest_idx_in_s[::-1]:
                r.insert(0, s[idx])

            for idx in sorted(smallest_idx_in_s, reverse=True):
                s.pop(idx)

        self.ranked_list_ = r
        self.selected_indices_ = np.array(self.ranked_list_[:self.n_features_to_select])

        svm_final = SVC(kernel="linear", C=self.C, class_weight="balanced")
        svm_final.fit(X[:, self.selected_indices_], y)
        self.final_weights_ = (svm_final.coef_[0]) ** 2
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_indices_]
