"""
Modul Optimasi Hyperparameter & Evaluasi Margin Guyon (2002)
Fokus: Memaksimalkan Sensitivitas (Recall) & Meminimalkan False Negative
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    recall_score, precision_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from src.config import RANDOM_STATE

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def compute_guyon_margins(decision_values: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Menghitung metrik kualitas dari Bagian 4.2 Paper Guyon et al. (2002):
    - Extremal Margin (E/D)
    - Median Margin (M/D)
    """
    pos_vals = decision_values[y_true == 1]
    neg_vals = decision_values[y_true == 0]

    D = np.max(decision_values) - np.min(decision_values) + 1e-8
    E = np.min(pos_vals) - np.max(neg_vals)
    M = np.median(pos_vals) - np.median(neg_vals)

    return {
        "extremal_margin": float(E / D),
        "median_margin": float(M / D)
    }


def optimize_svm_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 35):
    """
    Mencari konfigurasi hyperparameter terbaik (C, kernel, class_weight_pos)
    untuk memaksimalkan F2-Score (fokus Recall / meminimalkan False Negative).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if HAS_OPTUNA:
        def objective(trial):
            C = trial.suggest_float("C", 1e-3, 50.0, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
            pos_weight = trial.suggest_float("pos_class_weight", 1.5, 5.0)
            gamma = trial.suggest_float("gamma", 1e-4, 0.5, log=True) if kernel == "rbf" else "scale"

            f2_scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                clf = SVC(
                    C=C,
                    kernel=kernel,
                    gamma=gamma,
                    class_weight={0: 1.0, 1: pos_weight},
                    probability=True,
                    random_state=RANDOM_STATE
                )
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_val)
                f2 = fbeta_score(y_val, preds, beta=2, zero_division=0)
                f2_scores.append(f2)

            return float(np.mean(f2_scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params, study.best_value
    else:
        # Fallback grid search jika optuna belum terinstall
        best_score = -1.0
        best_params = {}
        for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
            for kernel in ["linear", "rbf"]:
                for pos_weight in [2.0, 3.0, 4.0]:
                    f2_scores = []
                    for train_idx, val_idx in cv.split(X_train, y_train):
                        clf = SVC(
                            C=C, kernel=kernel,
                            class_weight={0: 1.0, 1: pos_weight},
                            probability=True, random_state=RANDOM_STATE
                        )
                        clf.fit(X_train[train_idx], y_train[train_idx])
                        preds = clf.predict(X_train[val_idx])
                        f2_scores.append(fbeta_score(y_train[val_idx], preds, beta=2, zero_division=0))
                    mean_f2 = float(np.mean(f2_scores))
                    if mean_f2 > best_score:
                        best_score = mean_f2
                        best_params = {"C": C, "kernel": kernel, "pos_class_weight": pos_weight}
        return best_params, best_score


def evaluate_clinical_metrics(model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.30) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    decision_vals = model.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    margins = compute_guyon_margins(decision_vals, y_test)

    return {
        "accuracy": float((tp + tn) / len(y_test)),
        "recall_sensitivity": float(recall_score(y_test, preds)),
        "specificity": float(tn / (tn + fp + 1e-8)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "f2_score": float(fbeta_score(y_test, preds, beta=2)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "extremal_margin": margins["extremal_margin"],
        "median_margin": margins["median_margin"],
    }
