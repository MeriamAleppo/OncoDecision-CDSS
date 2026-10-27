"""
Pipeline Orkestrasi Pelatihan & Komparasi Eksperimen Tugas Akhir
Penyusun: Nayyara Rauf Ghani Al Khayri (NIM: A11.2024.16055)
Dosen Pembimbing: Abu Salam, M.Kom
"""

import json
import logging
import platform
import warnings
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.svm import SVC

from src.config import (
    MODEL_ARTIFACT_PATH, METADATA_PATH, PROCESSED_DATA_PATH,
    DEFAULT_N_FEATURES_TO_SELECT, RECALL_OPTIMIZED_THRESHOLD, RANDOM_STATE
)
from src.data_loader import get_train_test_data, guyon_preprocessing
from src.feature_selection import GolubBaselineSelector, SVMRFEGuyonSelector
from src.optimizer import optimize_svm_hyperparameters, evaluate_clinical_metrics

logger = logging.getLogger(__name__)


def run_training_pipeline(n_features: int = DEFAULT_N_FEATURES_TO_SELECT, n_trials: int = 35) -> dict:
    """
    Menjalankan end-to-end pipeline pelatihan SVM-RFE & optimasi Optuna.
    Hasil pelatihan disimpan dalam format dictionary joblib dan JSON metadata
    yang dilengkapi informasi versi scikit-learn dan Python.
    """
    print("[1/5] Memuat dataset medis dimensi tinggi (30 biomarker klinis + 70 gen noise)...")
    X_train_raw, X_test_raw, y_train, y_test, feature_names = get_train_test_data()

    print("[2/5] Menjalankan Preprocessing Guyon et al. (2002)...")
    X_train, X_test, mean_vec, std_vec = guyon_preprocessing(X_train_raw, X_test_raw, apply_squashing=True)

    print(f"[3/5] Mengeksekusi Seleksi Fitur Baseline Golub (1999) vs SVM-RFE Guyon (2002) ({n_features} fitur)...")
    golub_selector = GolubBaselineSelector(n_features_to_select=n_features).fit(X_train, y_train)
    X_tr_golub = golub_selector.transform(X_train)
    X_te_golub = golub_selector.transform(X_test)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*probability.*")
        baseline_clf = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
        baseline_clf.fit(X_tr_golub, y_train)
    baseline_metrics = evaluate_clinical_metrics(baseline_clf, X_te_golub, y_test, threshold=0.50)

    rfe_selector = SVMRFEGuyonSelector(n_features_to_select=n_features, C=0.1, step=1).fit(X_train, y_train)
    X_tr_rfe = rfe_selector.transform(X_train)
    X_te_rfe = rfe_selector.transform(X_test)

    selected_feature_names = [feature_names[i] for i in rfe_selector.selected_indices_]

    df_proc = pd.DataFrame(X_tr_rfe, columns=selected_feature_names)
    df_proc["target_malignant"] = y_train
    df_proc.to_csv(PROCESSED_DATA_PATH, index=False)

    print("[4/5] Menjalankan Optimasi Hyperparameter berorientasi F2-Score & Recall...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*probability.*")
        best_params, best_cv_f2 = optimize_svm_hyperparameters(X_tr_rfe, y_train, n_trials=n_trials)

        final_model = SVC(
            C=best_params["C"],
            kernel=best_params["kernel"],
            gamma=best_params.get("gamma", "scale"),
            class_weight={0: 1.0, 1: best_params["pos_class_weight"]},
            probability=True,
            random_state=RANDOM_STATE
        )
        final_model.fit(X_tr_rfe, y_train)

    proposed_metrics = evaluate_clinical_metrics(
        final_model, X_te_rfe, y_test, threshold=RECALL_OPTIMIZED_THRESHOLD
    )

    print("[5/5] Menyimpan artefak model dan metadata ke folder models/...")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_sklearn_version = sklearn.__version__
    current_python_version = platform.python_version()

    artifact = {
        "model": final_model,
        "rfe_selector": rfe_selector,
        "mean_vec": mean_vec,
        "std_vec": std_vec,
        "all_feature_names": feature_names,
        "selected_feature_names": selected_feature_names,
        "selected_indices": rfe_selector.selected_indices_.tolist(),
        "rfe_weights_squared": rfe_selector.final_weights_.tolist(),
        "threshold": RECALL_OPTIMIZED_THRESHOLD,
        "X_train_selected_background": X_tr_rfe[:50],
        "sklearn_version": current_sklearn_version,
        "python_version": current_python_version,
        "created_at": current_time
    }
    joblib.dump(artifact, MODEL_ARTIFACT_PATH)

    metadata = {
        "sklearn_version": current_sklearn_version,
        "python_version": current_python_version,
        "created_at": current_time,
        "total_initial_features": len(feature_names),
        "selected_features_count": n_features,
        "selected_features": selected_feature_names,
        "best_hyperparameters": best_params,
        "baseline_golub_metrics": baseline_metrics,
        "proposed_svmrfe_optuna_metrics": proposed_metrics
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n=== HASIL KOMPARASI EKSPERIMEN TUGAS AKHIR ===")
    print(f"Fitur Terpilih (SVM-RFE) : {selected_feature_names}")
    print(f"Baseline (Golub 1999)    -> Recall: {baseline_metrics['recall_sensitivity']*100:.2f}% | False Negative: {baseline_metrics['false_negatives']}")
    print(f"Proposed (SVM-RFE+Optuna)-> Recall: {proposed_metrics['recall_sensitivity']*100:.2f}% | False Negative: {proposed_metrics['false_negatives']}")
    return metadata


def load_cdss_artifact(
    artifact_path: Path = MODEL_ARTIFACT_PATH,
    metadata_path: Path = METADATA_PATH,
    force_retrain: bool = False
) -> tuple[dict, dict]:
    """
    Memuat artefak model dan metadata dengan pengecekan kompatibilitas versi scikit-learn.

    Mekanisme Keandalan (Self-Healing Fallback):
    1. Memeriksa keberadaan file artefak dan metadata.
    2. Melakukan pre-check versi scikit-learn melalui metadata JSON.
    3. Menginspeksi unpickling dengan penangkapan InconsistentVersionWarning.
    4. Jika terdeteksi mismatch versi, korupsi data, atau modul usang,
       fungsi secara otomatis mengeksekusi retraining pipeline fallback
       agar artefak sinkron 100% dengan interpreter aktif tanpa melempar warning.

    Returns
    -------
    tuple[dict, dict]
        (artifact, metadata)
    """
    need_retrain = force_retrain or not artifact_path.exists() or not metadata_path.exists()
    retrain_reason = ""

    # 1. Pengecekan versi scikit-learn pada metadata JSON
    if not need_retrain:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_cached = json.load(f)
            saved_sklearn_ver = meta_cached.get("sklearn_version")
            if saved_sklearn_ver and saved_sklearn_ver != sklearn.__version__:
                need_retrain = True
                retrain_reason = (
                    f"Mismatch versi scikit-learn terdeteksi pada metadata "
                    f"(Artefak: v{saved_sklearn_ver}, Lingkungan: v{sklearn.__version__})"
                )
        except Exception as e:
            need_retrain = True
            retrain_reason = f"Gagal membaca metadata JSON: {e}"

    # 2. Pengujian deserialisasi artefak model dengan penangkapan warning
    if not need_retrain:
        try:
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")
                temp_artifact = joblib.load(artifact_path)

                # Deteksi warning ketidakkonsistenan versi dari sklearn
                for w in captured_warnings:
                    if issubclass(w.category, InconsistentVersionWarning) or "unpickle estimator" in str(w.message).lower():
                        need_retrain = True
                        retrain_reason = f"InconsistentVersionWarning saat unpickling: {w.message}"
                        break

                if not need_retrain:
                    art_sklearn_ver = temp_artifact.get("sklearn_version")
                    if art_sklearn_ver and art_sklearn_ver != sklearn.__version__:
                        need_retrain = True
                        retrain_reason = (
                            f"Mismatch versi scikit-learn pada artefak model "
                            f"(Artefak: v{art_sklearn_ver}, Lingkungan: v{sklearn.__version__})"
                        )
                    elif art_sklearn_ver is None:
                        need_retrain = True
                        retrain_reason = "Artefak model tidak memiliki label versi scikit-learn (artefak lama/legacy)"

        except Exception as err:
            need_retrain = True
            retrain_reason = f"Kesalahan saat unpickling artefak model: {err}"

    # 3. Eksekusi fallback retraining bila diperlukan
    if need_retrain:
        print(f"\n[OncoDecision AI] [!] Memicu Fallback Retraining: {retrain_reason}")
        print(f"[OncoDecision AI] [*] Melatih ulang pipeline dengan scikit-learn {sklearn.__version__}...")
        run_training_pipeline()

    # 4. Deserialisasi bersih final
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        artifact = joblib.load(artifact_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return artifact, metadata


if __name__ == "__main__":
    run_training_pipeline()
