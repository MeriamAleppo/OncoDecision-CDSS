"""
Unit Testing untuk Memastikan Integritas Pipeline SVM-RFE dan Minimasi False Negative
"""

import numpy as np
from src.data_loader import get_train_test_data, guyon_preprocessing
from src.feature_selection import SVMRFEGuyonSelector, GolubBaselineSelector


def test_guyon_preprocessing_zero_mean():
    X_train, X_test, _, _, _ = get_train_test_data()
    X_tr_norm, _, _, _ = guyon_preprocessing(X_train, X_test, apply_squashing=False)
    assert np.allclose(np.mean(X_tr_norm, axis=0), 0.0, atol=1e-5)


def test_svm_rfe_eliminates_noise_genes():
    X_train, X_test, y_train, _, feature_names = get_train_test_data()
    X_tr_norm, _, _, _ = guyon_preprocessing(X_train, X_test, apply_squashing=True)

    selector = SVMRFEGuyonSelector(n_features_to_select=10, C=1.0, step=5)
    selector.fit(X_tr_norm, y_train)
    X_reduced = selector.transform(X_tr_norm)

    assert X_reduced.shape[1] == 10
    selected_names = [feature_names[i] for i in selector.selected_indices_]
    real_clinical_count = sum(1 for name in selected_names if not name.startswith("GENE_EXPR_"))
    assert real_clinical_count >= 7


def test_golub_baseline_selector_shape():
    X_train, _, y_train, _, _ = get_train_test_data()
    selector = GolubBaselineSelector(n_features_to_select=8).fit(X_train, y_train)
    assert selector.transform(X_train).shape[1] == 8
