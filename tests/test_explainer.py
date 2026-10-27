"""
Unit Testing untuk Modul SHAP Formal (CDSSShapExplainer)
"""

import numpy as np
import matplotlib.pyplot as plt
from src.trainer import load_cdss_artifact
from src.explainer import CDSSShapExplainer


def test_shap_explainer_initialization_and_inference():
    artifact, _ = load_cdss_artifact(force_retrain=False)
    model = artifact["model"]
    bg = artifact["X_train_selected_background"]
    feature_names = artifact["selected_feature_names"]

    explainer = CDSSShapExplainer(
        model=model,
        background_data=bg,
        feature_names=feature_names,
        n_kmeans=10
    )

    assert explainer.expected_value >= 0.0 and explainer.expected_value <= 1.0

    # Test single instance explanation
    sample = bg[0]
    exp = explainer.explain_instance(sample, nsamples=60)
    assert len(exp.values) == len(feature_names)
    assert np.all(np.isfinite(exp.values))

    # Test top risk drivers
    drivers = explainer.get_top_risk_drivers(exp.values, top_k=3)
    assert isinstance(drivers, list)
    assert len(drivers) <= 3
    for d in drivers:
        assert "feature" in d
        assert "shap_value" in d
        assert d["shap_value"] > 0

    # Test clinical narrative generation
    narrative = explainer.generate_clinical_narrative(exp.values, patient_prob=0.85, threshold=0.30)
    assert isinstance(narrative, str)
    assert len(narrative) > 100

    # Test plots
    fig = explainer.plot_waterfall(exp)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    force_html = explainer.plot_force_html(exp)
    assert "<script" in force_html or "javascript" in force_html.lower() or "shap" in force_html.lower()
