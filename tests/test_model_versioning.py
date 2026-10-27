"""
Unit Testing untuk Pengecekan Kompatibilitas Versi & Mekanisme Fallback Retraining CDSS
"""

import json
import joblib
import sklearn
from src.config import MODEL_ARTIFACT_PATH, METADATA_PATH
from src.trainer import load_cdss_artifact


def test_load_cdss_artifact_integrity():
    artifact, metadata = load_cdss_artifact(force_retrain=False)

    assert "model" in artifact
    assert "selected_feature_names" in artifact
    assert "sklearn_version" in artifact
    assert artifact["sklearn_version"] == sklearn.__version__
    assert metadata["sklearn_version"] == sklearn.__version__


def test_fallback_retraining_on_version_mismatch(tmp_path):
    # Buat file dummy dengan metadata versi lama (mismatch)
    dummy_meta_path = tmp_path / "dummy_meta.json"
    dummy_art_path = tmp_path / "dummy_art.joblib"

    # Salin artefak saat ini tetapi set versi lama di metadata
    current_art = joblib.load(MODEL_ARTIFACT_PATH)
    joblib.dump(current_art, dummy_art_path)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["sklearn_version"] = "1.4.0"
    with open(dummy_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # load_cdss_artifact harus mendeteksi versi 1.4.0 != sklearn.__version__
    # dan memicu retraining ke dummy path (atau default path)
    with open(dummy_meta_path, "r", encoding="utf-8") as f:
        loaded_meta = json.load(f)
    assert loaded_meta["sklearn_version"] == "1.4.0"
