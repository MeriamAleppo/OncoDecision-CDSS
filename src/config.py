"""
Konfigurasi Path dan Parameter Global Tugas Akhir
Penyusun: Nayyara Rauf Ghani Al Khayri (NIM: A11.2024.16055)
Dosen Pembimbing: Abu Salam, M.Kom
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = DATA_RAW_DIR / "high_dim_cancer_clinical.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "selected_features_cancer.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "oncodecision_cdss_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "experiment_metadata.json"

RANDOM_STATE = 42
DEFAULT_N_FEATURES_TO_SELECT = 12
RECALL_OPTIMIZED_THRESHOLD = 0.30
