"""
Unit Testing untuk Generator Laporan Rekam Medis PDF (generate_clinical_pdf_report)
"""

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.report_generator import generate_clinical_pdf_report


def test_generate_clinical_pdf_report_valid_bytes():
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(["Biomarker A", "Biomarker B"], [0.25, -0.15])
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches="tight")
    plt.close(fig)
    img_buf.seek(0)

    patient_data = {
        "worst perimeter": 130.5,
        "worst texture": 28.4,
        "mean concavity": 0.12
    }

    pdf_bytes = generate_clinical_pdf_report(
        patient_data=patient_data,
        diagnosis="MALIGNANT",
        probability=0.912,
        top_features=["worst perimeter (+35.0%)", "worst texture (+20.0%)"],
        chart_img=img_buf.getvalue(),
        sample_id="ONCO-TEST-UNIT"
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_clinical_pdf_report_benign():
    patient_data = {
        "worst perimeter": 80.2,
        "worst texture": 15.1
    }

    pdf_bytes = generate_clinical_pdf_report(
        patient_data=patient_data,
        diagnosis=False,
        probability=0.085,
        top_features=[],
        chart_img=None,
        sample_id="ONCO-BENIGN-01"
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
