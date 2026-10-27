"""
Modul Generator Laporan Rekam Medis Klinis (PDF Report Generator)
Clinical Decision Support System - OncoDecision AI
Penyusun: Nayyara Rauf Ghani Al Khayri (NIM: A11.2024.16055)
Dosen Pembimbing: Abu Salam, M.Kom
"""

import io
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Kustomisasi Canvas ReportLab untuk menambahkan nomor halaman dinamis
    (Halaman X dari Y) dan footer kerahasiaan medis pada setiap halaman.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748b"))

        # Garis pembatas footer
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.6)
        self.line(36, 32, A4[0] - 36, 32)

        # Footer kiri & kanan
        left_text = "OncoDecision AI — Dokumen Rekam Medis Rahasia & Terlindungi (Confidential Medical Record)"
        right_text = f"Halaman {self._pageNumber} dari {page_count}"
        self.drawString(36, 21, left_text)
        self.drawRightString(A4[0] - 36, 21, right_text)

        self.restoreState()


def generate_clinical_pdf_report(
    patient_data: dict,
    diagnosis: str | bool,
    probability: float,
    top_features: list,
    chart_img: bytes | io.BytesIO | str,
    sample_id: str = "ONCO-2026-X8921",
    hospital_name: str = "Pusat Diagnostik Onkologi & Patologi Terpadu (RS Akademik)",
    threshold: float = 0.30,
    df_raw: pd.DataFrame = None,
    clinical_notes: str = None
) -> bytes:
    """
    Menghasilkan dokumen resmi PDF Rekam Medis Klinis CDSS OncoDecision AI.

    Parameters
    ----------
    patient_data : dict
        Dictionary biomarker dan nilainya {nama_biomarker: nilai}.
    diagnosis : str | bool
        Hasil keputusan klinis ('MALIGNANT', 'BENIGN', atau boolean is_malignant).
    probability : float
        Nilai probabilitas risiko keganasan (0.0 s.d 1.0).
    top_features : list
        Daftar biomarker pemicu utama (nama fitur atau tuple/dict dengan kontribusi).
    chart_img : bytes | io.BytesIO | str
        Gambar diagram Explainable AI (SHAP Waterfall/Bar) dalam format bytes atau path.
    sample_id : str
        ID unik sampel pasien.
    hospital_name : str
        Nama institusi medis / laboratorium rujukan.
    threshold : float
        Ambang batas kalibrasi sensitivitas tinggi (default: 0.30).
    df_raw : pd.DataFrame, optional
        Dataset acuan untuk menghitung rentang nilai rujukan normal.
    clinical_notes : str, optional
        Catatan naratif tambahan dari dokter pemeriksa.

    Returns
    -------
    bytes
        Buffer byte dokumen PDF yang siap diunduh atau disimpan.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Kustomisasi Tipografi & Gaya Paragraf
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        spaceAfter=6
    )
    section_h2 = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#334155")
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b")
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold"
    )

    story = []

    # -------------------------------------------------------------------------
    # 1. HEADER DOKUMEN: Identitas Sistem & Institusi Medis
    # -------------------------------------------------------------------------
    current_time_str = datetime.now().strftime("%d %B %Y, %H:%M:%S WIB")

    header_table_data = [
        [
            Paragraph(f"<b>{hospital_name.upper()}</b>", title_style),
            Paragraph("<b>DOKUMEN RESMI CDSS</b><br/><font color='#64748b' size='7.5'>FORMULIR RM-ONCO-01</font>", ParagraphStyle("RightHdr", parent=table_cell_bold, alignment=2))
        ],
        [
            Paragraph("Sistem Pendukung Keputusan Klinis (CDSS) Deteksi Kanker — <b>OncoDecision AI</b>", subtitle_style),
            Paragraph(f"<font color='#64748b'>Waktu Pemeriksaan: {current_time_str}</font>", ParagraphStyle("RightTime", parent=table_cell, alignment=2))
        ]
    ]
    header_table = Table(header_table_data, colWidths=[350, 173])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f172a"), spaceBefore=6, spaceAfter=8))

    # Identitas Pasien & Sampel
    meta_data = [
        [
            Paragraph("<b>ID Rekam Medis / Sampel:</b>", table_cell),
            Paragraph(f"<b><font color='#0284c7'>{sample_id}</font></b>", table_cell_bold),
            Paragraph("<b>Metodologi Pipeline:</b>", table_cell),
            Paragraph("SVM-RFE (Guyon 2002) + Optuna", table_cell)
        ],
        [
            Paragraph("<b>Penyusun Sistem:</b>", table_cell),
            Paragraph("Nayyara Rauf Ghani (A11.2024.16055)", table_cell),
            Paragraph("<b>Dosen Pembimbing:</b>", table_cell),
            Paragraph("Abu Salam, M.Kom", table_cell)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[120, 150, 110, 143])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # 2. HASIL KEPUTUSAN PREDIKTIF CDSS (Diagnostic Callout Box)
    # -------------------------------------------------------------------------
    prob_pct = probability * 100
    is_mal = (diagnosis is True) or (isinstance(diagnosis, str) and "MALIGNANT" in diagnosis.upper())

    if is_mal:
        banner_bg = colors.HexColor("#fef2f2")
        banner_border = colors.HexColor("#ef4444")
        banner_title_color = "#991b1b"
        status_label = "TERINDIKASI MALIGNANT (KANKER / POSITIF)"
        rec_text = (
            "<b>Rekomendasi Onkologi:</b> Pasien memiliki profil biomarker dengan risiko keganasan tinggi di atas ambang "
            f"batas klinis ({threshold*100:.0f}%). Disarankan segera melakukan pemeriksaan histopatologi definitif "
            "(Core Needle Biopsy) dan konsultasi onkologi multidisiplin."
        )
    else:
        banner_bg = colors.HexColor("#f0fdf4")
        banner_border = colors.HexColor("#10b981")
        banner_title_color = "#166534"
        status_label = "TERINDIKASI BENIGN (JINAK / NEGATIF)"
        rec_text = (
            "<b>Rekomendasi Onkologi:</b> Nilai biomarker berada dalam kisaran kecurigaan rendah di bawah ambang batas "
            f"kritis ({threshold*100:.0f}%). Pasien disarankan melanjutkan pemantauan klinis berkala (surveillance rutin)."
        )

    diag_data = [
        [
            Paragraph(f"<font size='11' color='{banner_title_color}'><b>HASIL SKRINING: {status_label}</b></font>", table_cell_bold),
            Paragraph(f"<font size='11' color='{banner_title_color}'><b>Confidence: {prob_pct:.1f}%</b></font>", ParagraphStyle("DiagProb", parent=table_cell_bold, alignment=2))
        ],
        [
            Paragraph(f"<font color='#334155'>{rec_text}</font>", table_cell),
            Paragraph(f"<font size='7.5' color='#64748b'>Ambang Batas (Threshold): {threshold*100:.0f}%<br/>Sensitivitas Dioptimalkan</font>", ParagraphStyle("Thresh", parent=table_cell, alignment=2))
        ]
    ]
    diag_table = Table(diag_data, colWidths=[380, 143])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), banner_bg),
        ("BOX", (0, 0), (-1, -1), 1.2, banner_border),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 3. TABEL PARAMETER BIOMARKER TERPILIH (SVM-RFE)
    # -------------------------------------------------------------------------
    story.append(Paragraph("<b>1. Profil Nilai Parameter Biomarker Pasien vs Batas Rujukan</b>", section_h2))

    table_headers = [
        Paragraph("<b>No</b>", table_cell_bold),
        Paragraph("<b>Nama Biomarker (SVM-RFE)</b>", table_cell_bold),
        Paragraph("<b>Nilai Pasien</b>", ParagraphStyle("HdrC", parent=table_cell_bold, alignment=1)),
        Paragraph("<b>Rentang Rujukan (Min - Max)</b>", ParagraphStyle("HdrC", parent=table_cell_bold, alignment=1)),
        Paragraph("<b>Rata-rata Acuan</b>", ParagraphStyle("HdrC", parent=table_cell_bold, alignment=1)),
        Paragraph("<b>Status</b>", ParagraphStyle("HdrC", parent=table_cell_bold, alignment=1)),
    ]
    table_rows = [table_headers]

    for idx, (feat_name, val) in enumerate(patient_data.items(), 1):
        if df_raw is not None and feat_name in df_raw.columns:
            ref_min = float(df_raw[feat_name].min())
            ref_max = float(df_raw[feat_name].max())
            ref_mean = float(df_raw[feat_name].mean())
            ref_std = float(df_raw[feat_name].std())
            ref_str = f"{ref_min:.2f} - {ref_max:.2f}"
            mean_str = f"{ref_mean:.2f}"
            
            # Kategori status klinis
            if val > ref_mean + 0.8 * ref_std:
                status_txt = "<font color='#dc2626'><b>Elevasi (Risiko)</b></font>"
            elif val < ref_mean - 0.8 * ref_std:
                status_txt = "<font color='#0284c7'>Rendah</font>"
            else:
                status_txt = "<font color='#16a34a'>Normal</font>"
        else:
            ref_str = "Tersedia di DB"
            mean_str = "-"
            status_txt = "Terekam"

        row = [
            Paragraph(str(idx), table_cell),
            Paragraph(f"<b>{feat_name}</b>", table_cell),
            Paragraph(f"{float(val):.4f}", ParagraphStyle("ValC", parent=table_cell, alignment=1)),
            Paragraph(ref_str, ParagraphStyle("RefC", parent=table_cell, alignment=1)),
            Paragraph(mean_str, ParagraphStyle("MeanC", parent=table_cell, alignment=1)),
            Paragraph(status_txt, ParagraphStyle("StatC", parent=table_cell, alignment=1)),
        ]
        table_rows.append(row)

    param_table = Table(table_rows, colWidths=[24, 180, 75, 105, 65, 74])
    param_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 1), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 2.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(param_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 4. EXPLAINABLE AI (XAI) CHART & TOP DRIVER FEATURES
    # -------------------------------------------------------------------------
    story.append(Paragraph("<b>2. Analisis Explainable AI (SHAP Waterfall Attribution — Lundberg & Lee, 2017)</b>", section_h2))

    # Format ringkasan top features
    top_feature_names = []
    for item in top_features:
        if isinstance(item, dict):
            top_feature_names.append(f"<b>{item.get('feature', 'Biomarker')}</b> (+{item.get('shap_value', 0)*100:.1f}%)")
        else:
            top_feature_names.append(f"<b>{str(item)}</b>")
    
    top_str = ", ".join(top_feature_names[:3]) if top_feature_names else "Data biomarker individual"
    driver_desc = (
        f"Diagram di bawah memperlihatkan dekomposisi kontribusi marginal masing-masing biomarker dalam menggeser "
        f"probabilitas dasar populasi ke arah keputusan akhir. <b>Biomarker pendorong risiko utama:</b> {top_str}."
    )
    story.append(Paragraph(driver_desc, body_style))
    story.append(Spacer(1, 6))

    # Sematkan gambar XAI Chart
    if chart_img is not None:
        try:
            if isinstance(chart_img, bytes):
                img_io = io.BytesIO(chart_img)
            elif isinstance(chart_img, io.BytesIO):
                img_io = chart_img
            else:
                img_io = chart_img

            xai_image = Image(img_io, width=470, height=195)
            story.append(xai_image)
        except Exception as e:
            story.append(Paragraph(f"<i>[Gambar grafik XAI tidak dapat dirender: {e}]</i>", body_style))

    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 5. FOOTER & OFFICIAL MEDICAL DISCLAIMER
    # -------------------------------------------------------------------------
    disclaimer_text = (
        "<b>PERINGATAN & DISCLAIMER MEDIS RESMI:</b><br/>"
        "Dokumen ini diterbitkan secara otomatis oleh sistem <b>OncoDecision AI</b> berbasis kecerdasan buatan "
        "sebagai instrumen <i>Clinical Decision Support System (CDSS)</i>. Seluruh estimasi risiko dan klasifikasi biomarker "
        "berfungsi sebagai <u>alat bantu penunjang keputusan klinis</u> dan <b>TIDAK</b> menggantikan pertimbangan profesional "
        "maupun diagnosis final dokter spesialis. Konfirmasi diagnostik definitif <b>WAJIB</b> dilakukan oleh "
        "<b>Dokter Spesialis Patologi Anatomi</b> melalui pemeriksaan histopatologi biopsi jaringan resmi."
    )

    sign_data = [
        [
            Paragraph(disclaimer_text, ParagraphStyle("Disc", parent=body_style, fontSize=7.5, leading=10, textColor=colors.HexColor("#475569"))),
            Paragraph(
                "<b>Diverifikasi Sistem CDSS:</b><br/>"
                "<i>OncoDecision AI v1.0</i><br/>"
                "<font color='#0284c7'>[TEREVALUASI DENGAN OPTUNA]</font><br/><br/>"
                "<b>Dokter / Operator:</b><br/>"
                "__________________________",
                ParagraphStyle("Sign", parent=table_cell, fontSize=7.5, leading=10, alignment=2)
            )
        ]
    ]
    sign_table = Table(sign_data, colWidths=[360, 163])
    sign_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (0, 0), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(KeepTogether(sign_table))

    # Bangun dokumen PDF dengan NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
