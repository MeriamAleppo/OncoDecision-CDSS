"""
Aplikasi Web Clinical Decision Support System (CDSS) - OncoDecision AI
Sistem Deteksi Dini Kanker Berbasis Seleksi Fitur SVM-RFE & Optimasi Hyperparameter (Optuna)
Dilengkapi Modul Explainable AI Formal (SHAP) & Ekspor Rekam Medis Resmi (PDF ReportLab)
Penyusun: Nayyara Rauf Ghani Al Khayri (NIM: A11.2024.16055)
Dosen Pembimbing: Abu Salam, M.Kom
"""

import io
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import sklearn

from src.config import RAW_DATA_PATH
from src.trainer import load_cdss_artifact
from src.explainer import CDSSShapExplainer
from src.report_generator import generate_clinical_pdf_report

# -----------------------------------------------------------------------------
# Konfigurasi Halaman Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="OncoDecision AI - CDSS Kanker",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# Streamlit Caching Optimization
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat dan memvalidasi model OncoDecision AI...")
def get_cdss_pipeline() -> tuple[dict, dict]:
    """
    Memuat artefak model ML dan metadata eksperimen dengan validasi kompatibilitas versi.
    Dichace menggunakan @st.cache_resource agar objek model SVC, bobot RFE, dan vektor
    normalisasi tetap berada di memori dan tidak di-reload berulang saat pengguna
    berinteraksi dengan input formulir biomarker.
    """
    return load_cdss_artifact()


@st.cache_resource(show_spinner="Menginisialisasi modul Explainable AI (SHAP KernelExplainer)...")
def get_shap_explainer(_model, _background_data: np.ndarray, feature_names: tuple[str, ...]) -> CDSSShapExplainer:
    """
    Menginisialisasi instance CDSSShapExplainer dengan data latar (background dataset).
    Dichace menggunakan @st.cache_resource agar kompresi latar k-means dan inisialisasi
    KernelExplainer hanya dieksekusi satu kali untuk efisiensi latensi.
    """
    return CDSSShapExplainer(
        model=_model,
        background_data=_background_data,
        feature_names=list(feature_names),
        n_kmeans=10
    )


@st.cache_data(show_spinner="Memuat dataset referensi biomarker...")
def get_raw_dataset() -> pd.DataFrame:
    """
    Memuat dataset acuan biomarker klinis dari disk.
    Dichace menggunakan @st.cache_data untuk mengeliminasi disk I/O berulang pada setiap rerun.
    """
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def generate_sample_csv(_df_raw: pd.DataFrame, features: tuple[str, ...]) -> bytes:
    """
    Menghasilkan data template CSV untuk skrining massal secara ter-cache.
    Prefix underscore pada `_df_raw` digunakan agar Streamlit tidak melakukan hashing berlebih.
    """
    return _df_raw[list(features)].head(10).to_csv(index=False).encode("utf-8")


# Inisialisasi resource ter-cache
artifact, metadata = get_cdss_pipeline()
df_raw = get_raw_dataset()

model = artifact["model"]
selected_features = artifact["selected_feature_names"]
selected_indices = artifact["selected_indices"]
mean_vec = artifact["mean_vec"]
std_vec = artifact["std_vec"]
threshold = float(artifact["threshold"])
background_data = artifact["X_train_selected_background"]

# Inisialisasi SHAP Explainer ter-cache
explainer = get_shap_explainer(model, background_data, tuple(selected_features))


# -----------------------------------------------------------------------------
# Helper UI: Clinical Workstation AI Card (Modern SaaS Dark Aesthetic)
# -----------------------------------------------------------------------------
def render_workstation_card(is_malignant: bool, prob: float, latency: float, top_feature: str) -> None:
    """
    Menampilkan kartu stasiun kerja klinis interaktif ('Clinical Workstation AI Card')
    dengan desain modern, rounded container, dan dark-mode aesthetic ala platform SaaS medis OncoDoc AI.
    """
    prob_pct = prob * 100
    if latency < 0.08:
        latency_str = f"< 0.08 Detik ({latency * 1000:.1f} ms)"
    else:
        latency_str = f"{latency:.3f} Detik"

    if is_malignant:
        badge_bg = "rgba(239, 68, 68, 0.18)"
        badge_border = "rgba(239, 68, 68, 0.45)"
        badge_color = "#f87171"
        badge_text = "MALIGNANT INDICATION"
        badge_icon = "⚠️"
        card_gradient = "linear-gradient(145deg, rgba(239, 68, 68, 0.14) 0%, rgba(15, 23, 42, 0.85) 100%)"
        card_border = "rgba(239, 68, 68, 0.35)"
        confidence_color = "#f87171"
        rec_text = "Pasien membutuhkan pemeriksaan histopatologi / biopsi lanjutan segera."
        rec_color = "#fca5a5"
    else:
        badge_bg = "rgba(16, 185, 129, 0.18)"
        badge_border = "rgba(16, 185, 129, 0.45)"
        badge_color = "#34d399"
        badge_text = "BENIGN INDICATION"
        badge_icon = "✅"
        card_gradient = "linear-gradient(145deg, rgba(16, 185, 129, 0.14) 0%, rgba(15, 23, 42, 0.85) 100%)"
        card_border = "rgba(16, 185, 129, 0.35)"
        confidence_color = "#34d399"
        rec_text = "Risiko keganasan rendah berdasarkan profil biomarker saat ini."
        rec_color = "#6ee7b7"

    html_card = f"""
    <div style="
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif;
        color: #f8fafc;
        margin-bottom: 16px;
    ">
        <!-- Header: Status Indicator & SaaS Badge -->
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="
                    width: 9px;
                    height: 9px;
                    border-radius: 50%;
                    background: #a855f7;
                    box-shadow: 0 0 10px #a855f7, 0 0 4px #c084fc;
                    display: inline-block;
                "></span>
                <span style="font-size: 13px; font-weight: 600; color: #e2e8f0; letter-spacing: 0.3px;">
                    Oncology Workstation AI &mdash; Real-time Screening
                </span>
            </div>
            <span style="
                background: rgba(168, 85, 247, 0.15);
                color: #d8b4fe;
                border: 1px solid rgba(168, 85, 247, 0.4);
                border-radius: 9999px;
                padding: 3px 10px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.6px;
                text-transform: uppercase;
            ">
                Live Inference
            </span>
        </div>

        <!-- Main Diagnostic Box -->
        <div style="
            background: {card_gradient};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 16px;
        ">
            <div style="display: inline-flex; align-items: center; gap: 6px; background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}; padding: 5px 12px; border-radius: 8px; font-weight: 700; font-size: 12px; letter-spacing: 0.5px; margin-bottom: 12px;">
                <span>{badge_icon}</span> {badge_text}
            </div>

            <div style="margin-bottom: 12px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; font-weight: 600; letter-spacing: 0.6px; margin-bottom: 2px;">
                    Estimasi Probabilitas Risiko
                </div>
                <div style="font-size: 28px; font-weight: 800; color: {confidence_color}; letter-spacing: -0.5px;">
                    Confidence: {prob_pct:.1f}%
                </div>
            </div>

            <div style="font-size: 12px; color: {rec_color}; margin-bottom: 14px; line-height: 1.4;">
                {rec_text}
            </div>

            <!-- Top Driver Feature -->
            <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 10px 14px;">
                <div style="font-size: 10px; text-transform: uppercase; color: #94a3b8; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 3px;">
                    Biomarker Pemicu Utama (SHAP Top Driver)
                </div>
                <div style="color: #38bdf8; font-family: 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 13px; font-weight: 700;">
                    {top_feature}
                </div>
            </div>
        </div>

        <!-- Operational Metrics (2-Kolom Grid) -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; padding: 10px 12px;">
                <div style="font-size: 10px; text-transform: uppercase; color: #94a3b8; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 3px;">
                    Processing Time
                </div>
                <div style="font-size: 13px; font-weight: 700; color: #f1f5f9;">
                    {latency_str}
                </div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; padding: 10px 12px;">
                <div style="font-size: 10px; text-transform: uppercase; color: #94a3b8; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 3px;">
                    Pipeline Standard
                </div>
                <div style="font-size: 13px; font-weight: 700; color: #f1f5f9;">
                    SVM-RFE / SHAP XAI
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(html_card, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar: Informasi Status Sistem & Kontrol Retraining
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Status Sistem CDSS")
    st.success("✅ **Status:** Pipeline Aktif & Terkalibrasi")
    st.markdown(
        f"- **scikit-learn:** `v{sklearn.__version__}`\n"
        f"- **Python:** `{metadata.get('python_version', 'Aktif')}`\n"
        f"- **Biomarker Terpilih:** `{len(selected_features)}` dari `{metadata.get('total_initial_features', 100)}`\n"
        f"- **Explainable AI:** `SHAP (KernelExplainer)`\n"
        f"- **Ambang Batas (Threshold):** `{threshold * 100:.0f}%`\n"
        f"- **Ekspor PDF:** `Tersedia (ReportLab)`\n"
        f"- **Sinkronisasi Terakhir:** `{metadata.get('created_at', 'Tersedia')}`"
    )
    st.divider()

    st.subheader("🛠️ Pengelolaan Model")
    if st.button("🔄 Latih Ulang Model (Force Retrain)", help="Paksa retraining pipeline SVM-RFE & Optuna dan bersihkan cache"):
        with st.spinner("Menjalankan retraining pipeline SVM-RFE & Optuna..."):
            st.cache_resource.clear()
            st.cache_data.clear()
            load_cdss_artifact(force_retrain=True)
            st.success("Retraining selesai! Memuat ulang sistem...")
            st.rerun()


# -----------------------------------------------------------------------------
# Header Utama Aplikasi
# -----------------------------------------------------------------------------
st.title("🧬 OncoDecision AI: Clinical Decision Support System")
st.markdown(
    "**Sistem Deteksi Dini Kanker Berbasis Seleksi Fitur SVM-RFE (Guyon et al., 2002), "
    "Optimasi Hyperparameter (Optuna), & Formal Explainable AI (SHAP Lundberg & Lee)**"
)
st.caption("Penyusun: Nayyara Rauf Ghani Al Khayri (NIM: A11.2024.16055) | Dosen Pembimbing: Abu Salam, M.Kom")
st.divider()


# -----------------------------------------------------------------------------
# Tab Navigasi Fitur CDSS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🩺 Skrining Klinis Pasien (Single Input)",
    "📂 Skrining Rekam Medis Massal (Batch CSV)",
    "📊 Komparasi Eksperimen Paper (Guyon et al., 2002)"
])


# -----------------------------------------------------------------------------
# TAB 1: Skrining Klinis Pasien Individual, Analisis SHAP & Ekspor PDF
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Formulir Input Biomarker Terpilih (Hasil Reduksi Dimensi SVM-RFE)")
    st.info(
        f"Dari **{metadata['total_initial_features']} fitur berdimensi tinggi**, algoritma **SVM-RFE** "
        f"berhasil menyaring **{len(selected_features)} biomarker paling diskriminatif** (bebas redundansi)."
    )

    col_id, _ = st.columns([1, 2])
    with col_id:
        patient_id = st.text_input(
            "📋 No. Rekam Medis / ID Sampel Pasien:",
            value="ONCO-2026-X8921",
            help="Nomor identitas unik pasien yang akan dicantumkan pada berkas PDF rekam medis resmi"
        )

    cols = st.columns(3)
    user_inputs = {}
    for idx, feat_name in enumerate(selected_features):
        col_mean = float(df_raw[feat_name].mean())
        col_min = float(df_raw[feat_name].min())
        col_max = float(df_raw[feat_name].max())
        with cols[idx % 3]:
            user_inputs[feat_name] = st.number_input(
                f"{feat_name}",
                min_value=col_min * 0.5 if col_min > 0 else col_min * 2.0,
                max_value=col_max * 1.5 if col_max > 0 else col_max * 0.5,
                value=col_mean,
                format="%.4f"
            )

    if st.button("🔍 Jalankan Diagnosis Prediktif CDSS", type="primary"):
        t_start = time.perf_counter()

        # Normalisasi Guyon et al. (2002): Zero-mean, unit-variance, dilanjutkan squashing non-linear
        x_selected_norm = np.zeros((1, len(selected_features)))
        for i, feat_idx in enumerate(selected_indices):
            feat_name = selected_features[i]
            val_norm = (user_inputs[feat_name] - mean_vec[feat_idx]) / std_vec[feat_idx]
            x_selected_norm[0, i] = 2.0 * np.arctan(val_norm / 2.0)

        prob_malignant = float(model.predict_proba(x_selected_norm)[0, 1])
        is_malignant = prob_malignant >= threshold

        # Hitung atribusi formal SHAP (Lundberg & Lee) untuk sampel pasien
        shap_exp = explainer.explain_instance(x_selected_norm[0])
        shap_values = shap_exp.values
        latency = time.perf_counter() - t_start

        # Tentukan biomarker pemicu terbesar (top risk driver) via SHAP
        top_risk_drivers = explainer.get_top_risk_drivers(shap_values, top_k=3)
        if top_risk_drivers:
            top_feature = f"{top_risk_drivers[0]['feature']} (+{top_risk_drivers[0]['shap_value']*100:.1f}%)"
        else:
            max_idx = int(np.argmax(shap_values))
            top_feature = f"{selected_features[max_idx]} ({shap_values[max_idx]*100:.1f}%)"

        # Hasilkan grafik SHAP Waterfall Plot dan simpan ke buffer gambar untuk PDF
        fig_waterfall = explainer.plot_waterfall(shap_exp)
        chart_buf = io.BytesIO()
        fig_waterfall.savefig(
            chart_buf,
            format="png",
            bbox_inches="tight",
            dpi=150,
            facecolor=fig_waterfall.get_facecolor()
        )
        chart_buf.seek(0)
        chart_bytes = chart_buf.getvalue()

        # Bangun Narasi Klinis
        narrative_md = explainer.generate_clinical_narrative(
            shap_values=shap_values,
            patient_prob=prob_malignant,
            threshold=threshold
        )

        # Bangun Berkas Rekam Medis PDF Resmi
        pdf_report_bytes = generate_clinical_pdf_report(
            patient_data=user_inputs,
            diagnosis=is_malignant,
            probability=prob_malignant,
            top_features=top_risk_drivers,
            chart_img=chart_bytes,
            sample_id=patient_id,
            hospital_name="Pusat Diagnostik Onkologi & Patologi Terpadu (RS Akademik)",
            threshold=threshold,
            df_raw=df_raw,
            clinical_notes=narrative_md
        )

        st.divider()
        c1, c2 = st.columns([1, 1.25])
        with c1:
            render_workstation_card(
                is_malignant=is_malignant,
                prob=prob_malignant,
                latency=latency,
                top_feature=top_feature
            )

            # Tombol Ekspor PDF Rekam Medis Resmi
            st.download_button(
                label="📄 Ekspor Ringkasan Diagnosis ke PDF",
                data=pdf_report_bytes,
                file_name=f"Rekam_Medis_OncoDecision_{patient_id}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                help="Unduh berkas PDF rekam medis resmi lengkap dengan parameter biomarker dan analisis SHAP"
            )

        with c2:
            st.markdown(r"### Explainable AI: Atribusi Fitur Pasien SHAP ($E[f(X)] \to f(x)$)")
            viz_tab1, viz_tab2 = st.tabs(["📊 SHAP Waterfall Plot", "🌊 Interactive Force Plot"])
            with viz_tab1:
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)

            with viz_tab2:
                force_html = explainer.plot_force_html(shap_exp)
                components.html(force_html, height=170, scrolling=True)

        st.markdown("---")
        st.markdown(narrative_md)


# -----------------------------------------------------------------------------
# TAB 2: Skrining Massal Rekam Medis (Batch CSV)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Skrining Massal Rekam Medis Tabular (.CSV)")
    st.write("Unggah file CSV yang memuat kolom biomarker untuk skrining otomatis.")

    sample_csv = generate_sample_csv(df_raw, tuple(selected_features))
    st.download_button(
        "📥 Unduh Contoh Template CSV Rekam Medis (10 Pasien)",
        data=sample_csv,
        file_name="template_pasien_kanker.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Pilih file CSV rekam medis", type=["csv"])
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        missing_cols = [c for c in selected_features if c not in df_batch.columns]
        if missing_cols:
            st.error(f"Kolom biomarker berikut belum ada di CSV Anda: {missing_cols}")
        else:
            X_batch_norm = np.zeros((len(df_batch), len(selected_features)))
            for i, feat_idx in enumerate(selected_indices):
                fname = selected_features[i]
                v_norm = (df_batch[fname].values - mean_vec[feat_idx]) / std_vec[feat_idx]
                X_batch_norm[:, i] = 2.0 * np.arctan(v_norm / 2.0)

            probs = model.predict_proba(X_batch_norm)[:, 1]
            preds = np.where(probs >= threshold, "MALIGNANT (Kanker)", "BENIGN (Jinak)")

            df_result = df_batch.copy()
            df_result.insert(0, "Keputusan_CDSS", preds)
            df_result.insert(1, "Probabilitas_Kanker_Pct", (probs * 100).round(2))

            st.success(f"Berhasil memproses {len(df_result)} sampel pasien!")
            st.dataframe(df_result, use_container_width=True)

            out_csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Ekspor Hasil Diagnosis Massal (.CSV)",
                data=out_csv,
                file_name="hasil_diagnosis_oncodecision.csv",
                mime="text/csv"
            )


# -----------------------------------------------------------------------------
# TAB 3: Komparasi Eksperimen Metodologi Paper (Guyon et al., 2002)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Bukti Empiris: Baseline Golub (1999) vs SVM-RFE Guyon (2002) + Optuna")
    b_met = metadata["baseline_golub_metrics"]
    p_met = metadata["proposed_svmrfe_optuna_metrics"]

    comp_df = pd.DataFrame({
        "Metrik Evaluasi Klinis": [
            "Sensitivitas / Recall (%) [Target Utama]",
            "Jumlah False Negative (Pasien Kanker Lolos) [Wajib Minim]",
            "F2-Score (Penalti 2x pada FN)",
            "PR-AUC",
            "ROC-AUC",
            "Akurasi (%)",
            "Median Margin (M/D - Guyon 2002)"
        ],
        "Baseline: Korelasi Golub (1999) + Default SVM": [
            f"{b_met['recall_sensitivity']*100:.2f}%",
            f"{b_met['false_negatives']} pasien",
            f"{b_met['f2_score']:.4f}",
            f"{b_met['pr_auc']:.4f}",
            f"{b_met['roc_auc']:.4f}",
            f"{b_met['accuracy']*100:.2f}%",
            f"{b_met['median_margin']:.4f}"
        ],
        "Usulan: SVM-RFE (Guyon 2002) + Optuna HPO": [
            f"{p_met['recall_sensitivity']*100:.2f}%",
            f"{p_met['false_negatives']} pasien",
            f"{p_met['f2_score']:.4f}",
            f"{p_met['pr_auc']:.4f}",
            f"{p_met['roc_auc']:.4f}",
            f"{p_met['accuracy']*100:.2f}%",
            f"{p_met['median_margin']:.4f}"
        ]
    })
    st.table(comp_df)

    st.markdown(r"#### Peringkat Biomarker Terpilih berdasarkan Kriteria Bobot SVM-RFE $c_i = (w_i)^2$")
    rfe_df = pd.DataFrame({
        "Nama Biomarker": selected_features,
        "Kriteria Bobot (w_i^2)": artifact["rfe_weights_squared"]
    }).sort_values(by="Kriteria Bobot (w_i^2)", ascending=False)
    st.bar_chart(rfe_df.set_index("Nama Biomarker"))
