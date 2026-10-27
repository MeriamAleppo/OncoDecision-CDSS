"""
Modul Explainable AI (XAI) Formal Berbasis SHAP (Lundberg & Lee)
Clinical Decision Support System - OncoDecision AI
Penyusun: Nayyara Rauf Ghani Al Khayri (NIM: A11.2024.16055)
Dosen Pembimbing: Abu Salam, M.Kom
"""

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap


class CDSSShapExplainer:
    """
    Wrapper SHAP Formal untuk Model Klasifikasi Klinis OncoDecision AI.
    Menggunakan KernelExplainer yang dioptimalkan dengan representasi data latar
    (background dataset) menggunakan clustering k-means agar komputasi atribusi
    Shapley berlangsung cepat (< 0.05 detik) untuk inferensi interaktif waktu-nyata.
    """

    def __init__(
        self,
        model,
        background_data: np.ndarray,
        feature_names: list[str],
        n_kmeans: int = 10,
        random_state: int = 42
    ):
        self.model = model
        self.feature_names = list(feature_names)

        # Kompresi data latar menggunakan shap.kmeans untuk efisiensi komputasi
        if len(background_data) > n_kmeans:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.background_summary = shap.kmeans(background_data, n_kmeans)
        else:
            self.background_summary = background_data

        # Predict function untuk probabilitas kelas positif (Malignant, indeks 1)
        self.predict_fn = lambda x: self.model.predict_proba(x)[:, 1]

        # Inisialisasi KernelExplainer model-agnostic
        self.explainer = shap.KernelExplainer(self.predict_fn, self.background_summary)
        
        # Expected value (probabilitas risiko rata-rata populasi latar)
        raw_exp = self.explainer.expected_value
        if isinstance(raw_exp, (list, np.ndarray)):
            self.expected_value = float(raw_exp[1] if len(raw_exp) > 1 else raw_exp[0])
        else:
            self.expected_value = float(raw_exp)

    def explain_instance(self, x_patient_norm: np.ndarray, nsamples: int = 100) -> shap.Explanation:
        """
        Menghitung nilai SHAP untuk satu sampel pasien ternormalisasi.

        Parameters
        ----------
        x_patient_norm : np.ndarray
            Array 1D atau 2D (1, n_features) nilai fitur pasien hasil normalisasi Guyon.
        nsamples : int
            Jumlah permutasi sampling koalisi untuk estimasi Shapley kernel.

        Returns
        -------
        shap.Explanation
            Objek penjelasan formal SHAP yang memuat nilai atribusi, base value, dan data.
        """
        x_2d = np.atleast_2d(x_patient_norm)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            raw_shap = self.explainer.shap_values(x_2d, nsamples=nsamples, silent=True)

        # Ekstrak array 1D
        if isinstance(raw_shap, list):
            # Jika mengembalikan list per kelas
            vals = np.array(raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]).squeeze()
        else:
            vals = np.array(raw_shap).squeeze()

        data_1d = x_2d[0].copy()

        explanation = shap.Explanation(
            values=vals,
            base_values=self.expected_value,
            data=data_1d,
            feature_names=self.feature_names
        )
        return explanation

    def get_top_risk_drivers(self, shap_values: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Mengembalikan daftar biomarker yang memberikan dorongan positif terbesar
        terhadap risiko keganasan (malignant risk drivers).
        """
        sorted_indices = np.argsort(shap_values)[::-1]
        positive_drivers = []
        for idx in sorted_indices:
            if shap_values[idx] > 0:
                positive_drivers.append({
                    "feature": self.feature_names[idx],
                    "shap_value": float(shap_values[idx]),
                    "index": int(idx)
                })
                if len(positive_drivers) == top_k:
                    break
        return positive_drivers

    def generate_clinical_narrative(
        self,
        shap_values: np.ndarray,
        patient_prob: float,
        threshold: float = 0.30
    ) -> str:
        """
        Menghasilkan narasi klinis otomatis berbasis aturan onkologi berdasarkan nilai SHAP.
        Menyebutkan secara eksplisit 3 biomarker spesifik yang paling menaikkan risiko
        keganasan serta memberikan interpretasi klinis terhadap ambang batas sensitivitas.
        """
        sorted_desc = np.argsort(shap_values)[::-1]
        positive_indices = [i for i in sorted_desc if shap_values[i] > 0]
        negative_indices = [i for i in np.argsort(shap_values) if shap_values[i] < 0]

        prob_pct = patient_prob * 100
        base_pct = self.expected_value * 100
        shift_pct = prob_pct - base_pct

        lines = []
        lines.append("### Narasi Klinis Otomatis Berbasis Aturan (SHAP Diagnostic Narrative)")

        if positive_indices:
            top_indices = positive_indices[:3]
            top_items = [
                f"**{self.feature_names[i]}** (+{shap_values[i]*100:.2f}%)"
                for i in top_indices
            ]

            if len(top_items) == 3:
                biomarkers_mention = f"{top_items[0]}, {top_items[1]}, dan {top_items[2]}"
            elif len(top_items) == 2:
                biomarkers_mention = f"{top_items[0]} dan {top_items[1]}"
            else:
                biomarkers_mention = f"{top_items[0]}"

            lines.append(
                f"Model CDSS mengestimasi probabilitas risiko keganasan pasien sebesar **{prob_pct:.1f}%**, "
                f"mengalami deviasi {'+' if shift_pct >= 0 else ''}{shift_pct:.1f}% terhadap probabilitas acuan "
                f"populasi latar (*baseline* $E[f(X)] = {base_pct:.1f}\\%$). "
                f"Berdasarkan formalisasi atribusi Shapley (Lundberg & Lee), terdapat **{len(positive_indices)} biomarker** "
                f"yang secara aktif mendorong kenaikan risiko, dengan **3 biomarker pemicu utama (*Top 3 Risk Drivers*)** adalah: "
                f"{biomarkers_mention}."
            )

            lines.append("\n**Dampak Marginal Biomarker Pemicu Terhadap Risiko Kanker:**")
            for rank, idx in enumerate(top_indices, 1):
                feat_name = self.feature_names[idx]
                val = shap_values[idx]
                lines.append(
                    f"{rank}. **{feat_name}:** Memberikan dorongan risiko sebesar **+{val*100:.2f}%** "
                    f"ke arah keputusan *malignant*."
                )

            if patient_prob >= threshold:
                lines.append(
                    f"\n> **Rekomendasi Tindakan Onkologi:** Akumulasi dorongan biomarker di atas "
                    f"telah melampaui ambang batas kalibrasi sensitivitas tinggi ({threshold*100:.0f}%). "
                    f"Diindikasikan kecurigaan tinggi terhadap lesi ganas (*malignant*). "
                    f"Disarankan untuk segera menjadwalkan pemeriksaan histopatologi definitif (*Core Needle Biopsy*) "
                    f"serta pemindaian radiologis komplementer."
                )
            else:
                lines.append(
                    f"\n> **Catatan Klinis:** Meskipun terdapat peningkatan risiko parsial dari {top_items[0]}, "
                    f"kontribusi penekan risiko dari biomarker lain berhasil mempertahankan probabilitas akhir "
                    f"di bawah ambang batas risiko ({threshold*100:.0f}%). Pasien diindikasikan jinak (*benign*), "
                    f"disarankan observasi dan pemantauan klinis berkala (*surveillance* rutin)."
                )
        else:
            # Tidak ada biomarker yang bernilai positif (semua bernilai protektif/jinak)
            top_prot = [
                f"**{self.feature_names[i]}** ({shap_values[i]*100:.2f}%)"
                for i in negative_indices[:3]
            ]
            prot_str = ", ".join(top_prot)
            lines.append(
                f"Model CDSS mengestimasi risiko keganasan sangat rendah sebesar **{prob_pct:.1f}%** "
                f"(jauh di bawah baseline $E[f(X)] = {base_pct:.1f}\\%$). Tidak ditemukan biomarker yang "
                f"mendorong peningkatan risiko kanker. Faktor protektif yang paling menekan risiko pada pasien ini "
                f"adalah {prot_str}."
            )
            lines.append(
                f"\n> **Rekomendasi Tindakan Onkologi:** Profil biomarker konsisten dengan lesi jinak (*benign*). "
                f"Tidak terdapat indikasi untuk tindakan invasif segera."
            )

        return "\n".join(lines)

    def plot_waterfall(self, explanation: shap.Explanation, max_display: int = 12) -> plt.Figure:
        """
        Menghasilkan visualisasi SHAP Waterfall Plot berkualitas tinggi dengan estetika dark theme.
        """
        plt.close("all")
        fig = plt.figure(figsize=(9, 5.2))
        
        # Dark mode styling serasi dengan workstation card
        fig.patch.set_facecolor("#0f172a")

        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        ax = plt.gca()
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        ax.set_xlabel("E[f(X)] -> f(x) (Kontribusi Probabilitas Keganasan)", color="#e2e8f0", fontsize=10)

        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(axis="x", color="#334155", linestyle=":", alpha=0.6)

        # Ubah warna teks label fitur jika memungkinkan
        for text in ax.texts:
            text.set_color("#f8fafc")

        plt.tight_layout()
        return fig

    def plot_force_html(self, explanation: shap.Explanation) -> str:
        """
        Menghasilkan kode HTML interaktif SHAP Force Plot lengkap dengan pustaka JavaScript.
        """
        force_plot = shap.force_plot(
            explanation.base_values,
            explanation.values,
            explanation.data,
            feature_names=explanation.feature_names,
            matplotlib=False
        )
        html_content = f"""
        <html>
        <head>
            {shap.getjs()}
            <style>
                body {{
                    background-color: #0f172a !important;
                    color: #f8fafc !important;
                    margin: 0;
                    padding: 8px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }}
            </style>
        </head>
        <body>
            <div style="background-color: #0f172a; padding: 10px; border-radius: 8px;">
                {force_plot.html()}
            </div>
        </body>
        </html>
        """
        return html_content
