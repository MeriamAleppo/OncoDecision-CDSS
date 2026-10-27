# 🧬 OncoDecision AI: Clinical Decision Support System (CDSS)

**Rancang Bangun Clinical Decision Support System Deteksi Kanker Menggunakan Seleksi Fitur Dimensi Tinggi (SVM-RFE) dan Optimasi Hyperparameter untuk Meminimalkan False Negative**

- **Penyusun:** Nayyara Rauf Ghani Al Khayri (NIM: A11.2024.16055)
- **Program Studi:** Teknik Informatika
- **Dosen Pembimbing:** Abu Salam, M.Kom
- **Landasan Utama:** Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). *Gene Selection for Cancer Classification using Support Vector Machines*. Machine Learning, 46, 389–422.

---

## 📁 Struktur Direktori Proyek
```text
OncoDecision-CDSS/
├── data/
│   ├── raw/                      # Dataset klinis + 70 gen dimensi tinggi (100 fitur)
│   └── processed/                # Dataset tereduksi hasil seleksi SVM-RFE (12 fitur inti)
├── models/                       # Artefak model terlatih (.joblib) & hasil metrik JSON
├── notebooks/                    # Notebook eksperimen riset komparatif
├── src/                          # Source code modular (data_loader, feature_selection, optimizer, trainer)
├── tests/                        # Unit testing (Pytest)
├── app.py                        # Aplikasi Web CDSS Berbasis Streamlit ("Barang" Produk TA)
├── requirements.txt              # Daftar pustaka Python
└── README.md
```

---

## 🚀 Cara Menjalankan di VS Code (Windows PowerShell)

1. **Ekstrak file ZIP ini dan buka folder `OncoDecision-CDSS` di VS Code.**
2. **Buat dan aktifkan Virtual Environment (opsional jika belum ada):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. **Jalankan Pipeline Pelatihan (Opsional - Model sudah otomatis dilatih di dalam paket ini):**
   ```powershell
   python -m src.trainer
   ```
4. **Jalankan Unit Testing:**
   ```powershell
   pytest -v
   ```
5. **Jalankan Aplikasi Web CDSS:**
   ```powershell
   streamlit run app.py
   ```
