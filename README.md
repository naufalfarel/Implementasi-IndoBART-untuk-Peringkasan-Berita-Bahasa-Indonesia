# Proyek UAS: Abstractive Text Summarization Berita Kesehatan Indonesia

**Mata Kuliah:** Pemrosesan Bahasa Alami  
**Model:** IndoBART fine-tuned vs IndoBART Base (zero-shot)  
**Dataset:** IndoSum (subset kesehatan)  
**Evaluasi:** ROUGE-1, ROUGE-2, ROUGE-L + Human Evaluation

---

## Setup Cepat

```bash
# 1. Buat virtual environment
python -m venv .venv

# 2. Aktifkan virtual environment
# Di Windows PowerShell:
.venv\Scripts\Activate.ps1
# Atau di Linux/macOS:
# source .venv/bin/activate

# 3. Install dependensi
pip install -r app/requirements.txt

# 4. Buka notebook di VS Code
# (ekstensi Jupyter harus sudah terinstal)
# Buka: notebook/IndoBART_Summarization_UAS.ipynb

# 5. Setelah training selesai, jalankan aplikasi demo
streamlit run app/app.py
```

## Isi Proyek

- `notebook/` — Notebook Jupyter utama (9 bagian, dari setup hingga evaluasi)
- `app/app.py` — Aplikasi demo Streamlit (input URL atau teks)
- `docs/PANDUAN_PROYEK.md` — Panduan lengkap, troubleshooting, dan pertanyaan UAS
- `outputs/` — Hasil training (dibuat otomatis saat notebook dijalankan)

Lihat `docs/PANDUAN_PROYEK.md` untuk panduan lengkap.
