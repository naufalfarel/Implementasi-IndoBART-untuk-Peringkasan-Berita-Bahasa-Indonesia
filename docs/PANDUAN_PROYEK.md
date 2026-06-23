# Panduan Proyek: Summarization Berita Kesehatan Indonesia

**Mata Kuliah:** Pemrosesan Bahasa Alami  
**Task:** Abstractive Text Summarization  
**Model:** IndoBART (indobenchmark/indobart-v2)  
**Dataset:** IndoSum (subset kesehatan)

---

## Struktur Direktori

```
nlp-summarization/
├── notebook/
│   └── IndoBART_Summarization_UAS.ipynb   -- Notebook utama (jalankan ini)
├── app/
│   ├── app.py                             -- Aplikasi demo Streamlit
│   └── requirements.txt                   -- Dependensi Python
├── docs/
│   └── PANDUAN_PROYEK.md                  -- File ini
└── outputs/                               -- Hasil training (otomatis dibuat)
    ├── train_health.csv
    ├── val_health.csv
    ├── test_health.csv
    ├── rouge_comparison.csv
    ├── qualitative_results.csv
    ├── human_eval_template.csv
    ├── eksplorasi_dataset.png
    ├── rouge_comparison.png
    ├── human_eval_chart.png
    └── indobart_finetuned/                -- Checkpoint model
```

---

## Persiapan Lingkungan (VS Code)

### 1. Buat virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install dependensi

```bash
pip install -r app/requirements.txt
```

### 3. Buka notebook di VS Code

Pastikan ekstensi **Jupyter** sudah terinstal di VS Code.  
Pilih kernel: `venv` yang baru dibuat.

Buka file: `notebook/IndoBART_Summarization_UAS.ipynb`

### 4. Jalankan aplikasi demo

```bash
streamlit run app/app.py
```

Setelah model selesai di-training, isi kolom **Path model** di sidebar dengan:

```
outputs/indobart_finetuned
```

---

## Urutan Pengerjaan

| Langkah | Keterangan | Estimasi Waktu |
|---|---|---|
| 1 | Setup environment dan install packages | 10-15 menit |
| 2 | Jalankan notebook Bagian 1-3 (setup, dataset, preprocessing) | 10-20 menit |
| 3 | Jalankan Bagian 4 (load model) | 5-10 menit |
| 4 | Jalankan Bagian 5 (fine-tuning) | 30-90 menit (GPU) |
| 5 | Jalankan Bagian 6 (evaluasi ROUGE) | 10-15 menit |
| 6 | Jalankan Bagian 7-8 (analisis kualitatif, inferensi URL) | 10 menit |
| 7 | Isi template human evaluation (`outputs/human_eval_template.csv`) | Sesuai evaluator |
| 8 | Update nilai human evaluation di Bagian 7 notebook, jalankan ulang | 5 menit |
| 9 | Jalankan aplikasi Streamlit untuk live demo | 5 menit |

---

## Catatan Teknis

### GPU vs CPU

Training di CPU membutuhkan waktu jauh lebih lama (bisa 6-12 jam untuk 3 epoch).  
Jika tidak punya GPU lokal, gunakan Google Colab dengan mengubah:

- Hapus cell Bagian 1 (instalasi dengan subprocess)
- Ganti cell setup direktori (`SAVE_DIR`) menjadi:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  SAVE_DIR = pathlib.Path('/content/drive/MyDrive/NLP_UAS')
  ```
- Tambahkan `!` di depan perintah pip install

### Dataset IndoSum

Jika dataset gagal diunduh via HuggingFace, gunakan cell alternatif (Bagian 2, Cell 7):

```bash
wget https://raw.githubusercontent.com/kata-ai/indosum/master/data/train.01.jsonl
wget https://raw.githubusercontent.com/kata-ai/indosum/master/data/dev.01.jsonl
wget https://raw.githubusercontent.com/kata-ai/indosum/master/data/test.01.jsonl
```

### Subset kesehatan

IndoSum mencakup berita umum dari Liputan6.  
Jika filter kategori kesehatan menghasilkan kurang dari 200 sampel, notebook secara otomatis menggunakan sample acak dari seluruh dataset (fallback ke 2000 sampel train).  
Ini valid untuk tujuan demonstrasi UAS.

### Mixed precision (fp16)

Parameter `fp16=True` hanya aktif jika GPU tersedia.  
Notebook sudah mendeteksi ini secara otomatis dengan kondisi `fp16=(device == "cuda")`.

---

## Referensi

| Sumber | Keterangan |
|---|---|
| Kurniawan & Louvan (2018) | Paper IndoSum — benchmark resmi |
| Cahyawijaya et al. (2021) | IndoBART dan IndoNLG benchmark |
| Lin (2004) | Pengenalan metrik ROUGE |
| Howard & Ruder (2018) | Justifikasi learning rate fine-tuning |
| Vinyals et al. (2015) | Beam search dalam sequence-to-sequence |

---

## Pertanyaan Umum dari Dosen

**Q: Mengapa IndoBART dan bukan mBART-50?**  
A: IndoBART lebih ringan dan dilatih secara khusus pada korpus bahasa Indonesia, sehingga lebih efisien dan lebih relevan untuk task monolingual. mBART-50 didesain untuk multilingual translation.

**Q: Mengapa ROUGE dan bukan hanya human evaluation?**  
A: ROUGE memungkinkan perbandingan otomatis dan reproducible. Human evaluation melengkapi karena ROUGE tidak menangkap kualitas bahasa dan kesetiaan semantik secara penuh.

**Q: Apakah 3 epoch cukup?**  
A: Untuk dataset ~2000 sampel dan tujuan demonstrasi, 3 epoch dengan early stopping sudah memadai. Tren loss pada setiap epoch dapat dilihat di log training.

**Q: Apa perbedaan extractive vs abstractive summarization?**  
A: Extractive memilih kalimat penting langsung dari teks asli. Abstractive menghasilkan kalimat baru yang merangkum konten — lebih natural namun lebih sulit dilatih.

**Q: Mengapa gradient_accumulation_steps=4?**  
A: Dengan batch size 4 per device dan accumulation 4, efektif batch size menjadi 16 tanpa membutuhkan lebih banyak VRAM. Ini teknik standar untuk training dengan GPU terbatas.

**Q: Bagaimana cara membuktikan fine-tuning membantu?**  
A: Perbandingan ROUGE antara base model (zero-shot) dan fine-tuned model menunjukkan peningkatan kuantitatif. Analisis kualitatif di Bagian 7 menunjukkan perbedaan kualitas secara visual.
