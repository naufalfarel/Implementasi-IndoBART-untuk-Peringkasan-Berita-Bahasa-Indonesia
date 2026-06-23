"""
Aplikasi Demo: Summarizer Berita Kesehatan Indonesia
Model: IndoBART fine-tuned (checkpoint terbaik: checkpoint-625, ROUGE-L 13.15)

Cara menjalankan:
    venv\Scripts\streamlit run app\app.py
"""

import re
import requests
from pathlib import Path

import streamlit as st
import torch
from bs4 import BeautifulSoup
from transformers import MBartForConditionalGeneration, MBart50Tokenizer


# ==============================================================================
#  Konfigurasi halaman
# ==============================================================================

st.set_page_config(
    page_title="Summarizer Berita Kesehatan Indonesia",
    page_icon="🧬",
    layout="wide",
)

# Inject custom modern CSS
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
        
        /* Font Family */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* Top Accent Bar (Medical Red) */
        .stApp {
            border-top: 5px solid #E11D48;
        }
        
        /* Titles & Headings */
        h1 {
            font-weight: 700 !important;
            letter-spacing: -0.025em;
            color: #1E293B;
            margin-bottom: 0.25rem !important;
        }
        
        .subtitle-text {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        
        /* Textarea inputs */
        div.stTextArea textarea {
            border-radius: 8px !important;
            font-size: 0.95rem !important;
            transition: all 0.2s ease-in-out !important;
        }
        
        div.stTextArea textarea:focus {
            border-color: #E11D48 !important;
            box-shadow: 0 0 0 3px rgba(225, 29, 72, 0.15) !important;
        }
        
        /* Primary Button (Medical Red) */
        div.stButton button {
            background-color: #E11D48 !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 1.5rem !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        }
        
        div.stButton button:hover {
            background-color: #BE123C !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 6px -1px rgba(225, 29, 72, 0.1), 0 2px 4px -1px rgba(225, 29, 72, 0.06) !important;
        }
        
        div.stButton button:active {
            transform: translateY(0) !important;
        }
        
        /* Custom Tab styling */
        button[data-baseweb="tab"] {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 500 !important;
            transition: all 0.2s !important;
        }
        
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #E11D48 !important;
            border-bottom-color: #E11D48 !important;
            font-weight: 600 !important;
        }
        
        /* Sidebar styling override */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(0,0,0,0.05);
        }
        
        /* Footer text styling */
        .footer-text {
            text-align: center;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid rgba(0,0,0,0.05);
            color: #94A3B8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Summarizer Berita Kesehatan Indonesia")
st.markdown(
    '<div class="subtitle-text">Meringkas artikel berita kesehatan panjang menjadi ringkasan yang padat dan akurat menggunakan model IndoBART fine-tuned.</div>',
    unsafe_allow_html=True,
)
st.divider()


# ==============================================================================
#  Load model — langsung dari checkpoint-625 (best checkpoint)
# ==============================================================================

# Path ke checkpoint-625: model terbaik berdasarkan trainer_state.json
# best_metric: 13.15 (ROUGE-L), epoch 2.0
DEFAULT_MODEL_PATH = "notebook/outputs/indobart_finetuned/checkpoint-625"


@st.cache_resource(show_spinner="⏳ Memuat model IndoBART fine-tuned...")
def load_model(model_path: str):
    """
    Load model dan tokenizer IndoBART dari checkpoint lokal.
    Menggunakan checkpoint-625 yang merupakan best checkpoint (ROUGE-L 13.15).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer: gunakan MBart50Tokenizer (bukan Fast) untuk menghindari
    # bug truncation huruf kapital pada IndoNLGTokenizer/sentencepiece
    tokenizer = MBart50Tokenizer.from_pretrained(model_path)
    tokenizer.src_lang = "id_ID"

    model = MBartForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    return model, tokenizer, device


# Sidebar: informasi model & pengaturan path
st.sidebar.header("Konfigurasi Model")

MODEL_PATH = st.sidebar.text_input(
    "Path model (checkpoint)",
    value=DEFAULT_MODEL_PATH,
    help="Path ke folder checkpoint. Default: checkpoint-625 (best model, ROUGE-L 13.15).",
)

try:
    model, tokenizer, device = load_model(MODEL_PATH)
    st.sidebar.caption(f"Status: Model dimuat (Device: {device.upper()})")
except Exception as e:
    st.sidebar.error(f"Gagal memuat model:\n{e}")
    st.stop()


# ==============================================================================
#  Parameter generasi
# ==============================================================================

st.sidebar.header("Parameter Generasi")

num_beams = st.sidebar.slider(
    "Jumlah beam (beam search)",
    min_value=1, max_value=8, value=4,
    help="Semakin tinggi: kualitas lebih baik, kecepatan lebih lambat.",
)
max_summary_len = st.sidebar.slider(
    "Panjang maksimum ringkasan (token)",
    min_value=50, max_value=200, value=128,
)
no_repeat_ngram = st.sidebar.slider(
    "No-repeat n-gram size",
    min_value=0, max_value=5, value=3,
    help="Mencegah pengulangan frasa. Set 0 untuk nonaktifkan.",
)


# ==============================================================================
#  Fungsi utilitas
# ==============================================================================

def clean_text(text: str) -> str:
    """Normalisasi teks: whitespace, URL, dan karakter tidak terlihat."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text.strip()


def scrape_article(url: str) -> str:
    """
    Ambil konten teks dari URL artikel berita.
    Mendukung Kompas, Detik, Liputan6, dan situs dengan struktur umum.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.content, "html.parser")

        selectors = [
            "div.read__content",
            "div.itp_bodycontent",
            "div.article-content-body__item-page",
            "article",
            "div.article-body",
            "div.post-content",
            "div.entry-content",
        ]

        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                paragraphs = content.find_all("p")
                text = " ".join([p.get_text() for p in paragraphs])
                if len(text) > 100:
                    return clean_text(text)

        # Fallback: semua tag <p>
        all_p = soup.find_all("p")
        return clean_text(" ".join([p.get_text() for p in all_p]))

    except Exception as e:
        return f"Error: {e}"


def clean_bias_prefixes(text: str) -> str:
    """Membersihkan awalan bias dataset (seperti 'buah penelitian') di awal ringkasan."""
    patterns = [
        r'^buah penelitian tentang\s+',
        r'^buah penelitian\s+',
        r'^buah laporan tentang\s+',
        r'^buah laporan\s+',
        r'^buah artikel tentang\s+',
        r'^buah artikel\s+',
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def capitalize_sentences(text: str) -> str:
    """Mengkapitalisasi huruf pertama setiap kalimat dan nama diri umum."""
    if not text:
        return ""
    
    text = re.sub(r"\s+", " ", text).strip()
    text = text[0].upper() + text[1:] if len(text) > 0 else text
    
    def repl(match):
        return match.group(1) + match.group(2).upper()
    text = re.sub(r'([.!?]\s+)([a-z])', repl, text)
    
    # Daftar kata penyesuaian kapitalisasi yang umum
    words_to_capitalize = {
        "jakarta": "Jakarta",
        "indonesia": "Indonesia",
        "kompas": "Kompas",
        "detik": "Detik",
        "liputan6": "Liputan6",
        "kemenkes": "Kemenkes",
        "dbd": "DBD",
        "idai": "IDAI",
        "papdi": "PAPDI",
        "takeda": "Takeda",
        "aedes aegypti": "Aedes aegypti",
        "aedes": "Aedes"
    }
    for word, cap_word in words_to_capitalize.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', cap_word, text, flags=re.IGNORECASE)
        
    return text


def generate_summary(text: str) -> str:
    """Generate ringkasan dari teks artikel menggunakan beam search."""
    # IndoBART ditraining eksklusif pada teks huruf kecil (lowercase).
    # Input wajib di-lowercase untuk menghindari output sampah (byte fallbacks).
    lowercased_text = text.lower()
    
    inputs = tokenizer(
        lowercased_text,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_summary_len,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram if no_repeat_ngram > 0 else None,
            forced_bos_token_id=tokenizer.lang_code_to_id["id_ID"],
        )

    raw_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Bersihkan prefiks bias dataset dan rapikan kapitalisasi kalimat
    cleaned_summary = clean_bias_prefixes(raw_summary)
    return capitalize_sentences(cleaned_summary)


def compression_ratio(original: str, summary: str) -> float:
    """Hitung rasio kompresi (panjang ringkasan / panjang artikel)."""
    orig_len = len(original.split())
    summ_len = len(summary.split())
    if orig_len == 0:
        return 0.0
    return round((summ_len / orig_len) * 100, 1)


# ==============================================================================
#  Antarmuka utama: dua tab
# ==============================================================================

tab_url, tab_text = st.tabs(["Input URL Artikel", "Input Teks Langsung"])

# ── Tab 1: URL
with tab_url:
    st.subheader("Ringkas dari URL Artikel")
    st.caption("Tempelkan URL artikel berita kesehatan (Kompas, Detik, Liputan6, dll.)")
    url_input = st.text_input(
        "URL artikel",
        placeholder="https://health.kompas.com/...",
        label_visibility="collapsed",
    )

    if st.button("Ambil dan Ringkas", key="btn_url", type="primary"):
        if not url_input.strip():
            st.warning("Masukkan URL terlebih dahulu.")
        else:
            with st.spinner("Mengambil konten artikel..."):
                article_text = scrape_article(url_input)

            if article_text.startswith("Error"):
                st.error(f"Gagal mengambil artikel: {article_text}")
            elif len(article_text.split()) < 30:
                st.warning("Konten artikel terlalu pendek. Coba URL lain atau gunakan tab Input Teks Langsung.")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    word_count = len(article_text.split())
                    st.write(f"**Artikel Asli** — {word_count} kata")
                    st.text_area("", value=article_text, height=350, disabled=True, key="art_url", label_visibility="collapsed")

                with col2:
                    st.write("**Hasil Ringkasan**")
                    with st.spinner("Menghasilkan ringkasan..."):
                        summary = generate_summary(article_text)
                    ratio = compression_ratio(article_text, summary)
                    st.caption(f"Rasio kompresi: {ratio}% dari panjang asli")
                    st.text_area("", value=summary, height=200, disabled=True, key="sum_url", label_visibility="collapsed")
                    st.download_button(
                        label="Unduh Ringkasan (.txt)",
                        data=summary,
                        file_name="ringkasan.txt",
                        mime="text/plain",
                    )

# ── Tab 2: Teks langsung
with tab_text:
    st.subheader("Ringkas dari Teks")
    st.caption("Tempelkan teks artikel langsung untuk diringkas (minimal 30 kata).")
    text_input = st.text_area(
        "Tempelkan teks artikel di sini",
        height=250,
        placeholder="Masukkan teks artikel kesehatan di sini...",
        label_visibility="collapsed",
    )

    if st.button("Ringkas Teks", key="btn_text", type="primary"):
        if len(text_input.strip().split()) < 30:
            st.warning("Teks terlalu pendek. Minimal 30 kata.")
        else:
            clean = clean_text(text_input)
            col1, col2 = st.columns(2)

            with col1:
                word_count = len(clean.split())
                st.write(f"**Teks Asli** — {word_count} kata")
                st.text_area("", value=clean, height=300, disabled=True, key="art_text", label_visibility="collapsed")

            with col2:
                st.write("**Hasil Ringkasan**")
                with st.spinner("Menghasilkan ringkasan..."):
                    summary = generate_summary(clean)
                ratio = compression_ratio(clean, summary)
                st.caption(f"Rasio kompresi: {ratio}% dari panjang asli")
                st.text_area("", value=summary, height=200, disabled=True, key="sum_text", label_visibility="collapsed")
                st.download_button(
                    label="Unduh Ringkasan (.txt)",
                    data=summary,
                    file_name="ringkasan.txt",
                    mime="text/plain",
                )


# ==============================================================================
#  Footer
# ==============================================================================

st.divider()
st.markdown(
    '<div class="footer-text">'
    'Model: IndoBART fine-tuned (checkpoint-625) | Base: indobenchmark/indobart-v2 | '
    'Dataset: XL-Sum Indonesian | Proyek UAS Pemrosesan Bahasa Alami'
    '</div>',
    unsafe_allow_html=True,
)
