"""
🚗 Car Price Predictor - Streamlit App
Prediksi harga mobil menggunakan Random Forest dari dataset UCI Automobile (imports-85)
Jalankan: streamlit run app_streamlit3.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
    --brand: #22d3ee;
    --brand-2: #34d399;
    --brand-glow: rgba(34,211,238,0.45);
    --surface: #07070a;
    --panel: #131318;
    --panel-2: #1a1a21;
    --border: rgba(255,255,255,0.07);
    --border-strong: rgba(255,255,255,0.13);
    --text: #e7e7ea;
    --text-muted: #a1a1aa;
    --text-dim: #71717a;
}

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

/* APP SHELL */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,0.10), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(52,211,153,0.06), transparent 60%),
        linear-gradient(180deg, #0a0a10 0%, #07070a 100%) !important;
    color: var(--text) !important;
    font-feature-settings: 'cv02','cv03','cv04','cv11','ss01','ss03';
}
[data-testid="stHeader"] { background: transparent !important; backdrop-filter: blur(8px); }
.block-container { padding-top: 3rem !important; max-width: 1200px !important; }

h1, h2, h3, h4, h5 { color: #fafafa !important; letter-spacing: -0.02em !important; font-weight: 700 !important; }
h1 { font-size: 2.4rem !important; line-height: 1.15 !important; background: linear-gradient(180deg, #ffffff 0%, #a1a1aa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.2rem !important; }
p, span, label, li, div { color: var(--text); }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #06070d 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: inset -1px 0 0 rgba(34,211,238,0.05);
}
[data-testid="stSidebar"] > div { padding-top: 1rem !important; }
[data-testid="stSidebar"] * { color: #d4d4d8; font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #fafafa !important; -webkit-text-fill-color: #fafafa !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; opacity: 1 !important; margin: 0.5rem 0 !important; }
[data-testid="stSidebar"] input { color: #0a0e1a !important; }

/* Sidebar custom blocks */
.lv-brand {
    display: flex; flex-direction: column; align-items: center; padding: 0.5rem 0 1.2rem;
    border-bottom: 1px solid var(--border); margin-bottom: 1.2rem;
}
.lv-brand-logo {
    width: 64px; height: 64px; border-radius: 18px;
    background: linear-gradient(135deg, #22d3ee 0%, #0ea5e9 60%, #1e3a8a 100%);
    display: grid; place-items: center;
    box-shadow: 0 14px 40px -12px rgba(34,211,238,0.55), inset 0 1px 0 rgba(255,255,255,0.25);
    margin-bottom: 0.85rem;
    position: relative;
}
.lv-brand-logo::after {
    content: ""; position: absolute; inset: -2px; border-radius: 20px;
    background: linear-gradient(135deg, rgba(34,211,238,0.6), transparent 50%);
    z-index: -1; filter: blur(10px); opacity: 0.7;
}
.lv-brand-name { font-size: 1.15rem; font-weight: 700; color: #fafafa; letter-spacing: -0.01em; }
.lv-brand-tag { font-size: 0.7rem; color: var(--text-muted); letter-spacing: 0.18em; text-transform: uppercase; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

.lv-section-title { font-size: 0.68rem; font-weight: 700; color: var(--text-dim); letter-spacing: 0.22em; text-transform: uppercase; margin: 0.5rem 0 1rem; }

.lv-step { display: flex; gap: 12px; margin-bottom: 1.1rem; align-items: flex-start; }
.lv-step-num {
    flex-shrink: 0; width: 26px; height: 26px; border-radius: 8px;
    background: rgba(34,211,238,0.10); border: 1px solid rgba(34,211,238,0.25);
    color: var(--brand); font-size: 0.72rem; font-weight: 700;
    display: grid; place-items: center; font-family: 'JetBrains Mono', monospace;
}
.lv-step-text { font-size: 0.82rem; color: #d4d4d8; line-height: 1.5; }
.lv-step-text b { color: #fafafa; font-weight: 600; }

.lv-stats {
    margin-top: 1rem; padding: 1rem; border-radius: 14px;
    background: linear-gradient(180deg, rgba(34,211,238,0.06), rgba(34,211,238,0.0));
    border: 1px solid var(--border-strong);
}
.lv-stat-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; }
.lv-stat-row + .lv-stat-row { border-top: 1px dashed var(--border); }
.lv-stat-k { font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
.lv-stat-v { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--brand); font-weight: 700; }

.lv-foot { margin-top: 1.2rem; text-align: center; font-size: 0.68rem; color: var(--text-dim); letter-spacing: 0.1em; }

/* TABS — pill nav with glow */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: linear-gradient(180deg, rgba(20,20,28,0.85), rgba(14,14,20,0.85));
    padding: 8px;
    border-radius: 16px;
    border: 1px solid var(--border-strong);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px -12px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 10px !important;
    color: var(--text-muted) !important; font-weight: 600 !important;
    padding: 0.65rem 1.4rem !important; border: none !important;
    font-size: 0.92rem !important;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #fafafa !important; background: rgba(255,255,255,0.04) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(34,211,238,0.18), rgba(34,211,238,0.06)) !important;
    color: var(--brand) !important;
    box-shadow: 0 0 0 1px rgba(34,211,238,0.35), 0 8px 24px -10px var(--brand-glow);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* INPUTS */
.stTextInput input, .stNumberInput input, .stSelectbox > div > div, .stTextArea textarea {
    background: linear-gradient(180deg, rgba(26,26,33,0.9), rgba(19,19,24,0.9)) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 12px !important;
    color: #fafafa !important;
    height: 48px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: all 0.2s;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--brand) !important;
    box-shadow: 0 0 0 4px rgba(34,211,238,0.15), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    outline: none !important;
}
.stSelectbox > div > div { color: #fafafa !important; }

/*list pilihan — dark theme, teks putih  */
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[data-baseweb="menu"],
[data-baseweb="menu"] * {
    background: #1a1a26 !important;
    color: #e7e7ea !important;
}
[role="option"] {
    background: #1a1a26 !important;
    color: #e7e7ea !important;
    padding: 10px 16px !important;
}
[role="option"]:hover {
    background: rgba(34,211,238,0.12) !important;
    color: #22d3ee !important;
}
[role="option"][aria-selected="true"] {
    background: rgba(34,211,238,0.18) !important;
    color: #22d3ee !important;
}
[role="option"][aria-selected="true"] {
    background: rgba(34,211,238,0.18) !important;
    color: #22d3ee !important;
}
div[data-baseweb="popover"] {
    border: 1px solid var(--border-strong) !important;
    border-radius: 14px !important;
    box-shadow: 0 16px 48px -12px rgba(0,0,0,0.8) !important;
    overflow: hidden !important;
}
label, .stNumberInput label, .stSelectbox label, .stTextInput label {
    color: var(--text-muted) !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem !important;
}

.stNumberInput button {
    background: rgba(255,255,255,0.04) !important;
    border-color: var(--border-strong) !important;
    color: var(--text-muted) !important;
    height: 48px !important;
}
.stNumberInput button:hover { background: rgba(34,211,238,0.1) !important; color: var(--brand) !important; }

/* BUTTONS */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(180deg, rgba(26,26,33,0.9), rgba(19,19,24,0.9)) !important;
    color: #fafafa !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    height: 48px !important;
    transition: all 0.18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: rgba(34,211,238,0.08) !important;
    border-color: rgba(34,211,238,0.5) !important;
    color: var(--brand) !important;
    transform: translateY(-1px);
}

/* PRIMARY: big glowing predict button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #22d3ee 0%, #0ea5e9 50%, #06b6d4 100%) !important;
    color: #04131a !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    height: 58px !important;
    border-radius: 14px !important;
    box-shadow:
        0 18px 40px -10px var(--brand-glow),
        inset 0 1px 0 rgba(255,255,255,0.5),
        inset 0 -2px 0 rgba(0,0,0,0.15) !important;
    text-transform: uppercase;
    position: relative;
    overflow: hidden;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #67e8f9 0%, #22d3ee 50%, #0ea5e9 100%) !important;
    color: #04131a !important;
    transform: translateY(-2px);
    box-shadow:
        0 24px 48px -10px var(--brand-glow),
        inset 0 1px 0 rgba(255,255,255,0.6) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0); }

/* METRIC CARDS — fancier */
[data-testid="stMetric"] {
    background: linear-gradient(160deg, rgba(34,211,238,0.08) 0%, rgba(20,20,28,0.85) 40%, rgba(14,14,20,0.9) 100%) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 18px !important;
    padding: 1.6rem 1.4rem !important;
    position: relative; overflow: hidden;
    box-shadow: 0 12px 40px -16px rgba(0,0,0,0.6);
}
[data-testid="stMetric"]::before {
    content: ""; position: absolute; top: -60px; right: -60px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(34,211,238,0.25), transparent 70%);
    filter: blur(24px); pointer-events: none;
}
[data-testid="stMetric"]::after {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--brand), transparent);
}
[data-testid="stMetricLabel"] {
    color: var(--text-dim) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    color: #fafafa !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    margin-top: 0.4rem !important;
}

/* INFO / ALERT */
.stAlert, [data-testid="stAlert"] {
    background: linear-gradient(135deg, rgba(34,211,238,0.10), rgba(34,211,238,0.02)) !important;
    border: 1px solid rgba(34,211,238,0.25) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    padding: 1rem 1.2rem !important;
}
.stAlert p { color: var(--text) !important; }

/* EXPANDER */
[data-testid="stExpander"] {
    background: linear-gradient(180deg, rgba(20,20,28,0.7), rgba(14,14,20,0.7)) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 14px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary { color: #fafafa !important; font-weight: 600 !important; padding: 0.85rem 1rem !important; }
[data-testid="stExpander"] svg { color: var(--brand) !important; }
[data-testid="stExpander"] summary:hover { background: rgba(34,211,238,0.04); }

/* DIVIDER */
hr { border: none !important; height: 1px !important; background: linear-gradient(90deg, transparent, var(--border-strong), transparent) !important; opacity: 1 !important; margin: 2rem 0 !important; }

/* DATAFRAME */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--border-strong) !important;
    border-radius: 14px !important;
    overflow: hidden;
    background: rgba(14,14,20,0.6) !important;
}

/* PROGRESS BAR */
.stProgress > div > div > div > div { background: linear-gradient(90deg, var(--brand), var(--brand-2)) !important; box-shadow: 0 0 12px var(--brand-glow); }
.stProgress > div > div > div { background: rgba(255,255,255,0.06) !important; border-radius: 999px !important; height: 8px !important; }

/* CODE BLOCKS */
.stCode, pre, code {
    background: #0a0a0f !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* CAPTION */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-dim) !important;
    font-size: 0.78rem !important;
}

/* Result hero card */
.lv-result-hero {
    margin: 1rem 0 1.5rem; padding: 1.4rem 1.6rem; border-radius: 18px;
    background: linear-gradient(135deg, rgba(34,211,238,0.12) 0%, rgba(52,211,153,0.06) 100%);
    border: 1px solid rgba(34,211,238,0.3);
    display: flex; justify-content: space-between; align-items: center;
    box-shadow: 0 12px 40px -16px var(--brand-glow);
}
.lv-result-hero .lv-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--brand); letter-spacing: 0.2em; text-transform: uppercase; }
.lv-result-hero .lv-pill { padding: 6px 14px; border-radius: 999px; background: rgba(34,211,238,0.15); border: 1px solid rgba(34,211,238,0.4); color: var(--brand); font-weight: 700; font-size: 0.85rem; }

/* Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #07070a; }
::-webkit-scrollbar-thumb { background: #27272a; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #3f3f46; }

/* HELP ICON (tanda tanya) — lebih terlihat */
button[data-testid="stTooltipHoverTarget"],
[data-testid="stTooltipIcon"] {
    color: var(--brand) !important;
    opacity: 1 !important;
    filter: none !important;
}
button[data-testid="stTooltipHoverTarget"] svg,
[data-testid="stTooltipIcon"] svg {
    color: var(--brand) !important;
    fill: var(--brand) !important;
    opacity: 1 !important;
    width: 18px !important;
    height: 18px !important;
}
button[data-testid="stTooltipHoverTarget"]:hover svg {
    color: #67e8f9 !important;
    fill: #67e8f9 !important;
}

/* TOOLTIP POPUP — dark background, teks terbaca */
[data-testid="stTooltipContent"],
div[role="tooltip"],
div[data-radix-popper-content-wrapper] > div {
    background: #1a1a26 !important;
    color: #e7e7ea !important;
    border: 1px solid rgba(34,211,238,0.35) !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    font-size: 0.85rem !important;
    box-shadow: 0 8px 32px -8px rgba(0,0,0,0.8) !important;
}
div[data-radix-popper-content-wrapper] > div * {
    color: #e7e7ea !important;
}
</style>
""")

# AMBIL KURS USD → IDR (scraping, tanpa API)
# ============================================================
@st.cache_data(ttl=3600)   # cache 1 jam supaya tidak scrape terus-terusan
def get_kurs_usd_idr():
    """
    Ambil kurs USD→IDR otomatis dari Google Finance / Yahoo Finance.
    Tanpa API key, cukup pakai requests + BeautifulSoup.
    Kalau keduanya gagal, return fallback 15500.
    """
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # ── Coba 1: Google Finance ───────────────────────────────
    try:
        url = "https://www.google.com/finance/quote/USD-IDR"
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        tag = soup.find(class_="YMlKec fxKbKc")   # class kurs di Google Finance
        if tag:
            return int(float(tag.text.replace(",", "")))
    except Exception:
        pass

    # ── Coba 2: Yahoo Finance JSON API ───────────────────────
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/USDIDR=X?interval=1d&range=1d"
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        kurs = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return int(kurs)
    except Exception:
        pass

    # ── Fallback: nilai default ───────────────────────────────
    return 15500


if "usd_to_idr" not in st.session_state:
    st.session_state.usd_to_idr = get_kurs_usd_idr()

# LOAD & PREPROCESS DATASET (cached)
# ============================================================
@st.cache_data
def load_data():
    columns = [
        "simbol_risiko", "kerugian_normal", "merek", "jenis_bahan_bakar", "aspirasi",
        "jumlah_pintu", "tipe_bodi", "penggerak_roda", "letak_mesin",
        "jarak_sumbu_roda", "panjang", "lebar", "tinggi", "berat_kosong",
        "tipe_mesin", "jumlah_silinder", "ukuran_mesin", "sistem_bahan_bakar",
        "diameter_bore", "langkah_piston", "rasio_kompresi", "tenaga_mesin",
        "rpm_puncak", "konsumsi_kota", "konsumsi_tol", "harga",
    ]

    # Coba load dari file lokal dulu, fallback ke URL jika tidak ada
    import os
    local_path = "imports-85.data"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

    if os.path.exists(local_path):
        df = pd.read_csv(local_path, names=columns)
    else:
        try:
            df = pd.read_csv(url, names=columns)
        except Exception:
            st.error("❌ File 'imports-85.data' tidak ditemukan dan tidak bisa akses internet.\n\n"
                     "Letakkan file `imports-85.data` di folder yang sama dengan `app_streamlit3.py`, lalu jalankan ulang.")
            st.stop()
    df.replace("?", np.nan, inplace=True)
    df = df.drop(columns=["kerugian_normal"])
    kolom_numerik = [
        "diameter_bore", "langkah_piston",
        "tenaga_mesin", "rpm_puncak", "harga",
        "jarak_sumbu_roda", "panjang", "lebar", "tinggi",
        "berat_kosong", "ukuran_mesin", "rasio_kompresi",
        "konsumsi_kota", "konsumsi_tol"
    ]
    for col in kolom_numerik:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df

@st.cache_data
def load_raw_data():
    columns = [
        "simbol_risiko", "kerugian_normal", "merek", "jenis_bahan_bakar", "aspirasi",
        "jumlah_pintu", "tipe_bodi", "penggerak_roda", "letak_mesin",
        "jarak_sumbu_roda", "panjang", "lebar", "tinggi", "berat_kosong",
        "tipe_mesin", "jumlah_silinder", "ukuran_mesin", "sistem_bahan_bakar",
        "diameter_bore", "langkah_piston", "rasio_kompresi", "tenaga_mesin",
        "rpm_puncak", "konsumsi_kota", "konsumsi_tol", "harga",
    ]
    import os
    local_path = "imports-85.data"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

    if os.path.exists(local_path):
        return pd.read_csv(local_path, names=columns)
    else:
        return pd.read_csv(url, names=columns)

@st.cache_resource
def train_all_models(df):
    fitur_numerik = [
        "ukuran_mesin", "tenaga_mesin", "berat_kosong",
        "lebar", "panjang", "konsumsi_tol"
    ]
    fitur_kategorikal = [
        "tipe_bodi", "jenis_bahan_bakar", "penggerak_roda"
    ]
    semua_fitur = fitur_numerik + fitur_kategorikal

    X = df[semua_fitur]
    y = df["harga"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessing = ColumnTransformer(transformers=[
        ("scaler", StandardScaler(), fitur_numerik),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), fitur_kategorikal),
    ])

    hasil = {}
    for nama, algo in [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree",     DecisionTreeRegressor(max_depth=5, random_state=42)),
        ("Random Forest",     RandomForestRegressor(max_depth=5, random_state=42)),
    ]:
        pipe = Pipeline(steps=[
            ("preprocessing", preprocessing),
            ("model", algo),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        hasil[nama] = {
            "model":    pipe,
            "r2":       r2_score(y_test, y_pred),
            "mae":      mean_absolute_error(y_test, y_pred),
            "mse":      mean_squared_error(y_test, y_pred),
        }

    best = hasil["Random Forest"]
    best["fitur_numerik"]     = fitur_numerik
    best["fitur_kategorikal"] = fitur_kategorikal
    best["semua_fitur"]       = semua_fitur
    best["perbandingan"]      = hasil

    return best

def format_idr(usd):
    return f"Rp {int(usd * st.session_state.usd_to_idr):,}".replace(",", ".")


# ============================================================
# LOAD DATA
# ============================================================
with st.spinner("📥 Memuat & memproses dataset..."):
    df = load_data()
    df_raw = load_raw_data()
    model_info = train_all_models(df)


# ============================================================
# SIDEBAR — Logo + Cara Penggunaan
# ============================================================
with st.sidebar:
        # Logo
    st.markdown("""
        <div style="text-align:center; padding: 10px 0 16px 0;">
            <div style="font-size: 48px;">🚗</div>
            <div style="font-size: 20px; font-weight: 700; letter-spacing: 1px;">Car Price</div>
            <div style="font-size: 13px; opacity: 0.8;">Prediction App</div>
        </div>
        """, unsafe_allow_html=True)
    st.html(f"""
        <div class="lv-section-title">Cara Penggunaan</div>

        <div class="lv-step">
            <div class="lv-step-num">01</div>
            <div class="lv-step-text"><b>Pelajari Data</b> — buka tab <b>Analisis Data</b> untuk melihat statistik dataset dan alur pengolahan.</div>
        </div>
        <div class="lv-step">
            <div class="lv-step-num">02</div>
            <div class="lv-step-text"><b>Input Spesifikasi</b> — isi formulir dengan detail teknis mobil (mesin, tenaga, dimensi, tipe bodi).</div>
        </div>
        <div class="lv-step">
            <div class="lv-step-num">03</div>
            <div class="lv-step-text"><b>Klik Prediksi</b> — model <b>Random Forest</b> akan memproses data secara instan.</div>
        </div>
        <div class="lv-step">
            <div class="lv-step-num">04</div>
            <div class="lv-step-text"><b>Lihat Hasil</b> — estimasi harga akan muncul dalam USD dan IDR.</div>
        </div>

        <div class="lv-foot">© 2026 · CarPrice AI</div>
        """)


# ============================================================
# NAVIGASI TAB DI ATAS
# ============================================================
tab_prediksi, tab_tentang_aplikasi, tab_analisis, tab_tentang = st.tabs([
    "Prediksi Harga",
    "Tentang Aplikasi",
    "Analisis Data",
    "👤 Tentang Saya",
])


# ============================================================
# TAB: HOME
# ============================================================
with tab_tentang_aplikasi:
# Kode ini diletakkan di dalam logika menu Home
    st.title("🚗 Car Price Predictor")
    st.markdown("---")
    
    # Bagian Latar Belakang
    st.subheader("Latar Belakang")
    st.write("""
    Industri otomotif memiliki variasi harga yang sangat dinamis tergantung pada spesifikasi teknis kendaraan. 
    Seringkali, pembeli atau penjual kesulitan menentukan harga pasar yang adil bagi sebuah kendaraan berdasarkan fitur-fitur yang dimilikinya. 
    Aplikasi ini dikembangkan untuk memberikan solusi cerdas dalam mengestimasi harga jual mobil secara instan.
    """)
    
    st.write("""
    Dengan memanfaatkan algoritma **Machine Learning (Random Forest)**, aplikasi ini mempelajari pola dari sebuah data spesifikasi mobil
    untuk memprediksi harga mobil berdasarkan variabel teknis yang diinputkan.
    """)

    st.markdown("---")

    # Bagian Kegunaan Aplikasi
    st.subheader("Kegunaan Aplikasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Estimasi Harga Cepat**
        Membantu pengguna mendapatkan gambaran harga jual atau beli mobil tanpa harus melakukan riset manual yang memakan waktu lama.
        
        **2. Referensi Pengambilan Keputusan**
        Menjadi dasar pertimbangan bagi penjual untuk memasang harga yang kompetitif atau bagi pembeli agar tidak membeli di atas harga pasar.
        """)
        
    st.info("💡 Klik menu **Prediksi Harga** di samping untuk mulai mencoba perhitungan harga mobil Anda.")

# ============================================================
# TAB: PREDIKSI HARGA
# ============================================================
with tab_prediksi:
    st.title("🔮 Prediksi Harga Mobil")
    st.markdown(f"""
    Estimasi harga menggunakan **Random Forest** (akurasi R² = **{model_info['r2']:.2%}**,
    rata-rata error ±${model_info['mae']:,.0f}).
    """)
    st.divider()

    # Kurs USD → IDR — otomatis dari internet
    with st.expander("💱 Kurs USD → IDR (Otomatis)", expanded=False):
        col_kurs1, col_kurs2 = st.columns([3, 1])
        with col_kurs1:
            st.markdown(f"**Kurs aktif: 1 USD = Rp {st.session_state.usd_to_idr:,}**".replace(",", "."))
            st.caption("Diambil otomatis dari Google Finance / Yahoo Finance. Diperbarui setiap 1 jam.")
        with col_kurs2:
            if st.button("🔄 Refresh Kurs", use_container_width=True):
                get_kurs_usd_idr.clear()                        # hapus cache
                st.session_state.usd_to_idr = get_kurs_usd_idr()  # ambil ulang
                st.rerun()
        st.info(f"💡 Kurs terakhir diambil: **1 USD = Rp {st.session_state.usd_to_idr:,}**".replace(",", "."))

    st.divider()
    st.subheader("📝 Form Input Spesifikasi")

    col1, col2 = st.columns(2)
    with col1:
        ukuran_mesin = st.number_input(
            "🔧 Ukuran Mesin (cc)",
            int(df["ukuran_mesin"].min()), int(df["ukuran_mesin"].max()), 130,
            help=f"Range dataset: {int(df['ukuran_mesin'].min())} – {int(df['ukuran_mesin'].max())}"
        )
        tenaga_mesin = st.number_input(
            "🐎 Tenaga Mesin (hp)",
            int(df["tenaga_mesin"].min()), int(df["tenaga_mesin"].max()), 110,
            help=f"Range dataset: {int(df['tenaga_mesin'].min())} – {int(df['tenaga_mesin'].max())}"
        )
        berat_kosong = st.number_input(
            "⚖️ Berat Kosong (lbs)",
            int(df["berat_kosong"].min()), int(df["berat_kosong"].max()), 2500,
            help=f"Range dataset: {int(df['berat_kosong'].min())} – {int(df['berat_kosong'].max())}"
        )
        lebar = st.number_input(
            "📏 Lebar (inch)",
            float(df["lebar"].min()), float(df["lebar"].max()), 65.5, step=0.1,
            help=f"Range dataset: {df['lebar'].min():.1f} – {df['lebar'].max():.1f}"
        )
        panjang = st.number_input(
            "📐 Panjang (inch)",
            float(df["panjang"].min()), float(df["panjang"].max()), 170.0, step=0.1,
            help=f"Range dataset: {df['panjang'].min():.1f} – {df['panjang'].max():.1f}"
        )
        konsumsi_tol = st.number_input(
            "⛽ Konsumsi Tol (MPG)",
            int(df["konsumsi_tol"].min()), int(df["konsumsi_tol"].max()), 30,
            help=f"Range dataset: {int(df['konsumsi_tol'].min())} – {int(df['konsumsi_tol'].max())}"
        )
    with col2:
        tipe_bodi = st.selectbox(
            "🚙 Tipe Bodi",
            sorted(df["tipe_bodi"].unique())
        )
        jenis_bahan_bakar = st.selectbox(
            "⛽ Jenis Bahan Bakar",
            sorted(df["jenis_bahan_bakar"].unique())
        )
        penggerak_roda = st.selectbox(
            "🔄 Penggerak Roda",
            sorted(df["penggerak_roda"].unique())
        )

    if st.button("🔍 Prediksi Harga", type="primary", use_container_width=True):
        input_df = pd.DataFrame([[
            ukuran_mesin, tenaga_mesin, berat_kosong,
            lebar, panjang, konsumsi_tol,
            tipe_bodi, jenis_bahan_bakar, penggerak_roda
        ]], columns=model_info["semua_fitur"])

        predicted = float(model_info["model"].predict(input_df)[0])
        predicted = max(predicted, 0)

        st.divider()
        st.subheader("✅ Hasil Estimasi")
        c1, c2 = st.columns(2)
        c1.metric("💵 Harga (USD)", f"${predicted:,.0f}")
        c2.metric("💰 Harga (IDR)", format_idr(predicted))

        if predicted < 10000:
            kat, emoji = "Ekonomis", "🟢"
        elif predicted < 20000:
            kat, emoji = "Menengah", "🟡"
        elif predicted < 35000:
            kat, emoji = "Premium", "🟠"
        else:
            kat, emoji = "Mewah", "🔴"
        st.info(f"{emoji} **Kategori: {kat}**")

        st.divider()
        st.subheader("📌 Mengapa harganya bisa segitu?")

        importances = model_info["model"].named_steps["model"].feature_importances_
        ohe = model_info["model"].named_steps["preprocessing"].named_transformers_["ohe"]
        cat_feature_names = ohe.get_feature_names_out(model_info["fitur_kategorikal"]).tolist()
        all_feature_names = model_info["fitur_numerik"] + cat_feature_names

        importance_dict = {}
        for fname in model_info["fitur_numerik"]:
            idx = all_feature_names.index(fname)
            importance_dict[fname] = importances[idx]
        for cat in model_info["fitur_kategorikal"]:
            idxs = [i for i, n in enumerate(all_feature_names) if n.startswith(cat + "_")]
            importance_dict[cat] = sum(importances[i] for i in idxs)

        sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        label_map = {
            "ukuran_mesin": "Ukuran Mesin",
            "tenaga_mesin": "Tenaga Mesin",
            "berat_kosong": "Berat Kosong",
            "lebar": "Lebar Mobil",
            "panjang": "Panjang Mobil",
            "konsumsi_tol": "Konsumsi BBM",
            "tipe_bodi": "Tipe Bodi",
            "jenis_bahan_bakar": "Jenis Bahan Bakar",
            "penggerak_roda": "Penggerak Roda",
        }

        input_map = {
            "ukuran_mesin": f"{ukuran_mesin} cc",
            "tenaga_mesin": f"{tenaga_mesin} hp",
            "berat_kosong": f"{berat_kosong} lbs",
            "lebar": f"{lebar} inch",
            "panjang": f"{panjang} inch",
            "konsumsi_tol": f"{konsumsi_tol} MPG",
            "tipe_bodi": tipe_bodi,
            "jenis_bahan_bakar": jenis_bahan_bakar,
            "penggerak_roda": penggerak_roda,
        }

        st.markdown("Berdasarkan model Random Forest, berikut faktor yang paling mempengaruhi harga prediksi ini:")
        st.write("")
        for fitur, imp in sorted_imp[:3]:
            nama = label_map[fitur]
            nilai = input_map[fitur]
            persen = imp * 100
            col_a, col_b = st.columns([3, 1])
            col_a.markdown(f"**{nama}** — {nilai}")
            col_b.markdown(f"`{persen:.1f}%`")
            st.progress(imp / sorted_imp[0][1])

        st.caption("Persentase menunjukkan seberapa besar pengaruh tiap fitur terhadap keputusan model secara keseluruhan.")

        st.divider()
        st.subheader("🔍 Mobil dengan Harga Mirip")
        similar = df.copy()
        similar["selisih"] = (similar["harga"] - predicted).abs()
        similar = similar.nsmallest(5, "selisih")[[
            "merek", "tipe_bodi", "jenis_bahan_bakar", "penggerak_roda",
            "ukuran_mesin", "tenaga_mesin", "berat_kosong", "harga"
        ]]
        similar["harga_idr"] = similar["harga"].apply(format_idr)
        st.dataframe(similar, use_container_width=True, hide_index=True)


# ============================================================
# TAB: ANALISIS DATA (Dataset + Analisis + Glosarium Istilah)
# ============================================================
with tab_analisis:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import cross_val_score
    import joblib
    import io

    st.title("📊 Analisis Data")
    st.markdown("Eksplorasi dataset, visualisasi, perbandingan model, dan penjelasan istilah teknis.")

    sub_dataset, sub_notebook, sub_glosarium = st.tabs([
        "📋 Dataset",
        "📓 Notebook",
        "📚 Penjelasan Istilah",
    ])

    with sub_dataset:
        st.markdown("### 📋 Tentang Dataset")
        st.markdown("""
        Dataset yang digunakan adalah **UCI Automobile Dataset (imports-85)**, 
        berisi data spesifikasi teknis dan harga mobil dari berbagai merek.
        """)

        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Data", f"{len(df_raw)} baris")
        col2.metric("Jumlah Kolom", f"{len(df_raw.columns)} kolom")
        col3.metric("Data Bersih", f"{len(df)} baris")

        st.divider()

        st.markdown("### Sumber Data")
        st.markdown("""
        | Info | Detail |
        |---|---|
        | **Nama Dataset** | Automobile Dataset |
        | **Sumber** | UCI Machine Learning Repository |
        | **Link** | https://archive.ics.uci.edu/dataset/10/automobile |
        | **Tahun** | 1985 |
        | **Target** | Harga mobil (USD) |
        """)

        st.divider()

        # Tabel data bersih
        st.markdown("### Data Bersih")
        st.dataframe(df, use_container_width=True, hide_index=False)

        # Tombol download
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Dataset (CSV)",
            data=csv_data,
            file_name="automobile_clean.csv",
            mime="text/csv",
            use_container_width=True,
        )
    # ── SUB TAB: NOTEBOOK ────────────────────────────────────
    with sub_notebook:
        st.markdown("Semua kode dari Jupyter Notebook ditampilkan lengkap beserta outputnya.")

        # ── Cell 1: Load data (head) ──────────────────────────
        st.markdown("##### Cell 1 — Load Dataset")
        st.code(
            'import pandas as pd\n\ndf = pd.read_csv("imports-85.data", header=None)\ndf.head()',
            language="python"
        )
        st.markdown("**Output:**")
        st.dataframe(df_raw.head(), use_container_width=True)
        st.divider()

        # ── Cell 2: df (tampilkan semua) ──────────────────────
        st.markdown("##### Cell 2 — Tampilkan Seluruh DataFrame")
        st.code(
            'import pandas as pd\n\ndf = pd.read_csv("imports-85.data", header=None)\ndf',
            language="python"
        )
        st.markdown("**Output:**")
        st.dataframe(df_raw, use_container_width=True)
        st.divider()

        # ── Cell 3: Rename kolom ──────────────────────────────
        st.markdown("##### Cell 3 — Beri Nama Kolom")
        st.code(
            """df.columns = [
                "simbol_risiko","kerugian_normal","merek","jenis_bahan_bakar","aspirasi",
                "jumlah_pintu","tipe_bodi","penggerak_roda","letak_mesin",
                "jarak_sumbu_roda","panjang","lebar","tinggi","berat_kosong","tipe_mesin",
                "jumlah_silinder","ukuran_mesin","sistem_bahan_bakar","diameter_bore","langkah_piston",
                "rasio_kompresi","tenaga_mesin","rpm_puncak","konsumsi_kota",
                "konsumsi_tol","harga"
            ]""",
            language="python"
        )
        st.markdown("**Output:** *(tidak ada output, kolom diubah in-place)*")
        st.divider()

        # ── Cell 4: df.head() ─────────────────────────────────
        st.markdown("##### Cell 4 — df.head()")
        st.code("df.head()", language="python")
        st.markdown("**Output:**")
        st.dataframe(df_raw.head(), use_container_width=True)
        st.divider()

        # ── Cell 5: df.tail() ─────────────────────────────────
        st.markdown("##### Cell 5 — df.tail()")
        st.code("df.tail()", language="python")
        st.markdown("**Output:**")
        st.dataframe(df_raw.tail(), use_container_width=True)
        st.divider()

        # ── Cell 6: df.shape ──────────────────────────────────
        st.markdown("##### Cell 6 — df.shape")
        st.code("df.shape", language="python")
        st.markdown("**Output:**")
        st.code(f"({len(df_raw)}, {len(df_raw.columns)})", language="text")
        st.divider()

        # ── Cell 7: df.columns ────────────────────────────────
        st.markdown("##### Cell 7 — df.columns")
        st.code("df.columns", language="python")
        st.markdown("**Output:**")
        st.code('Index([' + ', '.join([f"'{c}'" for c in df_raw.columns]) + '], dtype=\'object\')', language="text")
        st.divider()

        # ── Cell 8: df.info() ─────────────────────────────────
        st.markdown("##### Cell 8 — df.info()")
        st.code("df.info()", language="python")
        st.markdown("**Output:**")
        import io as _io
        buf_info = _io.StringIO()
        df_raw.info(buf=buf_info)
        st.code(buf_info.getvalue(), language="text")
        st.divider()

        # ── Cell 9: df.sample(5) ──────────────────────────────
        st.markdown("##### Cell 9 — df.sample(5)")
        st.code("df.sample(5)", language="python")
        st.markdown("**Output:**")
        st.dataframe(df_raw.sample(5, random_state=42), use_container_width=True)
        st.divider()

        # ── Cell 10: df.isna().sum() ──────────────────────────
        st.markdown("##### Cell 10 — df.isna().sum()")
        st.code("df.isna().sum()", language="python")
        st.markdown("**Output:**")
        st.code("\n".join([f"{k:<22} {v}" for k, v in df_raw.isna().sum().items()]) + "\ndtype: int64",language="text")
        st.divider()

        # ── Cell 11: Cek nilai '?' ────────────────────────────
        st.markdown("##### Cell 11 — Cek Nilai '?'")
        st.code('(df == "?").sum()', language="python")
        st.markdown("**Output:**")
        tanda_tanya = {
            "simbol_risiko": 0, "kerugian_normal": 41, "merek": 0,
            "jenis_bahan_bakar": 0, "aspirasi": 0, "jumlah_pintu": 2,
            "tipe_bodi": 0, "penggerak_roda": 0, "letak_mesin": 0,
            "jarak_sumbu_roda": 0, "panjang": 0, "lebar": 0, "tinggi": 0,
            "berat_kosong": 0, "tipe_mesin": 0, "jumlah_silinder": 0,
            "ukuran_mesin": 0, "sistem_bahan_bakar": 0, "diameter_bore": 4,
            "langkah_piston": 4, "rasio_kompresi": 0, "tenaga_mesin": 2,
            "rpm_puncak": 2, "konsumsi_kota": 0, "konsumsi_tol": 0, "harga": 4,
        }
        st.code("\n".join([f"{k:<22} {v}" for k, v in tanda_tanya.items()]) + "\ndtype: int64",language="text")
        st.divider()

        # ── Cell 12: Replace '?' → NaN ───────────────────────
        st.markdown("##### Cell 12 — Replace '?' → NaN")
        st.code(
            'import numpy as np\n\ndf.replace("?", np.nan, inplace=True)',
            language="python"
        )
        st.markdown("**Output:** *(tidak ada output, data diubah in-place)*")
        st.divider()

        # ── Cell 13: isna().sum() setelah replace ─────────────
        st.markdown("##### Cell 13 — df.isna().sum() setelah Replace")
        st.code("df.isna().sum()", language="python")
        st.markdown("**Output:**")
        st.code("\n".join([f"{k:<22} {v}" for k, v in tanda_tanya.items()]) + "\ndtype: int64",language="text")
        st.divider()

        # ── Cell 14: df.describe() ────────────────────────────
        st.markdown("##### Cell 14 — df.describe()")
        st.code("df.describe()", language="python")
        st.markdown("**Output:**")
        st.dataframe(df_raw.describe(), use_container_width=True)
        st.divider()

        # ── Cell 15: df.duplicated().sum() ───────────────────
        st.markdown("##### Cell 15 — df.duplicated().sum()")
        st.code("df.duplicated().sum()", language="python")
        st.markdown("**Output:**")
        st.code(str(df_raw.duplicated().sum()),language="text")
        st.divider()

        # ── Cell 16: Drop kolom & dropna ─────────────────────
        st.markdown("##### Cell 16 — Preprocessing: Hapus Kolom & dropna")
        st.code(
            """df = df.drop(columns=["kerugian_normal"])  # buang kolom
            df = df.dropna()""",
            language="python"
        )
        st.markdown("**Output:** *(tidak ada output, data diubah in-place)*")
        st.divider()

        # ── Cell 17: Histplot distribusi harga ───────────────
        st.markdown("##### Cell 17 — Visualisasi Distribusi Harga")
        st.code(
            """import seaborn as sns
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10,6))
            sns.histplot(df["harga"], kde=True, bins=20)
            plt.title("Distribusi Harga Mobil")
            plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(df["harga"], kde=True, bins=20, ax=ax1)
        ax1.set_title("Distribusi Harga Mobil")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        st.divider()

        # ── Cell 18: Scatter — ukuran_mesin ──────────────────
        st.markdown("##### Cell 18 — Scatter: Ukuran Mesin vs Harga")
        st.code(
            """sns.scatterplot(x="ukuran_mesin", y="harga", data=df)
            plt.title("Ukuran Mesin vs Harga")
            plt.show()
            plt.figure(figsize=(10,6))""",
            language="python"
        )
        st.markdown("**Output:**")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x="ukuran_mesin", y="harga", data=df, ax=ax2)
        ax2.set_title("Ukuran Mesin vs Harga")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        st.divider()

        # ── Cell 19: Scatter — tenaga_mesin ──────────────────
        st.markdown("##### Cell 19 — Scatter: Tenaga Mesin vs Harga")
        st.code(
            """sns.scatterplot(x="tenaga_mesin", y="harga", data=df)
            plt.title("Tenaga Mesin vs Harga")
            plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x="tenaga_mesin", y="harga", data=df, ax=ax3)
        ax3.set_title("Tenaga Mesin vs Harga")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
        st.divider()

        # ── Cell 20: Scatter — berat_kosong ──────────────────
        st.markdown("##### Cell 20 — Scatter: Berat Mobil vs Harga")
        st.code(
            """sns.scatterplot(x="berat_kosong", y="harga", data=df)
            plt.title("Berat Mobil vs Harga")
            plt.figure(figsize=(10,6))
            plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        fig4, ax4 = plt.subplots()
        sns.scatterplot(x="berat_kosong", y="harga", data=df, ax=ax4)
        ax4.set_title("Berat Mobil vs Harga")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
        st.divider()

        # ── Cell 21: Scatter — konsumsi_tol ──────────────────
        st.markdown("##### Cell 21 — Scatter: Konsumsi BBM Tol vs Harga")
        st.code(
            """sns.scatterplot(x="konsumsi_tol", y="harga", data=df)
            plt.title("Konsumsi BBM Tol vs Harga")
            plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        fig5, ax5 = plt.subplots()
        sns.scatterplot(x="konsumsi_tol", y="harga", data=df, ax=ax5)
        ax5.set_title("Konsumsi BBM Tol vs Harga")
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()
        st.divider()

        # ── Cell 22: Heatmap korelasi ─────────────────────────
        st.markdown("##### Cell 22 — Heatmap Korelasi Antar Fitur")
        st.code(
            """corr = df.corr(numeric_only=True)

            sns.heatmap(corr, annot=True)
            plt.title("Korelasi Antar Fitur")
            plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        fig6, ax6 = plt.subplots(figsize=(10, 7))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, ax=ax6, annot_kws={"size": 7})
        ax6.set_title("Korelasi Antar Fitur")
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()
        st.divider()

        # ── Cell 23: Konversi kolom numerik ──────────────────
        st.markdown("##### Cell 23 — Konversi Tipe Data Kolom Numerik")
        st.code(
            """kolom_numerik = [
                "diameter_bore","langkah_piston",
                "tenaga_mesin","rpm_puncak","harga"
            ]

            for col in kolom_numerik:
                df[col] = pd.to_numeric(df[col])""",
            language="python"
        )
        st.markdown("**Output:** *(tidak ada output, tipe data diubah in-place)*")
        st.divider()

        # ── Cell 24: df.dtypes ────────────────────────────────
        st.markdown("##### Cell 24 — df.dtypes")
        st.code("df.dtypes", language="python")
        st.markdown("**Output:**")
        st.code(str(df.dtypes),language="text")
        st.divider()

        # ── Cell 25: Boxplot outlier harga ───────────────────
        st.markdown("##### Cell 25 — Deteksi Outlier Harga (Boxplot)")
        st.code(
            """sns.boxplot(x=df["harga"])
            plt.title("Deteksi Outlier Harga")
            plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        fig7, ax7 = plt.subplots()
        sns.boxplot(x=df["harga"], ax=ax7)
        ax7.set_title("Deteksi Outlier Harga")
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()
        st.divider()

        # ── Cell 26: Boxplot tipe bodi vs harga ──────────────
        st.markdown("##### Cell 26 — Tipe Bodi vs Harga (Boxplot)")
        st.code(
            """sns.boxplot(x="tipe_bodi", y="harga", data=df)
            plt.title("Tipe Bodi vs Harga")
            plt.xticks(rotation=45)
            plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        fig8, ax8 = plt.subplots()
        sns.boxplot(x="tipe_bodi", y="harga", data=df, ax=ax8)
        ax8.set_title("Tipe Bodi vs Harga")
        ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig8)
        plt.close()
        st.divider()

        # ── Cell 27: Heatmap korelasi ke harga ───────────────
        st.markdown("##### Cell 27 — Korelasi Fitur terhadap Harga")
        st.code(
            """plt.figure(figsize=(6,8))

            corr = df.corr(numeric_only=True)

            sns.heatmap(
                corr.loc[:, ["harga"]].sort_values(by="harga", ascending=False),
                annot=True,
                cmap="coolwarm"
            )

            plt.title("Korelasi terhadap Harga")
            plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        fig9, ax9 = plt.subplots(figsize=(4, 7))
        corr_h = df.corr(numeric_only=True)[["harga"]].sort_values(by="harga", ascending=False)
        sns.heatmap(corr_h, annot=True, fmt=".2f", cmap="coolwarm", ax=ax9)
        ax9.set_title("Korelasi terhadap Harga")
        plt.tight_layout()
        st.pyplot(fig9)
        plt.close()
        st.divider()

        # ── Cell 28: Info merek ───────────────────────────────
        st.markdown("##### Cell 28 — Info Merek")
        st.code(
            """print("Jumlah data:", len(df))
            print("Jumlah merek unik:", df["merek"].nunique())
            print(df["merek"].value_counts())""",
            language="python"
        )
        st.markdown("**Output:**")
        merek_counts = df["merek"].value_counts()
        output_merek = f"Jumlah data: {len(df)}\nJumlah merek unik: {df['merek'].nunique()}\n"
        output_merek += "merek\n" + "\n".join([f"{m:<20} {c}" for m, c in merek_counts.items()])
        output_merek += "\nName: count, dtype: int64"
        st.code(output_merek)
        st.divider()

        # ── Cell 29: Linear Regression (tanpa merek, R2 only) ─
        st.markdown("##### Cell 29 — Linear Regression (Eksperimen Tanpa Merek)")
        st.code(
            """from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score

            # fitur
            X1 = df[[
                "ukuran_mesin",
                "tenaga_mesin",
                "berat_kosong",
                "lebar",
                "panjang",
                "konsumsi_tol",
                "tipe_bodi",
                "jenis_bahan_bakar",
                "penggerak_roda"
            ]]

            y = df["harga"]

            X_train1, X_test1, y_train, y_test = train_test_split(
                X1, y, test_size=0.2, random_state=42
            )

            num_cols = [
                "ukuran_mesin","tenaga_mesin","berat_kosong",
                "lebar","panjang","konsumsi_tol"
            ]

            cat_cols1 = [
                "tipe_bodi","jenis_bahan_bakar","penggerak_roda"
            ]

            pre1 = ColumnTransformer([
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols1)
            ])

            model1 = Pipeline([
                ("pre", pre1),
                ("model", LinearRegression())
            ])

            model1.fit(X_train1, y_train)
            y_pred1 = model1.predict(X_test1)

            r2_1 = r2_score(y_test, y_pred1)
            print("R2 tanpa merek:", r2_1)""",
            language="python"
        )
        st.markdown("**Output:**")
        lr_info = model_info["perbandingan"]["Linear Regression"]
        st.code(f"R2 tanpa merek: {lr_info['r2']}",language="text")
        st.divider()

        # ── Cell 30: Linear Regression (full metrics) ────────
        st.markdown("##### Cell 30 — Linear Regression (Full Metrics)")
        st.code(
            """from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer

            X = df[[
                "ukuran_mesin","tenaga_mesin","berat_kosong",
                "lebar","panjang","konsumsi_tol",
                "tipe_bodi","jenis_bahan_bakar","penggerak_roda"
            ]]

            y = df["harga"]

            X_train,X_test,y_train,y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            numeric_columns = [
                "ukuran_mesin","tenaga_mesin","berat_kosong",
                "lebar","panjang","konsumsi_tol"
            ]

            categorical_columns = [
                "tipe_bodi","jenis_bahan_bakar","penggerak_roda"
            ]

            preprocessing = ColumnTransformer(
                transformers=[
                    ("scaler", StandardScaler(), numeric_columns),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
                ]
            )

            model = Pipeline(
                steps=[
                    ("preprocessing", preprocessing),
                    ("model", LinearRegression())
                ]
            )

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            print("Linear Regression")
            print("R2 Score : ", r2_score(y_test, y_pred))
            print("MAE Score : ", mean_absolute_error(y_test, y_pred))
            print("MSE Score : ", mean_squared_error(y_test, y_pred))""",
            language="python"
        )
        st.markdown("**Output:**")
        lr_info = model_info["perbandingan"]["Linear Regression"]
        st.code(f"Linear Regression\nR2 Score :  {lr_info['r2']}\nMAE Score :  {lr_info['mae']}\nMSE Score :  {lr_info['mse']}",language="text")
        st.divider()

        # ── Cell 31: Decision Tree ────────────────────────────
        st.markdown("##### Cell 31 — Decision Tree Regressor")
        st.code(
            """from sklearn.tree import DecisionTreeRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer

            X = df[[
                "ukuran_mesin","tenaga_mesin","berat_kosong",
                "lebar","panjang","konsumsi_tol",
                "tipe_bodi","jenis_bahan_bakar","penggerak_roda"
            ]]

            y = df["harga"]

            X_train,X_test,y_train,y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            numeric_columns = [
                "ukuran_mesin","tenaga_mesin","berat_kosong",
                "lebar","panjang","konsumsi_tol"
            ]

            categorical_columns = [
                "tipe_bodi","jenis_bahan_bakar","penggerak_roda"
            ]

            preprocessing = ColumnTransformer(
                transformers=[
                    ("scaler", StandardScaler(), numeric_columns),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
                ]
            )

            model = Pipeline(
                steps=[
                    ("preprocessing", preprocessing),
                    ("model", DecisionTreeRegressor(max_depth=5, random_state=42))
                ]
            )

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            print("Decision Tree")
            print("R2 Score : ", r2_score(y_test, y_pred))
            print("MAE Score : ", mean_absolute_error(y_test, y_pred))
            print("MSE Score : ", mean_squared_error(y_test, y_pred))""",
                        language="python"
                    )
        st.markdown("**Output:**")
        dt_info = model_info["perbandingan"]["Decision Tree"]
        st.code(f"Decision Tree\nR2 Score :  {dt_info['r2']}\nMAE Score :  {dt_info['mae']}\nMSE Score :  {dt_info['mse']}",language="text")
        st.divider()

        # ── Cell 32: Random Forest ────────────────────────────
        st.markdown("##### Cell 32 — Random Forest Regressor")
        st.code(
                        """from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer

            X = df[[
                "ukuran_mesin","tenaga_mesin","berat_kosong",
                "lebar","panjang","konsumsi_tol",
                "tipe_bodi","jenis_bahan_bakar","penggerak_roda"
            ]]

            y = df["harga"]

            X_train,X_test,y_train,y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            numeric_columns = [
                "ukuran_mesin","tenaga_mesin","berat_kosong",
                "lebar","panjang","konsumsi_tol"
            ]

            categorical_columns = [
                "tipe_bodi","jenis_bahan_bakar","penggerak_roda"
            ]

            preprocessing = ColumnTransformer(
                transformers=[
                    ("scaler", StandardScaler(), numeric_columns),
                    ("ohe", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
                ]
            )

            model = Pipeline(
                steps=[
                    ("preprocessing", preprocessing),
                    ("model", RandomForestRegressor(random_state=42))
                ]
            )

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            print("Random Forest")
            print("R2 Score : ", r2_score(y_test, y_pred))
            print("MAE Score : ", mean_absolute_error(y_test, y_pred))
            print("MSE Score : ", mean_squared_error(y_test, y_pred))""",
            language="python"
        )
        st.markdown("**Output:**")
        rf_info = model_info["perbandingan"]["Random Forest"]
        st.code(f"Random Forest\nR2 Score :  {rf_info['r2']}\nMAE Score :  {rf_info['mae']}\nMSE Score :  {rf_info['mse']}",language="text")
        st.divider()

        # ── Cell 33: Cross Validation ─────────────────────────
        st.markdown("##### Cell 33 — Cross Validation (5-Fold)")
        st.code(
            """from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

            print("Score tiap fold :", scores)
            print("Mean Score :", scores.mean())""",
            language="python"
        )
        st.markdown("**Output:**")
        from sklearn.model_selection import cross_val_score as cvs
        X_cv = df[model_info["semua_fitur"]]
        y_cv = df["harga"]
        from sklearn.model_selection import train_test_split as tts
        Xtrcv, _, ytrcv, _ = tts(X_cv, y_cv, test_size=0.2, random_state=42)
        cv_scores = cvs(model_info["model"], Xtrcv, ytrcv, cv=5, scoring="r2")
        st.code(f"Score tiap fold : {cv_scores}\nMean Score : {cv_scores.mean()}",language="text")
        st.divider()

        # ── Cell 34: Save model ───────────────────────────────
        st.markdown("##### Cell 34 — Simpan Model (.joblib)")
        st.code(
            """import joblib
            joblib.dump(model, "model_prediksi_mobil.joblib")""",
            language="python"
        )
        st.markdown("**Output:**")
        st.code("['model_prediksi_mobil.joblib']",language="text")
        import io as _io2
        import joblib
        buf_dl = _io2.BytesIO()
        joblib.dump(model_info["model"], buf_dl)
        buf_dl.seek(0)
        st.download_button("📥 Download model_prediksi_mobil.joblib", buf_dl,
                           "model_prediksi_mobil.joblib", "application/octet-stream")
        st.divider()

        # ── Cell 35: Contoh prediksi ──────────────────────────
        st.markdown("##### Cell 35 — Contoh Prediksi Data Baru")
        st.code(
            """import joblib
            import pandas as pd

            model = joblib.load("model_prediksi_mobil.joblib")

            data_baru = pd.DataFrame(
                [["sedan","bensin","fwd",130,110,2500,65,170,30]],
                columns=[
                    "tipe_bodi","jenis_bahan_bakar","penggerak_roda",
                    "ukuran_mesin","tenaga_mesin","berat_kosong",
                    "lebar","panjang","konsumsi_tol"
                ]
            )

            prediksi = model.predict(data_baru)[0]

            print(f"Prediksi harga mobil : {prediksi:.0f} USD")
            rupiah = prediksi * 15000
            print(f"Harga: Rp {rupiah:,.0f}")""",
            language="python"
        )
        st.markdown("**Output:**")
        st.text("Prediksi harga mobil : 10826 USD\nHarga: Rp 162,397,000")
        st.divider()

        # ── Cell 36: Perbandingan prediksi vs asli ────────────
        st.markdown("##### Cell 36 — Perbandingan Prediksi vs Harga Asli (5 Sample)")
        st.code(
            """# Ambil 5 sample acak dari dataset
            sample = df.sample(5, random_state=42)

            # Input hanya 9 kolom yang dipakai model
            X_sample = sample[[
                "ukuran_mesin", "tenaga_mesin", "berat_kosong",
                "lebar", "panjang", "konsumsi_tol",
                "tipe_bodi", "jenis_bahan_bakar", "penggerak_roda"
            ]]

            # Prediksi
            prediksi = model.predict(X_sample)

            # Bandingkan
            hasil = pd.DataFrame({
                "Harga Asli (USD)": sample["harga"].values,
                "Prediksi (USD)": prediksi.astype(int),
                "Selisih (USD)": abs(sample["harga"].values - prediksi).astype(int)
            })

            print(hasil)""",
            language="python"
        )
        st.markdown("**Output:**")
        sample = df.sample(5, random_state=42)
        X_samp = sample[model_info["semua_fitur"]]
        pred_samp = model_info["model"].predict(X_samp)
        hasil_samp = pd.DataFrame({
            "Harga Asli (USD)": sample["harga"].values,
            "Prediksi (USD)": pred_samp.astype(int),
            "Selisih (USD)": abs(sample["harga"].values - pred_samp).astype(int)
        })
        st.dataframe(hasil_samp, use_container_width=True, hide_index=True)
        st.divider()

    # ── SUB TAB: ISTILAH ────────────────────────────
    with sub_glosarium:
        st.markdown("Penjelasan istilah teknis yang digunakan dalam aplikasi dan dataset ini.")
        st.divider()

        with st.expander("🔧 Ukuran Mesin"):
            st.write(f"""
            Ukuran mesin adalah angka yang menunjukkan seberapa besar kapasitas mesin sebuah mobil,
            dengan satuan cc (cubic centimeter).

            Semakin besar kapasitas mesinnya, semakin besar tenaga yang dihasilkan. Namun di sisi lain,
            mesin yang lebih besar juga cenderung lebih boros bahan bakar dan membuat harga mobil lebih tinggi.
            Istilah ini sering kita dengar dalam kehidupan sehari-hari, misalnya ketika seseorang menyebut
            "mobil bermesin 1500 cc" atau "2000 cc".

            Nilai yang dapat diisi: **{int(df['ukuran_mesin'].min())} hingga {int(df['ukuran_mesin'].max())} cc**
            """)

        with st.expander("🐎 Tenaga Mesin"):
            st.write(f"""
            Tenaga mesin adalah ukuran seberapa besar kekuatan yang mampu dihasilkan oleh mesin,
            dengan satuan hp (horsepower) atau tenaga kuda.

            Semakin tinggi nilai hp-nya, semakin responsif mobil saat berakselerasi. Mobil dengan tenaga
            mesin yang besar umumnya lebih cepat dan lebih bertenaga, namun harganya juga cenderung lebih mahal.

            Nilai yang dapat diisi: **{int(df['tenaga_mesin'].min())} hingga {int(df['tenaga_mesin'].max())} hp**
            """)

        with st.expander("⚖️ Berat Kosong"):
            st.write(f"""
            Berat kosong adalah bobot mobil dalam kondisi tanpa penumpang dan tanpa muatan apapun,
            dengan satuan lbs (pounds). Sebagai referensi, 1 kg setara dengan sekitar 2,2 lbs.

            Mobil yang lebih berat pada umumnya memiliki dimensi yang lebih besar, rangka yang lebih kokoh,
            dan kelengkapan fitur yang lebih banyak, sehingga harganya pun cenderung lebih tinggi.

            Nilai yang dapat diisi: **{int(df['berat_kosong'].min())} hingga {int(df['berat_kosong'].max())} lbs**
            """)

        with st.expander("📏 Lebar Mobil"):
            st.write(f"""
            Lebar mobil adalah jarak dari sisi kiri ke sisi kanan bodi kendaraan, tidak termasuk spion,
            dengan satuan inch. Sebagai referensi, 1 inch setara dengan 2,54 cm.

            Kendaraan dengan lebar yang lebih besar biasanya masuk dalam kategori kelas menengah ke atas.

            Nilai yang dapat diisi: **{df['lebar'].min():.1f} hingga {df['lebar'].max():.1f} inch**
            """)

        with st.expander("📐 Panjang Mobil"):
            st.write(f"""
            Panjang mobil adalah jarak dari ujung bemper depan hingga ujung bemper belakang kendaraan,
            dengan satuan inch.

            Mobil yang lebih panjang biasanya memiliki ruang kabin yang lebih lega serta kapasitas bagasi
            yang lebih besar.

            Nilai yang dapat diisi: **{df['panjang'].min():.1f} hingga {df['panjang'].max():.1f} inch**
            """)

        with st.expander("⛽ Konsumsi BBM di Jalan Tol"):
            st.write(f"""
            Angka ini menunjukkan efisiensi bahan bakar sebuah mobil saat melaju di jalan tol,
            dengan satuan MPG (miles per gallon). Semakin tinggi nilainya, semakin irit.

            Nilai yang dapat diisi: **{int(df['konsumsi_tol'].min())} hingga {int(df['konsumsi_tol'].max())} MPG**
            """)

        with st.expander("🚙 Tipe Bodi"):
            st.write("""
            Tipe bodi adalah bentuk desain keseluruhan dari sebuah kendaraan:

            **Convertible** — Kendaraan dengan atap yang dapat dibuka atau dilipat.

            **Hardtop** — Kendaraan dengan atap keras permanen, umumnya tanpa pilar tengah.

            **Hatchback** — Kendaraan kompak dengan ruang bagasi menyatu dengan kabin dan pintu di belakang.

            **Sedan** — Tipe paling umum dengan tiga bagian terpisah: mesin, kabin, dan bagasi.

            **Wagon** — Seperti sedan namun bagian belakang lebih panjang dan tinggi, kapasitas bagasi lebih besar.
            """)

        with st.expander("⛽ Jenis Bahan Bakar"):
            st.write("""
            **Gas** — Kendaraan menggunakan bensin, tipe paling umum untuk kendaraan penumpang.

            **Diesel** — Kendaraan menggunakan solar. Lebih efisien untuk perjalanan jarak jauh
            dan menghasilkan torsi besar pada putaran rendah.
            """)

        with st.expander("Penggerak Roda"):
            st.write("""
            **FWD (Front Wheel Drive)** — Tenaga disalurkan ke roda depan. Paling umum, efisien BBM.

            **RWD (Rear Wheel Drive)** — Tenaga disalurkan ke roda belakang. Banyak digunakan pada kendaraan sport/mewah.

            **4WD (Four Wheel Drive)** — Tenaga ke keempat roda. Cocok untuk medan berat, umum pada SUV/off-road.
            """)



# ============================================================
# TAB: TENTANG SAYA
# ============================================================
with tab_tentang:
    col_foto, col_bio = st.columns([1, 2])

    with col_foto:
        from PIL import Image
        import base64
        import os
        from io import BytesIO

        foto_path = "19012.jpg"  # sesuaikan nama file foto kamu

        if os.path.exists(foto_path):
            img = Image.open(foto_path)
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()

            st.markdown(f"""
            <div style="
                width: 140px;
                height: 140px;
                border-radius: 50%;
                overflow: hidden;
                margin: 0 auto;
                border: 3px solid #2563eb;
            ">
                <img src="data:image/png;base64,{img_b64}" style="
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                "/>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                width: 140px;
                height: 140px;
                border-radius: 50%;
                background: linear-gradient(135deg, #1a3a6b, #2563eb);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 64px;
                margin: 0 auto;
            ">👤</div>
            """, unsafe_allow_html=True)

    with col_bio:
        st.markdown("""
        ### Alecia Febriana
        **Jurusan Rekayasa Perangat Lunak**  
        SMK Negeri 1 Purbalingga

        📧 febrianaalecia@gmail.com  
        🐙 github.com/aleciafbri
        """)

    st.divider()
    st.subheader("Tentang Proyek Ini")
    st.markdown(f"""
    Aplikasi **Car Price Predictor** dibuat sebagai bagian dari tugas/proyek Machine Learning
    """)
