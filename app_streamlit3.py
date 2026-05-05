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

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #1a3a6b;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] input {
    color: #1a3a6b !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

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

    st.divider()

    st.markdown("### 📖 Cara Penggunaan")
    st.markdown("""
    1. **Pelajari Data**: Buka tab **Analisis Data** jika Anda ingin melihat statistik dataset dan alur kode pengolahan data asli.
    2. **Input Spesifikasi**: Pindah ke halaman **Prediksi Harga**. Isi formulir dengan detail teknis mobil seperti:
        * Kapasitas Mesin (cc).
        * Tenaga (hp).
        * Dimensi (panjang/lebar).
        * Tipe Bodi, dll.
    3. **Klik Prediksi**: Tekan tombol **🚀 Prediksi** untuk memproses data menggunakan model *Random Forest*[cite: 2].
    4. **Lihat Hasil**: Harga estimasi akan muncul dalam USD dan IDR (Rupiah).
    """)

    st.divider()
    st.caption(f"R² Model: **{model_info['r2']:.3f}** | MAE: **${model_info['mae']:,.0f}**")
    st.caption("© 2026 Car Price Predictor")


# ============================================================
# NAVIGASI TAB DI ATAS
# ============================================================
tab_home, tab_analisis, tab_prediksi, tab_tentang = st.tabs([
    "🏠 Home",
    "📊 Analisis Data",
    "🔮 Prediksi Harga",
    "👤 Tentang Saya",
])


# ============================================================
# TAB: HOME
# ============================================================
with tab_home:
# Kode ini diletakkan di dalam logika menu Home
    st.title("🚗 Car Price Predictor")
    st.markdown("---")
    
    # Bagian Latar Belakang
    st.subheader("📜 Latar Belakang")
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
    st.subheader("💡 Kegunaan Aplikasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Estimasi Harga Cepat**
        Membantu pengguna mendapatkan gambaran harga jual atau beli mobil tanpa harus melakukan riset manual yang memakan waktu lama.
        
        **2. Referensi Pengambilan Keputusan**
        Menjadi dasar pertimbangan bagi penjual untuk memasang harga yang kompetitif atau bagi pembeli agar tidak membeli di atas harga pasar.
        """)

    with col2:
        st.markdown("""
        **3. Analisis Fitur Kendaraan**
        Pengguna dapat melihat bagaimana perubahan pada spesifikasi (seperti tenaga mesin atau efisiensi BBM) dapat mempengaruhi nilai ekonomi kendaraan.
        
        **4. Digitalisasi Penaksiran Harga**
        Menggantikan metode taksiran tradisional dengan pendekatan berbasis data (Data-Driven) yang lebih objektif.
        """)

    st.info("💡 Klik menu **Prediksi Harga** di samping untuk mulai mencoba simulasi perhitungan harga mobil Anda.")

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

    # ── SUB TAB: DATASET ──────────────────────────────────────
    with sub_dataset:
        st.markdown(f"Dataset UCI Machine Learning Repository setelah preprocessing — **{len(df)} baris × {len(df.columns)} kolom**.")
        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah Data", len(df))
        c2.metric("Jumlah Kolom", len(df.columns))
        c3.metric("Data Berharga", df["harga"].notna().sum())
        c4.metric("Jumlah Merek", df["merek"].nunique())

        col_search, col_filter = st.columns([2, 1])
        search = col_search.text_input("🔍 Cari merek / tipe bodi", "")
        makes = ["Semua"] + sorted(df["merek"].dropna().unique().tolist())
        selected_make = col_filter.selectbox("Filter Merek", makes)

        filtered = df.copy()
        if selected_make != "Semua":
            filtered = filtered[filtered["merek"] == selected_make]
        if search:
            s = search.lower()
            filtered = filtered[
                filtered["merek"].str.lower().str.contains(s, na=False) |
                filtered["tipe_bodi"].str.lower().str.contains(s, na=False)
            ]

        display = filtered[[
            "merek", "tipe_bodi", "jenis_bahan_bakar", "penggerak_roda",
            "ukuran_mesin", "tenaga_mesin", "berat_kosong",
            "lebar", "panjang", "konsumsi_tol", "harga"
        ]].copy()
        display["harga_idr"] = display["harga"].apply(
            lambda x: format_idr(x) if pd.notna(x) else "-"
        )
        st.caption(f"Menampilkan **{len(display)}** baris")
        st.dataframe(display, use_container_width=True, height=450)

        csv = filtered.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "automobile_dataset.csv", "text/csv")

    # ── SUB TAB: NOTEBOOK ────────────────────────────────────
    with sub_notebook:
        st.markdown("Semua kode dari Jupyter Notebook ditampilkan lengkap beserta outputnya.")

        def nb_cell(judul, kode, output):
            """Helper: tampilkan 1 cell notebook — judul, kode, lalu output."""
            st.markdown(f"##### {judul}")
            st.code(kode, language="python")
            st.markdown("**Output:**")
            output
            st.divider()

        # ── Cell 1: Load data ─────────────────────────────────
        st.markdown("##### Cell 1 — Load Dataset")
        st.code(
            'import pandas as pd\n\ndf = pd.read_csv("imports-85.data", header=None)\ndf.head()',
            language="python"
        )
        st.markdown("**Output:**")
        st.dataframe(df.head(), use_container_width=True)
        st.divider()

        # ── Cell 2: Rename kolom ──────────────────────────────
        st.markdown("##### Cell 2 — Beri Nama Kolom")
        st.code(
            """columns = [
    "simbol_risiko","kerugian_normal","merek","jenis_bahan_bakar","aspirasi",
    "jumlah_pintu","tipe_bodi","penggerak_roda","letak_mesin",
    "jarak_sumbu_roda","panjang","lebar","tinggi","berat_kosong",
    "tipe_mesin","jumlah_silinder","ukuran_mesin","sistem_bahan_bakar",
    "diameter_bore","langkah_piston","rasio_kompresi","tenaga_mesin",
    "rpm_puncak","konsumsi_kota","konsumsi_tol","harga",
]
df.columns = columns
df.head()""",
            language="python"
        )
        st.markdown("**Output:**")
        st.dataframe(df.head(), use_container_width=True)
        st.divider()

        # ── Cell 3: df.tail() ─────────────────────────────────
        st.markdown("##### Cell 3 — df.tail()")
        st.code("df.tail()", language="python")
        st.markdown("**Output:**")
        st.dataframe(df.tail(), use_container_width=True)
        st.divider()

        # ── Cell 4: df.shape ──────────────────────────────────
        st.markdown("##### Cell 4 — df.shape")
        st.code("df.shape", language="python")
        st.markdown("**Output:**")
        st.text(f"({len(df)}, {len(df.columns)})")
        st.divider()

        # ── Cell 5: df.columns ────────────────────────────────
        st.markdown("##### Cell 5 — df.columns")
        st.code("df.columns", language="python")
        st.markdown("**Output:**")
        st.text("Index([" + ", ".join([f"'{c}'" for c in df.columns]) + "], dtype='object')")
        st.divider()

        # ── Cell 6: df.info() ─────────────────────────────────
        st.markdown("##### Cell 6 — df.info()")
        st.code("df.info()", language="python")
        st.markdown("**Output:**")
        import io as _io
        buf_info = _io.StringIO()
        df.info(buf=buf_info)
        st.text(buf_info.getvalue())
        st.divider()

        # ── Cell 7: df.sample(5) ──────────────────────────────
        st.markdown("##### Cell 7 — df.sample(5)")
        st.code("df.sample(5)", language="python")
        st.markdown("**Output:**")
        st.dataframe(df.sample(5, random_state=42), use_container_width=True)
        st.divider()

        # ── Cell 8: Cek nilai '?' ─────────────────────────────
        st.markdown("##### Cell 8 — Cek Nilai '?'")
        st.code('(df == "?").sum()', language="python")
        st.markdown("**Output:**")
        # df di sini sudah bersih, tampilkan nilai asli dari notebook
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
        st.text("\n".join([f"{k:<22} {v}" for k, v in tanda_tanya.items()]) + "\ndtype: int64")
        st.divider()

        # ── Cell 9: Replace '?' → NaN ─────────────────────────
        st.markdown("##### Cell 9 — Replace '?' → NaN")
        st.code(
            'import numpy as np\n\ndf.replace("?", np.nan, inplace=True)',
            language="python"
        )
        st.markdown("**Output:** *(tidak ada output, data diubah in-place)*")
        st.divider()

        # ── Cell 10: isna setelah replace ────────────────────
        st.markdown("##### Cell 10 — df.isna().sum() setelah replace")
        st.code("df.isna().sum()", language="python")
        st.markdown("**Output:**")
        st.text("\n".join([f"{k:<22} {v}" for k, v in tanda_tanya.items()]) + "\ndtype: int64")
        st.divider()

        # ── Cell 11: df.describe() ────────────────────────────
        st.markdown("##### Cell 11 — df.describe()")
        st.code("df.describe()", language="python")
        st.markdown("**Output:**")
        st.dataframe(df.describe(), use_container_width=True)
        st.divider()

        # ── Cell 12: Hapus kerugian_normal, dropna, konversi ─
        st.markdown("##### Cell 12 — Preprocessing: Hapus kolom, dropna, konversi tipe data")
        st.code(
            """df = df.drop(columns=["kerugian_normal"])

kolom_numerik = [
    "diameter_bore","langkah_piston","tenaga_mesin","rpm_puncak","harga",
    "jarak_sumbu_roda","panjang","lebar","tinggi","berat_kosong",
    "ukuran_mesin","rasio_kompresi","konsumsi_kota","konsumsi_tol"
]
for col in kolom_numerik:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()
print("Jumlah data:", len(df))
print("Jumlah merek unik:", df["merek"].nunique())
print(df["merek"].value_counts())""",
            language="python"
        )
        st.markdown("**Output:**")
        merek_counts = df["merek"].value_counts()
        output_merek = f"Jumlah data: {len(df)}\nJumlah merek unik: {df['merek'].nunique()}\n"
        output_merek += "merek\n" + "\n".join([f"{m:<20} {c}" for m, c in merek_counts.items()])
        output_merek += "\nName: count, dtype: int64"
        st.text(output_merek)
        st.divider()

        # ── Cell 13: Visualisasi distribusi harga ────────────
        st.markdown("##### Cell 13 — Visualisasi Distribusi Harga")
        st.code(
            """import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df["harga"], kde=True, bins=20)
plt.title("Distribusi Harga Mobil")
plt.xlabel("Harga (USD)")
plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.histplot(df["harga"], kde=True, bins=20, ax=ax1, color="#2563eb")
        ax1.set_title("Distribusi Harga Mobil")
        ax1.set_xlabel("Harga (USD)")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        st.divider()

        # ── Cell 14: Scatter plots ────────────────────────────
        st.markdown("##### Cell 14 — Scatter Plot Fitur vs Harga")
        st.code(
            """fig, axes = plt.subplots(2, 2, figsize=(12, 8))
pairs = [
    ("ukuran_mesin","Ukuran Mesin vs Harga"),
    ("tenaga_mesin","Tenaga Mesin vs Harga"),
    ("berat_kosong","Berat Mobil vs Harga"),
    ("konsumsi_tol","Konsumsi BBM Tol vs Harga"),
]
for ax, (fitur, judul) in zip(axes.flat, pairs):
    sns.scatterplot(x=fitur, y="harga", data=df, ax=ax)
    ax.set_title(judul)
plt.tight_layout()
plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (fitur, judul) in zip(axes2.flat, [
            ("ukuran_mesin","Ukuran Mesin vs Harga"),
            ("tenaga_mesin","Tenaga Mesin vs Harga"),
            ("berat_kosong","Berat Mobil vs Harga"),
            ("konsumsi_tol","Konsumsi BBM Tol vs Harga"),
        ]):
            sns.scatterplot(x=fitur, y="harga", data=df, ax=ax, color="#2563eb")
            ax.set_title(judul)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        st.divider()

        # ── Cell 15: Heatmap korelasi ─────────────────────────
        st.markdown("##### Cell 15 — Heatmap Korelasi Antar Fitur")
        st.code(
            """plt.figure(figsize=(10, 7))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Korelasi Antar Fitur")
plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3, annot_kws={"size": 7})
        ax3.set_title("Korelasi Antar Fitur")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
        st.divider()

        # ── Cell 16: Heatmap korelasi ke harga ───────────────
        st.markdown("##### Cell 16 — Korelasi Fitur terhadap Harga")
        st.code(
            """plt.figure(figsize=(6, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(
    corr.loc[:, ["harga"]].sort_values(by="harga", ascending=False),
    annot=True, cmap="coolwarm"
)
plt.title("Korelasi terhadap Harga")
plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        fig4, ax4 = plt.subplots(figsize=(4, 7))
        corr_h = df.corr(numeric_only=True)[["harga"]].sort_values(by="harga", ascending=False)
        sns.heatmap(corr_h, annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
        ax4.set_title("Korelasi terhadap Harga")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
        st.divider()

        # ── Cell 17: Boxplot outlier ──────────────────────────
        st.markdown("##### Cell 17 — Deteksi Outlier dengan Boxplot")
        st.code(
            """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x=df["harga"], ax=ax1)
ax1.set_title("Deteksi Outlier Harga")
sns.boxplot(x="tipe_bodi", y="harga", data=df, ax=ax2)
ax2.set_title("Tipe Bodi vs Harga")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()""",
            language="python"
        )
        st.markdown("**Output:**")
        fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(x=df["harga"], ax=ax5a, color="#2563eb")
        ax5a.set_title("Deteksi Outlier Harga")
        sns.boxplot(x="tipe_bodi", y="harga", data=df, ax=ax5b, palette="Set2")
        ax5b.set_title("Tipe Bodi vs Harga")
        ax5b.set_xticklabels(ax5b.get_xticklabels(), rotation=45)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()
        st.divider()

        # ── Cell 18: Linear Regression ────────────────────────
        st.markdown("##### Cell 18 — Linear Regression")
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

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_columns = ["ukuran_mesin","tenaga_mesin","berat_kosong","lebar","panjang","konsumsi_tol"]
categorical_columns = ["tipe_bodi","jenis_bahan_bakar","penggerak_roda"]

preprocessing = ColumnTransformer(transformers=[
    ("scaler", StandardScaler(), numeric_columns),
    ("ohe", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])

model = Pipeline(steps=[
    ("preprocessing", preprocessing),
    ("model", LinearRegression())
])

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
        st.text(f"Linear Regression\nR2 Score :  {lr_info['r2']}\nMAE Score :  {lr_info['mae']}\nMSE Score :  {lr_info['mse']}")
        st.divider()

        # ── Cell 19: Decision Tree ────────────────────────────
        st.markdown("##### Cell 19 — Decision Tree Regressor")
        st.code(
            """from sklearn.tree import DecisionTreeRegressor

model = Pipeline(steps=[
    ("preprocessing", preprocessing),
    ("model", DecisionTreeRegressor(max_depth=5, random_state=42))
])

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
        st.text(f"Decision Tree\nR2 Score :  {dt_info['r2']}\nMAE Score :  {dt_info['mae']}\nMSE Score :  {dt_info['mse']}")
        st.divider()

        # ── Cell 20: Random Forest ────────────────────────────
        st.markdown("##### Cell 20 — Random Forest Regressor")
        st.code(
            """from sklearn.ensemble import RandomForestRegressor

model = Pipeline(steps=[
    ("preprocessing", preprocessing),
    ("model", RandomForestRegressor(random_state=42))
])

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
        st.text(f"Random Forest\nR2 Score :  {rf_info['r2']}\nMAE Score :  {rf_info['mae']}\nMSE Score :  {rf_info['mse']}")
        st.divider()

        # ── Cell 21: Cross Validation ─────────────────────────
        st.markdown("##### Cell 21 — Cross Validation (5-Fold)")
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
        st.text(f"Score tiap fold : {cv_scores}\nMean Score : {cv_scores.mean()}")
        st.divider()

        # ── Cell 22: Save model ───────────────────────────────
        st.markdown("##### Cell 22 — Simpan Model (.joblib)")
        st.code(
            """import joblib

joblib.dump(model, "model_prediksi_mobil.joblib")""",
            language="python"
        )
        st.markdown("**Output:**")
        st.text("['model_prediksi_mobil.joblib']")
        import io as _io2
        import joblib
        buf_dl = _io2.BytesIO()
        joblib.dump(model_info["model"], buf_dl)
        buf_dl.seek(0)
        st.download_button("📥 Download model_prediksi_mobil.joblib", buf_dl,
                           "model_prediksi_mobil.joblib", "application/octet-stream")
        st.divider()

        # ── Cell 23: Contoh prediksi ──────────────────────────
        st.markdown("##### Cell 23 — Contoh Prediksi Data Baru")
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

        # ── Cell 24: Perbandingan prediksi vs asli ────────────
        st.markdown("##### Cell 24 — Perbandingan Prediksi vs Harga Asli (5 Sample)")
        st.code(
            """# Ambil 5 sample acak dari dataset
sample = df.sample(5, random_state=42)

X_sample = sample[[
    "ukuran_mesin","tenaga_mesin","berat_kosong",
    "lebar","panjang","konsumsi_tol",
    "tipe_bodi","jenis_bahan_bakar","penggerak_roda"
]]

prediksi = model.predict(X_sample)

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

    # ── SUB TAB: GLOSARIUM ISTILAH ────────────────────────────
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

        with st.expander("🔄 Penggerak Roda"):
            st.write("""
            **FWD (Front Wheel Drive)** — Tenaga disalurkan ke roda depan. Paling umum, efisien BBM.

            **RWD (Rear Wheel Drive)** — Tenaga disalurkan ke roda belakang. Banyak digunakan pada kendaraan sport/mewah.

            **4WD (Four Wheel Drive)** — Tenaga ke keempat roda. Cocok untuk medan berat, umum pada SUV/off-road.
            """)

        with st.expander("📊 Apa itu R² dan MAE?"):
            st.write(f"""
            **R² (R-squared) = {model_info['r2']:.3f}**
            Menunjukkan seberapa baik model menjelaskan pola data. Nilai maksimal 1,0 = sempurna.
            Model ini mampu menjelaskan **{model_info['r2']*100:.1f}%** pola harga dari data.

            **MAE (Mean Absolute Error) = ${model_info['mae']:,.0f}**
            Rata-rata selisih antara harga prediksi dan harga asli. Model ini dapat meleset sekitar
            **${model_info['mae']:,.0f}** dari harga sebenarnya.
            """)


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
        tipe_bodi = st.selectbox(
            "🚙 Tipe Bodi",
            sorted(df["tipe_bodi"].unique())
        )
    with col2:
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
# TAB: TENTANG SAYA
# ============================================================
with tab_tentang:
    col_foto, col_bio = st.columns([1, 2])
    with col_foto:
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
        SMKN 1 PURBALINGGA

        📧 febrianaalecia@gmail.com
        🐙 github.com/username
        """)

    st.divider()

    st.subheader("📌 Tentang Proyek Ini")
    st.markdown(f"""
    Aplikasi **Car Price Predictor** dibuat sebagai bagian dari tugas/proyek pembelajaran
    Machine Learning
    """)

    st.divider()
    st.subheader("🛠️ Teknologi yang Digunakan")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.info("🐍 Python")
    col_t1.info("🎈 Streamlit")
    col_t2.info("🤖 Scikit-learn")
    col_t2.info("🐼 Pandas & NumPy")
    col_t3.info("📊 Matplotlib & Seaborn")
    col_t3.info("📈 Plotly Express")