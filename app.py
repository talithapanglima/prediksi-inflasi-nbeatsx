import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="InflasiCast — N-BEATSx",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');
:root {
  --bg:#0a0a0a; --surface:#111; --surface2:#181818;
  --border:rgba(255,255,255,0.07); --accent:#b5ff4d; --accent2:#4dffb5;
  --text:#d4d4d4; --muted:#555; --danger:#ff6b6b; --warn:#ffd166;
}
html,body,[class*="css"]{font-family:'Syne',sans-serif;background:var(--bg)!important;color:var(--text);}
.block-container{padding:1.5rem 2rem 3rem 2rem!important;max-width:1400px;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
.hero{border:1px solid var(--border);border-left:4px solid var(--accent);background:var(--surface);padding:2rem 2.5rem;margin-bottom:2rem;border-radius:2px;}
.hero-eyebrow{font-family:'IBM Plex Mono',monospace;font-size:.65rem;letter-spacing:3px;text-transform:uppercase;color:var(--accent);margin-bottom:.6rem;}
.hero-title{font-size:2.2rem;font-weight:800;color:#fff;letter-spacing:-1px;line-height:1.1;margin-bottom:.6rem;}
.hero-sub{font-size:.85rem;color:var(--muted);max-width:540px;line-height:1.7;}
.step-badge{display:inline-flex;align-items:center;font-family:'IBM Plex Mono',monospace;font-size:.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--accent);border:1px solid rgba(181,255,77,.25);padding:.3rem .75rem;border-radius:2px;margin-bottom:.75rem;}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:2px;padding:1.5rem;margin-bottom:1.25rem;}
.panel-title{font-family:'IBM Plex Mono',monospace;font-size:.65rem;letter-spacing:2.5px;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);padding-bottom:.6rem;margin-bottom:1.1rem;}
.result-block{background:var(--surface2);border:1px solid rgba(181,255,77,.2);border-left:4px solid var(--accent);border-radius:2px;padding:1.5rem;margin-bottom:.75rem;}
.result-period{font-family:'IBM Plex Mono',monospace;font-size:.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:.3rem;}
.result-value{font-family:'IBM Plex Mono',monospace;font-size:2.6rem;font-weight:600;color:var(--accent);line-height:1;}
.result-pct{font-size:1.2rem;color:var(--muted);}
.result-badge{display:inline-block;margin-top:.6rem;padding:.2rem .65rem;border-radius:2px;font-family:'IBM Plex Mono',monospace;font-size:.64rem;font-weight:600;letter-spacing:1px;}
.alert-info{background:rgba(77,255,181,.05);border-left:3px solid var(--accent2);padding:.85rem 1rem;border-radius:0 2px 2px 0;font-size:.82rem;color:var(--text);margin:.75rem 0;line-height:1.6;}
.alert-warn{background:rgba(255,209,102,.05);border-left:3px solid var(--warn);padding:.85rem 1rem;border-radius:0 2px 2px 0;font-size:.82rem;color:var(--text);margin:.75rem 0;line-height:1.6;}
.alert-err{background:rgba(255,107,107,.07);border-left:3px solid var(--danger);padding:.85rem 1rem;border-radius:0 2px 2px 0;font-size:.82rem;color:var(--text);margin:.75rem 0;line-height:1.6;}
.template-code{font-family:'IBM Plex Mono',monospace;font-size:.72rem;background:#0d0d0d;border:1px solid var(--border);border-radius:2px;padding:1rem;color:#aaa;overflow-x:auto;line-height:1.8;}
.col-hi{color:var(--accent);} .col-opt{color:var(--accent2);}
.stButton>button{background:var(--accent)!important;color:#000!important;border:none!important;border-radius:2px!important;font-family:'IBM Plex Mono',monospace!important;font-size:.75rem!important;font-weight:600!important;letter-spacing:1.5px!important;text-transform:uppercase!important;padding:.6rem 1.5rem!important;}
div[data-testid="stMetric"]{background:var(--surface2);border:1px solid var(--border);border-radius:2px;padding:.9rem 1rem;}
div[data-testid="stMetric"] label{color:var(--muted)!important;font-size:.7rem!important;font-family:'IBM Plex Mono',monospace!important;}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:var(--accent)!important;font-family:'IBM Plex Mono',monospace!important;}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════
MODEL_PATH   = 'saved_nbeatsx/nf_model'
SCALERY_PATH = 'saved_nbeatsx/scaler_y.pkl'
SCALERX_PATH = 'saved_nbeatsx/scaler_exog.pkl'
PARAMS_PATH  = 'saved_nbeatsx/best_params_v2.pkl'
FULL_PATH    = 'saved_nbeatsx/full_df.parquet'
ASLI_PATH    = 'saved_nbeatsx/df_asli.parquet'

HIST_EXOG  = ['lag1','lag3','lag6','lag12','Harga Minyak Dunia','BI Rate','Kurs USD/IDR']
FUTR_EXOG  = ['Ramadhan','Idulfitri','Natal','Imlek']
ALL_EXOG   = HIST_EXOG + FUTR_EXOG

RAMADAN_MAP    = {2010:8,2011:8,2012:7,2013:7,2014:6,2015:6,2016:6,2017:5,
                  2018:5,2019:5,2020:4,2021:4,2022:4,2023:3,2024:3,2025:3,2026:2,2027:2}
IDUL_FITRI_MAP = {2010:9,2011:9,2012:8,2013:8,2014:7,2015:7,2016:7,2017:6,
                  2018:6,2019:6,2020:5,2021:5,2022:5,2023:4,2024:4,2025:3,2026:3,2027:3}

PLOT_BG    = "#0a0a0a"
PLOT_PAPER = "#111111"
PLOT_GRID  = "rgba(255,255,255,0.04)"
C_PRED     = "#b5ff4d"
C_HIST     = "#444"
C_ACTUAL   = "#d4d4d4"

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
def auto_dummies(ds):
    y, m = ds.year, ds.month
    return {
        'Ramadhan':  1 if RAMADAN_MAP.get(y) == m else 0,
        'Idulfitri': 1 if IDUL_FITRI_MAP.get(y) == m else 0,
        'Natal':     1 if m == 12 else 0,
        'Imlek':     1 if m == 1  else 0,
    }

def plotly_base(title="", height=320):
    return dict(
        title=dict(text=title, font=dict(size=11, color="#555", family="IBM Plex Mono"), x=0.01),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
        font=dict(family="Syne", color="#777", size=11),
        xaxis=dict(gridcolor=PLOT_GRID, linecolor="rgba(255,255,255,0.05)",
                   tickcolor="rgba(0,0,0,0)",
                   tickfont=dict(family="IBM Plex Mono", size=10)),
        yaxis=dict(gridcolor=PLOT_GRID, linecolor="rgba(255,255,255,0.05)",
                   tickcolor="rgba(0,0,0,0)", ticksuffix="%",
                   tickfont=dict(family="IBM Plex Mono", size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.06)", borderwidth=1),
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified", height=height,
    )

def inflation_level(v):
    if v < 2.0:  return "RENDAH",            "#4dffb5", "rgba(77,255,181,0.08)"
    if v <= 4.0: return "NORMAL — TARGET BI", "#b5ff4d", "rgba(181,255,77,0.08)"
    if v <= 6.0: return "MODERAT TINGGI",     "#ffd166", "rgba(255,209,102,0.08)"
    return             "TINGGI",              "#ff6b6b", "rgba(255,107,107,0.08)"

def compute_lags(series: pd.Series) -> dict:
    """Hitung lag dari series y (scaled). series diurutkan dari lama ke baru."""
    s = series.values
    return {
        'lag1':  float(s[-1])  if len(s) >= 1  else 0.0,
        'lag3':  float(s[-3])  if len(s) >= 3  else 0.0,
        'lag6':  float(s[-6])  if len(s) >= 6  else 0.0,
        'lag12': float(s[-12]) if len(s) >= 12 else 0.0,
    }

def parse_upload(file, required_cols):
    try:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        df.columns = df.columns.str.strip()
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"Kolom tidak ditemukan: **{', '.join(missing)}**"
        df['ds'] = pd.to_datetime(df['ds'])
        return df.sort_values('ds').reset_index(drop=True), ""
    except Exception as e:
        return None, str(e)

def make_template_csv() -> bytes:
    months = pd.date_range("2025-01-01", periods=3, freq="MS")
    rows = []
    for ds in months:
        d = auto_dummies(ds)
        rows.append({
            'ds': ds.strftime('%Y-%m-%d'),
            'y': '',
            'Harga Minyak Dunia': '',
            'BI Rate': '',
            'Kurs USD/IDR': '',
            **d
        })
    return pd.DataFrame(rows).to_csv(index=False).encode()

# ══════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=False)
def load_everything():
    # ── Patch untuk PyTorch 2.6+ ──────────────────────────────────
    import torch
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)
    torch.load = patched_load

    nf = NeuralForecast.load(path=MODEL_PATH)

    # Rekonstruksi scaler dari numpy (kompatibel semua versi Python)
    from sklearn.preprocessing import StandardScaler
    sy_params  = np.load('saved_nbeatsx/scaler_y_params.npy',   allow_pickle=True).item()
    sex_params = np.load('saved_nbeatsx/scaler_exog_params.npy', allow_pickle=True).item()

    scaler_y = StandardScaler()
    scaler_y.mean_           = sy_params['mean']
    scaler_y.scale_          = sy_params['scale']
    scaler_y.var_            = sy_params['var']
    scaler_y.n_features_in_  = sy_params['n_features_in']
    scaler_y.n_samples_seen_ = sy_params['n_samples_seen']

    scaler_exog = StandardScaler()
    scaler_exog.mean_           = sex_params['mean']
    scaler_exog.scale_          = sex_params['scale']
    scaler_exog.var_            = sex_params['var']
    scaler_exog.n_features_in_  = sex_params['n_features_in']
    scaler_exog.n_samples_seen_ = sex_params['n_samples_seen']

    with open(PARAMS_PATH, 'rb') as f: bp = pickle.load(f)
    full_df = pd.read_parquet(FULL_PATH)
    df_asli = pd.read_parquet(ASLI_PATH)

    return nf, scaler_y, scaler_exog, bp, full_df, df_asli  # ← fix di sini

with st.spinner("Memuat model N-BEATSx…"):
    try:
        nf, scaler_y, scaler_exog, best_params, full_df, df_asli = load_everything()
        MODEL_OK = True
    except Exception as e:
        MODEL_OK  = False
        MODEL_ERR = str(e)

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem 0;'>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:.6rem;letter-spacing:3px;
                  color:#b5ff4d;text-transform:uppercase;margin-bottom:.5rem;'>N-BEATSx · Bayesian Opt</div>
      <div style='font-size:1.3rem;font-weight:800;color:#fff;letter-spacing:-.5px;'>InflasiCast</div>
    </div><hr/>""", unsafe_allow_html=True)

    nav = st.radio("Menu",
        ["🏠 Beranda", "📤 Upload & Prediksi", "📊 Evaluasi Model", "📋 Panduan Format"],
        label_visibility="collapsed")

    st.markdown("""<hr/>
    <div style='font-family:"IBM Plex Mono",monospace;font-size:.62rem;color:#333;line-height:1.9;'>
      MODEL&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;N-BEATSx<br/>
      OPTIMIZER&nbsp;Bayesian/Optuna<br/>
      DATA&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;189 obs bulanan<br/>
      HIST EXOG&nbsp;Lag1·3·6·12 · Minyak · BI Rate · Kurs<br/>
      FUTR EXOG&nbsp;Ramadhan · Idulfitri · Natal · Imlek
    </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    sc = "#b5ff4d" if MODEL_OK else "#ff6b6b"
    st.markdown(f"<div style='font-family:\"IBM Plex Mono\",monospace;font-size:.6rem;color:{sc};'>● MODEL {'READY' if MODEL_OK else 'ERROR'}</div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE: BERANDA
# ══════════════════════════════════════════════════════════════════
if nav == "🏠 Beranda":
    st.markdown("""<div class='hero'>
      <div class='hero-eyebrow'>Prediction Tool · Skripsi</div>
      <div class='hero-title'>InflasiCast Indonesia</div>
      <div class='hero-sub'>Upload data historis untuk memprediksi inflasi bulanan menggunakan
      N-BEATSx dengan Bayesian Optimization.</div>
    </div>""", unsafe_allow_html=True)

    if not MODEL_OK:
        st.markdown(f"<div class='alert-err'>⚠️ <b>Model gagal dimuat:</b> {MODEL_ERR}</div>",
                    unsafe_allow_html=True)
        st.stop()

    # Stats
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Observasi", f"{len(df_asli)}")
    c2.metric("Periode",
              f"{df_asli['ds'].min().strftime('%b %Y')} – {df_asli['ds'].max().strftime('%b %Y')}")
    c3.metric("Rata-rata Inflasi", f"{df_asli['y_orig'].mean()*100:.2f}%")
    c4.metric("Maks Inflasi",      f"{df_asli['y_orig'].max()*100:.2f}%")

    st.markdown("<br/><div class='panel-title'>DATA INFLASI HISTORIS</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_asli['ds'], y=df_asli['y_orig']*100,
        name="Inflasi (%)", mode="lines",
        line=dict(color=C_PRED, width=2),
        fill="tozeroy", fillcolor="rgba(181,255,77,0.06)"
    ))
    fig.update_layout(**plotly_base("INFLASI BULANAN — % YoY", height=300))
    st.plotly_chart(fig, use_container_width=True)

    # Workflow
    st.markdown("<br/><div class='panel-title'>ALUR PENGGUNAAN</div>", unsafe_allow_html=True)
    s1,s2,s3 = st.columns(3)
    for col, num, title, desc in [
        (s1,"01","Upload Data",   "Upload CSV/Excel berisi data inflasi historis dan variabel makroekonomi."),
        (s2,"02","Atur Horizon",  "Pilih 1–6 bulan ke depan dan isi nilai eksogen masa depan."),
        (s3,"03","Lihat Prediksi","Dapatkan grafik dan tabel prediksi inflasi interaktif."),
    ]:
        with col:
            st.markdown(f"""<div class='panel'>
              <div style='font-family:"IBM Plex Mono",monospace;font-size:1.8rem;color:#222;margin-bottom:.5rem;'>{num}</div>
              <div style='font-weight:700;color:#fff;margin-bottom:.4rem;'>{title}</div>
              <div style='font-size:.82rem;color:#555;line-height:1.7;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE: UPLOAD & PREDIKSI
# ══════════════════════════════════════════════════════════════════
elif nav == "📤 Upload & Prediksi":
    st.markdown("""<div style='margin-bottom:1.75rem;'>
      <div class='hero-eyebrow'>Prediction Tool</div>
      <div style='font-size:1.75rem;font-weight:800;color:#fff;letter-spacing:-.5px;'>Upload & Prediksi</div>
      <div style='font-size:.82rem;color:#555;margin-top:.4rem;'>
        Upload data historis, isi nilai eksogen masa depan, dapatkan prediksi.
      </div>
    </div>""", unsafe_allow_html=True)

    if not MODEL_OK:
        st.markdown("<div class='alert-err'>⚠️ Model belum berhasil dimuat.</div>",
                    unsafe_allow_html=True)
        st.stop()

    # ── LANGKAH 1: UPLOAD ───────────────────────────────────────────
    st.markdown("<div class='step-badge'>◆ LANGKAH 1 — Upload Data Historis</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='alert-info'>
      Upload file CSV/Excel dengan kolom: 
      <code>ds</code>, <code>y</code>, <code>Harga Minyak Dunia</code>, 
      <code>BI Rate</code>, <code>Kurs USD/IDR</code><br/>
      Kolom dummy (<code>Ramadhan</code>, <code>Idulfitri</code>, <code>Natal</code>, <code>Imlek</code>) 
      opsional — dihitung otomatis jika tidak ada.<br/>
      <b>Nilai y dalam persen desimal</b>, contoh: inflasi 3.72% → tulis <code>0.0372</code>
    </div>""", unsafe_allow_html=True)

    col_dl, col_up = st.columns([1, 2])
    with col_dl:
        st.download_button("⬇ Template CSV", make_template_csv(),
                           "template.csv", "text/csv", use_container_width=True)
    with col_up:
        uploaded = st.file_uploader("Upload file", type=["csv","xlsx"],
                                    label_visibility="collapsed")

    hist_df = None
    if uploaded:
        req = ['ds', 'y', 'Harga Minyak Dunia', 'BI Rate', 'Kurs USD/IDR']
        df_up, err = parse_upload(uploaded, req)
        if err:
            st.markdown(f"<div class='alert-err'>⚠️ {err}</div>", unsafe_allow_html=True)
        else:
            # Tambah dummy kalender jika belum ada
            for col in FUTR_EXOG:
                if col not in df_up.columns:
                    df_up[col] = df_up['ds'].apply(lambda d: auto_dummies(d)[col])

            # Scale eksogen
            # Scale y dulu
            df_up['y'] = scaler_y.transform(df_up[['y']]).flatten()

            # Hitung lag dari y yang sudah di-scale
            df_up['lag1']  = df_up['y'].shift(1)
            df_up['lag3']  = df_up['y'].shift(3)
            df_up['lag6']  = df_up['y'].shift(6)
            df_up['lag12'] = df_up['y'].shift(12)
            df_up = df_up.dropna().reset_index(drop=True)

            # Scale 7 kolom eksogen sekaligus
            EXOG_SCALE_COLS = ['Harga Minyak Dunia', 'BI Rate', 'Kurs USD/IDR', 'lag1', 'lag3', 'lag6', 'lag12']
            df_up[EXOG_SCALE_COLS] = scaler_exog.transform(df_up[EXOG_SCALE_COLS])

            df_up['unique_id'] = 'inflasi'
            hist_df = df_up
            n  = len(hist_df)
            d0 = hist_df['ds'].min().strftime('%b %Y')
            d1 = hist_df['ds'].max().strftime('%b %Y')
            st.markdown(f"<div class='alert-info'>✓ <b>{n} baris</b> valid · Periode: <b>{d0} – {d1}</b></div>",
                        unsafe_allow_html=True)
            with st.expander("🔍 Preview data (5 baris terakhir)", expanded=False):
                disp = hist_df[['ds','y'] + FUTR_EXOG].tail(5).copy()
                disp['y_pct'] = scaler_y.inverse_transform(disp[['y']]).flatten() * 100
                st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── LANGKAH 2: HORIZON ─────────────────────────────────────────
    if hist_df is not None:
        st.markdown("<br/><div class='step-badge'>◆ LANGKAH 2 — Horizon & Nilai Eksogen Masa Depan</div>",
                    unsafe_allow_html=True)

        last_ds      = hist_df['ds'].max()
        next_months  = pd.date_range(start=last_ds + pd.DateOffset(months=1),
                                     periods=6, freq='MS')
        horizon      = st.slider("Jumlah bulan yang ingin diprediksi", 1, 6, 3)
        target_months = next_months[:horizon]

        st.markdown(f"""<div class='alert-info'>
          Prediksi untuk: <b>{', '.join([m.strftime('%B %Y') for m in target_months])}</b>
        </div>""", unsafe_allow_html=True)

        future_rows = []
        for i, ds in enumerate(target_months):
            d = auto_dummies(ds)
            with st.expander(f"📅  {ds.strftime('%B %Y')}", expanded=(i==0)):
                c1,c2,c3 = st.columns(3)
                minyak = c1.number_input("Harga Minyak Dunia (USD/bbl)", 0.0, 300.0, 80.0, 1.0, key=f"m_{i}")
                bi     = c2.number_input("BI Rate (%)",                  0.0,  25.0,  6.0, 0.25,key=f"b_{i}")
                kurs   = c3.number_input("Kurs USD/IDR",              5000.0,30000.0,15500.0,50.0,key=f"k_{i}")
                cd1,cd2,cd3,cd4 = st.columns(4)
                ram = cd1.checkbox("Ramadhan",  value=bool(d['Ramadhan']),  key=f"r_{i}")
                fit = cd2.checkbox("Idulfitri", value=bool(d['Idulfitri']), key=f"f_{i}")
                nat = cd3.checkbox("Natal",     value=bool(d['Natal']),     key=f"n_{i}")
                iml = cd4.checkbox("Imlek",     value=bool(d['Imlek']),     key=f"im_{i}")
            future_rows.append({
                'ds': ds, 'unique_id': 'inflasi',
                'Harga Minyak Dunia': minyak,
                'BI Rate': bi,
                'Kurs USD/IDR': kurs,
                'Ramadhan': int(ram), 'Idulfitri': int(fit),
                'Natal': int(nat), 'Imlek': int(iml),
            })

        # ── LANGKAH 3: PREDIKSI ─────────────────────────────────────
        st.markdown("<br/><div class='step-badge'>◆ LANGKAH 3 — Jalankan Prediksi</div>",
                    unsafe_allow_html=True)
        run = st.button("▶  JALANKAN PREDIKSI")

        if run:
            with st.spinner("Model sedang berjalan…"):
                try:
                    # Scale eksogen masa depan
                    # Hitung lag dari y historis yang sudah scaled
                    # Selalu buat 6 baris future (sesuai h=6 saat training)
                    next_6_months = pd.date_range(
                        start=hist_df['ds'].max() + pd.DateOffset(months=1),
                        periods=6, freq='MS'
                    )

                    # Buat futr_df untuk semua 6 bulan
                    all_future_rows = []
                    for i, ds in enumerate(next_6_months):
                        if i < horizon:
                            # Pakai nilai yang diinput user
                            row = future_rows[i].copy()
                        else:
                            # Pakai nilai default untuk bulan di luar horizon
                            d = auto_dummies(ds)
                            row = {
                                'ds': ds, 'unique_id': 'inflasi',
                                'Harga Minyak Dunia': future_rows[-1]['Harga Minyak Dunia'],
                                'BI Rate':            future_rows[-1]['BI Rate'],
                                'Kurs USD/IDR':       future_rows[-1]['Kurs USD/IDR'],
                                **d
                            }
                        all_future_rows.append(row)

                    lag_vals = compute_lags(hist_df['y'])
                    futr_df  = pd.DataFrame(all_future_rows)

                    # Buat array 7 kolom sesuai urutan training
                    futr_7col = np.column_stack([
                        futr_df['Harga Minyak Dunia'].values,
                        futr_df['BI Rate'].values,
                        futr_df['Kurs USD/IDR'].values,
                        np.full(6, lag_vals['lag1']),
                        np.full(6, lag_vals['lag3']),
                        np.full(6, lag_vals['lag6']),
                        np.full(6, lag_vals['lag12']),
                    ])
                    futr_scaled = scaler_exog.transform(futr_7col)

                    futr_df['Harga Minyak Dunia'] = futr_scaled[:, 0]
                    futr_df['BI Rate']            = futr_scaled[:, 1]
                    futr_df['Kurs USD/IDR']       = futr_scaled[:, 2]
                    futr_df['lag1']               = futr_scaled[:, 3]
                    futr_df['lag3']               = futr_scaled[:, 4]
                    futr_df['lag6']               = futr_scaled[:, 5]
                    futr_df['lag12']              = futr_scaled[:, 6]

                    futr_df = futr_df[['unique_id', 'ds'] + ALL_EXOG]

                    # hist_df untuk predict
                    hist_pred = hist_df[['unique_id','ds','y'] + ALL_EXOG].copy()

                    # Predict — ambil hanya sejumlah horizon yang dipilih user
                    preds     = nf.predict(df=hist_pred, futr_df=futr_df)
                    pred_col  = [c for c in preds.columns if c not in ['unique_id','ds']][0]
                    y_pred_sc = preds[pred_col].values[:horizon]
                    y_pred    = scaler_y.inverse_transform(
                                    y_pred_sc.reshape(-1,1)).flatten() * 100

                    # ── TAMPILKAN ────────────────────────────────────
                    st.markdown("---")
                    st.markdown("<div class='panel-title'>HASIL PREDIKSI</div>",
                                unsafe_allow_html=True)

                    cols_res = st.columns(min(horizon, 3))
                    for i, (ds, yp) in enumerate(zip(target_months, y_pred)):
                        level, color, bg = inflation_level(yp)
                        with cols_res[i % 3]:
                            st.markdown(f"""<div class='result-block'
                              style='border-left-color:{color};background:{bg};'>
                              <div class='result-period'>{ds.strftime('%B %Y')}</div>
                              <div class='result-value'>{yp:.2f}<span class='result-pct'>%</span></div>
                              <span class='result-badge'
                                style='background:{color}22;color:{color};border:1px solid {color}44;'>
                                {level}
                              </span>
                            </div>""", unsafe_allow_html=True)

                    # Chart historis + prediksi
                    st.markdown("<br/><div class='panel-title'>GRAFIK HISTORIS + PREDIKSI</div>",
                                unsafe_allow_html=True)
                    last_24   = df_asli.tail(24)
                    last_24_y = last_24['y_orig'].values * 100

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=last_24['ds'], y=last_24_y,
                        name="Historis (24 bln)", mode="lines",
                        line=dict(color=C_HIST, width=2),
                        fill="tozeroy", fillcolor="rgba(60,60,60,0.06)"
                    ))
                    fig2.add_trace(go.Scatter(
                        x=[last_24['ds'].iloc[-1], target_months[0]],
                        y=[last_24_y[-1], y_pred[0]],
                        mode="lines", line=dict(color="#333", width=1.5, dash="dot"),
                        showlegend=False
                    ))
                    fig2.add_trace(go.Scatter(
                        x=list(target_months), y=list(y_pred),
                        name="Prediksi N-BEATSx", mode="lines+markers",
                        line=dict(color=C_PRED, width=2.5),
                        marker=dict(color=C_PRED, size=10, symbol="diamond",
                                    line=dict(color="#000", width=1.5)),
                        fill="tozeroy", fillcolor="rgba(181,255,77,0.07)"
                    ))
                    fig2.add_hline(y=2, line_color="rgba(181,255,77,0.2)", line_dash="dash")
                    fig2.add_hline(y=4, line_color="rgba(255,209,102,0.2)", line_dash="dash")
                    fig2.update_layout(**plotly_base("HISTORIS 24 BULAN + PREDIKSI (%)", height=360))
                    st.plotly_chart(fig2, use_container_width=True)

                    # Bar chart
                    st.markdown("<div class='panel-title'>PERBANDINGAN PREDIKSI PER BULAN</div>",
                                unsafe_allow_html=True)
                    fig3 = go.Figure(go.Bar(
                        x=[m.strftime("%b %Y") for m in target_months],
                        y=list(y_pred),
                        marker_color=[inflation_level(v)[1] for v in y_pred],
                        text=[f"{v:.2f}%" for v in y_pred],
                        textposition="outside",
                        textfont=dict(family="IBM Plex Mono", size=11, color="#d4d4d4"),
                        width=0.45,
                    ))
                    fig3.add_hline(y=2, line_color="rgba(181,255,77,0.25)", line_dash="dash")
                    fig3.add_hline(y=4, line_color="rgba(255,209,102,0.25)", line_dash="dash")
                    fig3.update_layout(**plotly_base("PREDIKSI INFLASI PER BULAN (%)", height=300))
                    st.plotly_chart(fig3, use_container_width=True)

                    # Tabel
                    st.markdown("<div class='panel-title'>TABEL HASIL PREDIKSI</div>",
                                unsafe_allow_html=True)
                    result_df = pd.DataFrame({
                        "Bulan":                [m.strftime("%B %Y") for m in target_months],
                        "Prediksi Inflasi (%)": np.round(y_pred, 4),
                        "Harga Minyak (USD/bbl)":[r['Harga Minyak Dunia'] for r in future_rows[:horizon]],
                        "BI Rate (%)":          [r['BI Rate']   for r in future_rows[:horizon]],
                        "Kurs USD/IDR":         [r['Kurs USD/IDR'] for r in future_rows[:horizon]],
                        "Ramadhan":             [r['Ramadhan']  for r in future_rows[:horizon]],
                        "Idulfitri":            [r['Idulfitri'] for r in future_rows[:horizon]],
                        "Natal":                [r['Natal']     for r in future_rows[:horizon]],
                        "Imlek":                [r['Imlek']     for r in future_rows[:horizon]],
                    })
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.markdown(f"<div class='alert-err'>❌ <b>Error:</b><br/><code>{e}</code></div>",
                                unsafe_allow_html=True)
                    import traceback
                    with st.expander("Detail traceback"):
                        st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════
#  PAGE: EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════
elif nav == "📊 Evaluasi Model":
    st.markdown("""<div style='margin-bottom:1.75rem;'>
      <div class='hero-eyebrow'>Model Performance</div>
      <div style='font-size:1.75rem;font-weight:800;color:#fff;letter-spacing:-.5px;'>Evaluasi Model</div>
    </div>""", unsafe_allow_html=True)

    if not MODEL_OK:
        st.markdown("<div class='alert-err'>Model belum dimuat.</div>", unsafe_allow_html=True)
        st.stop()

    st.markdown("""<div class='alert-info'>
      Evaluasi menggunakan data <b>full_df</b> yang tersimpan (seluruh periode training + test).
      Model memprediksi 1 langkah ke depan secara rekursif.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Menghitung evaluasi…"):
        try:
            # Gunakan 80% awal sebagai hist, 20% akhir sebagai "test"
            n_total   = len(full_df)
            n_train   = int(n_total * 0.8)
            train_part = full_df.iloc[:n_train].copy()
            test_part  = full_df.iloc[n_train:].copy()

            preds    = nf.predict(df=train_part,
                                  futr_df=test_part[['unique_id','ds'] + ALL_EXOG])
            pred_col = [c for c in preds.columns if c not in ['unique_id','ds']][0]
            n        = min(len(preds), len(test_part))
            y_pred_sc = preds[pred_col].values[:n]
            y_true_sc = test_part['y'].values[:n]
            ds        = test_part['ds'].values[:n]

            y_pred = scaler_y.inverse_transform(y_pred_sc.reshape(-1,1)).flatten() * 100
            y_true = scaler_y.inverse_transform(y_true_sc.reshape(-1,1)).flatten() * 100

            mae  = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            mape = np.mean(np.abs((y_true - y_pred)/(y_true+1e-9)))*100
            r2   = 1 - np.sum((y_true-y_pred)**2)/(np.sum((y_true-np.mean(y_true))**2)+1e-9)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("MAE",  f"{mae:.4f}")
            m2.metric("RMSE", f"{rmse:.4f}")
            m3.metric("MAPE", f"{mape:.2f}%")
            m4.metric("R²",   f"{r2:.4f}")

            st.markdown("<br/><div class='panel-title'>AKTUAL VS PREDIKSI</div>",
                        unsafe_allow_html=True)
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=ds, y=y_true, name="Aktual",
                mode="lines+markers", line=dict(color=C_ACTUAL, width=2),
                marker=dict(size=5)))
            fig4.add_trace(go.Scatter(x=ds, y=y_pred, name="Prediksi N-BEATSx",
                mode="lines+markers", line=dict(color=C_PRED, width=2, dash="dot"),
                marker=dict(size=6, symbol="diamond", color=C_PRED)))
            fig4.update_layout(**plotly_base("AKTUAL vs PREDIKSI (%)", height=340))
            st.plotly_chart(fig4, use_container_width=True)

            residuals = y_true - y_pred
            st.markdown("<div class='panel-title'>RESIDUAL</div>", unsafe_allow_html=True)
            fig5 = make_subplots(rows=1, cols=2,
                                 subplot_titles=["Residual per Waktu","Distribusi Residual"])
            fig5.add_trace(go.Scatter(x=ds, y=residuals, mode="lines+markers",
                line=dict(color="#444", width=1.5),
                marker=dict(size=4, color=C_PRED)), row=1, col=1)
            fig5.add_hline(y=0, line_color="rgba(181,255,77,0.3)", line_dash="dash", row=1, col=1)
            fig5.add_trace(go.Histogram(x=residuals, nbinsx=14,
                marker_color="#2a2a2a", marker_line_color=C_PRED,
                marker_line_width=1), row=1, col=2)
            fig5.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
                font=dict(family="Syne", color="#777", size=11),
                height=280, showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
            fig5.update_xaxes(gridcolor=PLOT_GRID)
            fig5.update_yaxes(gridcolor=PLOT_GRID)
            st.plotly_chart(fig5, use_container_width=True)

            st.markdown("<div class='panel-title'>TABEL DETAIL</div>", unsafe_allow_html=True)
            tbl = pd.DataFrame({
                "Bulan":         pd.to_datetime(ds).strftime("%b %Y"),
                "Aktual (%)":   np.round(y_true, 4),
                "Prediksi (%)": np.round(y_pred, 4),
                "Error":        np.round(residuals, 4),
                "APE (%)":      np.round(np.abs(residuals/(y_true+1e-9))*100, 2),
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        except Exception as e:
            st.markdown(f"<div class='alert-err'>❌ {e}</div>", unsafe_allow_html=True)
            import traceback
            with st.expander("Detail"):
                st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════
#  PAGE: PANDUAN FORMAT
# ══════════════════════════════════════════════════════════════════
elif nav == "📋 Panduan Format":
    st.markdown("""<div style='margin-bottom:1.75rem;'>
      <div class='hero-eyebrow'>Dokumentasi</div>
      <div style='font-size:1.75rem;font-weight:800;color:#fff;letter-spacing:-.5px;'>Panduan Format Data</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='panel-title'>FORMAT FILE</div>", unsafe_allow_html=True)
    fa,fb = st.columns(2)
    with fa:
        st.markdown("<div class='panel'><div style='color:#b5ff4d;font-weight:700;margin-bottom:.4rem;'>✓ CSV (.csv)</div><div style='font-size:.82rem;color:#555;'>Delimiter koma, encoding UTF-8.</div></div>",
                    unsafe_allow_html=True)
    with fb:
        st.markdown("<div class='panel'><div style='color:#b5ff4d;font-weight:700;margin-bottom:.4rem;'>✓ Excel (.xlsx)</div><div style='font-size:.82rem;color:#555;'>Sheet pertama, header baris pertama.</div></div>",
                    unsafe_allow_html=True)

    st.markdown("<br/><div class='panel-title'>CONTOH FORMAT DATA</div>", unsafe_allow_html=True)
    st.markdown("""<div class='template-code'>
<span class='col-hi'>ds</span>,<span class='col-hi'>y</span>,<span class='col-hi'>Harga Minyak Dunia</span>,<span class='col-hi'>BI Rate</span>,<span class='col-hi'>Kurs USD/IDR</span>,<span class='col-opt'>Ramadhan</span>,<span class='col-opt'>Idulfitri</span>,<span class='col-opt'>Natal</span>,<span class='col-opt'>Imlek</span>
2025-01-01,0.0257,75.23,6.00,15650,0,0,0,1
2025-02-01,0.0281,76.80,6.00,15700,0,0,0,0
2025-03-01,0.0305,81.10,6.00,15800,1,0,0,0
</div>
<div class='alert-warn' style='margin-top:.5rem;'>
  ⚠️ Nilai <code>y</code> dalam <b>desimal</b> bukan persen — contoh: inflasi 2.57% → tulis <code>0.0257</code>
</div>""", unsafe_allow_html=True)

    st.markdown("<br/><div class='panel-title'>KETERANGAN KOLOM</div>", unsafe_allow_html=True)
    kol_df = pd.DataFrame({
        "Kolom":     ["ds","y","Harga Minyak Dunia","BI Rate","Kurs USD/IDR",
                      "Ramadhan","Idulfitri","Natal","Imlek"],
        "Tipe":      ["Date","Float (desimal)","Float","Float","Float","0/1","0/1","0/1","0/1"],
        "Contoh":    ["2025-01-01","0.0372","75.23","6.00","15650","1","0","0","0"],
        "Wajib":     ["✓","✓","✓","✓","✓","Opsional","Opsional","Opsional","Opsional"],
    })
    st.dataframe(kol_df, use_container_width=True, hide_index=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.download_button("⬇ Download Template CSV", make_template_csv(),
                       "template.csv", "text/csv", use_container_width=False)