import streamlit as st
import pandas as pd
import numpy as np
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
    page_title="Inflation Forecasting — N-BEATSx",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
#  CSS — Dark warm theme, easy on the eyes
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --bg:        #f1f5f9;   /* lebih soft dari putih */
  --surface:   #ffffff;
  --surface2:  #f8fafc;

  --border:    rgba(0,0,0,0.08);
  --border2:   rgba(0,0,0,0.05);

  --accent:    #16a34a;
  --accent-lt: rgba(22,163,74,0.10);
  --accent2:   #ea580c;

  --text:      #0f172a;   /* lebih gelap biar kontras */
  --muted:     #475569;
  --muted-lt:  #94a3b8;

  --danger:    #dc2626;
  --warn:      #ea580c;
  --good:      #16a34a;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg) !important;
  color: var(--text);
}

.block-container {
  padding: 2rem 2.5rem 3rem 2.5rem !important;
  max-width: 1300px;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
  color: var(--text) !important;
}

/* Hero */
.hero {
  background: var(--surface);
  border: 1px solid var(--border);
  border-top: 3px solid var(--accent);
  padding: 2.5rem 3rem;
  margin-bottom: 2rem;
  border-radius: 8px;
}

.hero-eyebrow {
  font-family: 'DM Mono', monospace;
  font-size: .65rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: .75rem;
}

.hero-title {
  font-size: 2.3rem;
  font-weight: 700;
  color: var(--text);
  line-height: 1.2;
}

.hero-sub {
  font-size: .9rem;
  color: var(--muted);
  max-width: 560px;
  line-height: 1.6;
}

/* Section */
.section-label {
  font-family: 'DM Mono', monospace;
  font-size: .6rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted-lt);
  border-bottom: 1px solid var(--border);
  padding-bottom: .4rem;
  margin-bottom: 1rem;
}

/* Card / Panel */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.25rem;
}

/* Result */
.result-block {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  border-radius: 8px;
  padding: 1.4rem;
  margin-bottom: .75rem;
}

.result-period {
  font-family: 'DM Mono', monospace;
  font-size: .65rem;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--muted);
}

.result-value {
  font-family: 'DM Mono', monospace;
  font-size: 2.5rem;
  font-weight: 600;
  color: var(--text);
}

.result-pct {
  font-size: 1.2rem;
  color: var(--muted);
}

/* Alerts */
.alert-info {
  background: rgba(22,163,74,0.08);
  border-left: 3px solid var(--accent);
  padding: .8rem 1rem;
  border-radius: 6px;
  color: var(--text);
}

.alert-warn {
  background: rgba(234,88,12,0.08);
  border-left: 3px solid var(--warn);
  padding: .8rem 1rem;
  border-radius: 6px;
  color: var(--text);
}

.alert-err {
  background: rgba(220,38,38,0.08);
  border-left: 3px solid var(--danger);
  padding: .8rem 1rem;
  border-radius: 6px;
  color: var(--text);
}

/* Template */
.template-code {
  font-family: 'DM Mono', monospace;
  font-size: .75rem;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem;
  color: var(--muted);
}

/* Button */
.stButton > button {
  background: var(--accent) !important;
  color: white !important;
  border-radius: 6px !important;
  font-weight: 600 !important;
  padding: .6rem 1.5rem !important;
}

.stButton > button:hover {
  opacity: 0.9 !important;
}

/* Metric */
div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
}

div[data-testid="stMetricValue"] {
  color: var(--accent) !important;
  font-weight: 600 !important;
}

hr {
  border-color: var(--border) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════
# ── PATH CONFIG ─────────────────────────────────────
BASE_DIR = "saved_nbeatsx"

MODEL_PATH   = f"{BASE_DIR}/nf_model"
SCALERY_PATH = f"{BASE_DIR}/scaler_y.pkl"
SCALERX_PATH = f"{BASE_DIR}/scaler_exog.pkl"
PARAMS_PATH  = f"{BASE_DIR}/best_params_v2.pkl"
FULL_PATH    = f"{BASE_DIR}/full_df.parquet"
ASLI_PATH    = f"{BASE_DIR}/df_asli.parquet"

# ── FEATURE CONFIG ──────────────────────────────────
HIST_EXOG = [
    'lag1', 'lag3', 'lag6', 'lag12',
    'Harga Minyak Dunia', 'BI Rate', 'Kurs USD/IDR'
]

FUTR_EXOG = [
    'Ramadhan', 'Idulfitri', 'Natal', 'Imlek'
]

ALL_EXOG = HIST_EXOG + FUTR_EXOG

# ── CALENDAR MAP ────────────────────────────────────
RAMADAN_MAP = {
    2010:8,2011:8,2012:7,2013:7,2014:6,2015:6,2016:6,2017:5,
    2018:5,2019:5,2020:4,2021:4,2022:4,2023:3,2024:3,2025:3,
    2026:2,2027:2
}

IDUL_FITRI_MAP = {
    2010:9,2011:9,2012:8,2013:8,2014:7,2015:7,2016:7,2017:6,
    2018:6,2019:6,2020:5,2021:5,2022:5,2023:4,2024:4,2025:3,
    2026:3,2027:3
}

# ── PLOT STYLE (MATCH UI THEME) ─────────────────────
PLOT_BG    = "#f8fafc"   # jangan pure putih biar enak dilihat
PLOT_PAPER = "#f1f5f9"
PLOT_GRID  = "rgba(15,23,42,0.06)"  # lebih soft & modern

# ── COLOR PALETTE ───────────────────────────────────
C_HIST   = "#64748b"  # abu modern (lebih enak dari abu lama)
C_PRED   = "#16a34a"  # hijau utama
C_ACTUAL = "#0f172a"  # teks utama (lebih kontras)


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
# ── AUTO DUMMIES ─────────────────────────────────────
def auto_dummies(ds):
    if pd.isna(ds):
        return {'Ramadhan': 0, 'Idulfitri': 0, 'Natal': 0, 'Imlek': 0}

    y, m = ds.year, ds.month

    return {
        'Ramadhan':  int(RAMADAN_MAP.get(y, -1) == m),
        'Idulfitri': int(IDUL_FITRI_MAP.get(y, -1) == m),
        'Natal':     int(m == 12),
        'Imlek':     int(m == 1)   # NOTE: ini simplifikasi (belum akurat lunar)
    }


# ── PLOTLY BASE STYLE ────────────────────────────────
def plotly_base(title="", height=320):
    return dict(
        title=dict(
            text=title,
            font=dict(size=12, color="#475569", family="DM Mono"),
            x=0.01
        ),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_PAPER,
        font=dict(family="DM Sans", color="#475569", size=11),

        xaxis=dict(
            gridcolor=PLOT_GRID,
            linecolor="rgba(0,0,0,0.1)",
            tickcolor="rgba(0,0,0,0)",
            tickfont=dict(family="DM Mono", size=10, color="#64748b")
        ),

        yaxis=dict(
            gridcolor=PLOT_GRID,
            linecolor="rgba(0,0,0,0.1)",
            tickcolor="rgba(0,0,0,0)",
            ticksuffix="%",
            tickfont=dict(family="DM Mono", size=10, color="#64748b")
        ),

        legend=dict(
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.05)",
            borderwidth=1,
            font=dict(size=11, color="#334155")
        ),

        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified",
        height=height,
    )


# ── INFLATION LEVEL ──────────────────────────────────
def inflation_level(v):
    try:
        v = float(v)
    except:
        return "UNKNOWN", "#9ca3af", "rgba(156,163,175,0.08)", "rgba(156,163,175,0.15)"

    if v < 2.0:
        return "RENDAH", "#60a5fa", "rgba(96,165,250,0.08)", "rgba(96,165,250,0.15)"
    elif v <= 4.0:
        return "NORMAL — TARGET BI", "#4ade80", "rgba(74,222,128,0.08)", "rgba(74,222,128,0.15)"
    elif v <= 6.0:
        return "MODERAT TINGGI", "#fb923c", "rgba(251,146,60,0.08)", "rgba(251,146,60,0.15)"
    else:
        return "TINGGI", "#f87171", "rgba(248,113,113,0.08)", "rgba(248,113,113,0.15)"


# ── LAG COMPUTATION ──────────────────────────────────
def compute_lags(series: pd.Series) -> dict:
    if series is None or len(series) == 0:
        return {'lag1': 0.0, 'lag3': 0.0, 'lag6': 0.0, 'lag12': 0.0}

    s = series.values

    return {
        'lag1':  float(s[-1])  if len(s) >= 1  else 0.0,
        'lag3':  float(s[-3])  if len(s) >= 3  else float(s[-1]),
        'lag6':  float(s[-6])  if len(s) >= 6  else float(s[-1]),
        'lag12': float(s[-12]) if len(s) >= 12 else float(s[-1]),
    }

# ── SMAPE ───────────────────────────────────────────
def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff  = np.abs(y_true - y_pred)

    # hindari division by zero
    mask = denom != 0
    return np.mean(diff[mask] / denom[mask]) * 100 if np.any(mask) else 0.0


# ── AUTO PERCENT CONVERTER ───────────────────────────
def to_pct(y):
    y = np.array(y, dtype=float)

    # lebih robust dari max → pakai percentile (hindari outlier)
    if np.percentile(np.abs(y), 90) < 1:
        return y * 100  # dari desimal ke persen
    return y


# ── PARSE UPLOAD FILE ────────────────────────────────
def parse_upload(file, required_cols):
    try:
        # baca file
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # rapikan nama kolom
        df.columns = df.columns.str.strip()

        # cek kolom wajib
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"Kolom tidak ditemukan: **{', '.join(missing)}**"

        # parse tanggal
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

        # drop tanggal invalid
        if df['ds'].isna().any():
            return None, "Kolom 'ds' mengandung tanggal tidak valid."

        # sort
        df = df.sort_values('ds').reset_index(drop=True)

        return df, ""

    except Exception as e:
        return None, f"Error parsing file: {str(e)}"


# ── TEMPLATE CSV GENERATOR ───────────────────────────
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
            'Ramadhan': d['Ramadhan'],
            'Idulfitri': d['Idulfitri'],
            'Natal': d['Natal'],
            'Imlek': d['Imlek'],
        })

    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_everything():
    import torch

    # ⚠️ PATCH torch.load (hindari error weights_only)
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)

    torch.load = patched_load

    # ── Load model ─────────────────────────────────────────────
    nf = NeuralForecast.load(path=str(MODEL_PATH))

    # ── Load scaler params ─────────────────────────────────────
    from sklearn.preprocessing import StandardScaler

    sy_params  = np.load(SCALERY_PATH.replace(".pkl", "_params.npy"), allow_pickle=True).item()
    sex_params = np.load(SCALERX_PATH.replace(".pkl", "_params.npy"), allow_pickle=True).item()

    # ── Rebuild scaler_y ───────────────────────────────────────
    scaler_y = StandardScaler()
    scaler_y.mean_           = np.array(sy_params['mean'])
    scaler_y.scale_          = np.array(sy_params['scale'])
    scaler_y.var_            = np.array(sy_params['var'])
    scaler_y.n_features_in_  = int(sy_params['n_features_in'])
    scaler_y.n_samples_seen_ = int(sy_params['n_samples_seen'])

    # ⚠️ penting biar sklearn ga error saat transform
    scaler_y.feature_names_in_ = np.array(['y'])

    # ── Rebuild scaler_exog ────────────────────────────────────
    scaler_exog = StandardScaler()
    scaler_exog.mean_           = np.array(sex_params['mean'])
    scaler_exog.scale_          = np.array(sex_params['scale'])
    scaler_exog.var_            = np.array(sex_params['var'])
    scaler_exog.n_features_in_  = int(sex_params['n_features_in'])
    scaler_exog.n_samples_seen_ = int(sex_params['n_samples_seen'])

    # ⚠️ WAJIB: urutan fitur HARUS sama dengan saat training
    scaler_exog.feature_names_in_ = np.array([
        'Harga Minyak Dunia',
        'BI Rate',
        'Kurs USD/IDR',
        'lag1',
        'lag3',
        'lag6',
        'lag12'
    ])

    # ── Load params & data ─────────────────────────────────────
    with open(PARAMS_PATH, 'rb') as f:
        bp = pickle.load(f)

    full_df = pd.read_parquet(FULL_PATH)
    df_asli = pd.read_parquet(ASLI_PATH)

    return nf, scaler_y, scaler_exog, bp, full_df, df_asli


# ── EXECUTE LOADING ──────────────────────────────────────────────
with st.spinner("Loading model…"):
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
      <div style='font-family:"DM Mono",monospace;
                  font-size:.6rem;
                  letter-spacing:2px;
                  color:#16a34a;
                  text-transform:uppercase;
                  margin-bottom:.4rem;'>
        N-BEATSx · Bayesian Opt
      </div>

      <div style='font-size:1.15rem;
                  font-weight:700;
                  color:#1c1917;
                  letter-spacing:-.3px;'>
        Inflation Forecasting
      </div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    # ── NAVIGATION ─────────────────────────────
    nav = st.radio(
        "Menu",
        ["🏠 Home", "📤 Upload & Forecast", "📊 Model Evaluation", "📋 Data Format Guide"],
        label_visibility="collapsed"
    )

    # ── INFO MODEL ────────────────────────────
    st.markdown("""
    <hr/>
    <div style='font-family:"DM Mono",monospace;
                font-size:.6rem;
                color:#78716c;
                line-height:1.9;'>

      MODEL&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;N-BEATSx<br/>
      OPTIMIZER&nbsp;Bayesian / Optuna<br/>
      DATA&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;189 monthly obs<br/>
      HIST EXOG&nbsp;Lag1·3·6·12 · Oil · BI Rate · FX<br/>
      FUTR EXOG&nbsp;Ramadhan · Idulfitri · Natal · Imlek

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    # ── STATUS MODEL ──────────────────────────
    model_ready = False
    try:
        model_ready = MODEL_OK
    except:
        model_ready = False

    status_color = "#16a34a" if model_ready else "#dc2626"
    status_text  = "READY" if model_ready else "ERROR"

    st.markdown(
        f"""
        <div style='font-family:"DM Mono",monospace;
                    font-size:.6rem;
                    color:{status_color};
                    letter-spacing:1px;'>
            ● MODEL {status_text}
        </div>
        """,
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if nav == "🏠 Home":
    st.markdown("""<div class='hero'>
      <div class='hero-eyebrow'>Indonesia · Deep Learning Forecasting</div>
      <div class='hero-title'>Inflation Forecasting</div>
      <div class='hero-sub'>
        Upload your own historical data to forecast monthly inflation 
        using N-BEATSx optimized with Bayesian Optimization — combining 
        macroeconomic variables and calendar effects.
      </div>
    </div>""", unsafe_allow_html=True)

    # ── VALIDASI MODEL ─────────────────────────────
    if not MODEL_OK:
        st.markdown(
            f"<div class='alert-err'>⚠️ <b>Model failed to load:</b> {MODEL_ERR}</div>",
            unsafe_allow_html=True
        )
        st.stop()

    # ── VALIDASI DATA ─────────────────────────────
    if df_asli is None or len(df_asli) == 0:
        st.warning("Data historis tidak tersedia.")
        st.stop()

    if 'y_orig' not in df_asli.columns:
        st.error("Kolom 'y_orig' tidak ditemukan pada data asli.")
        st.stop()

    # ── HANDLE NAN ─────────────────────────────
    df_clean = df_asli.dropna(subset=['y_orig', 'ds']).copy()

    if len(df_clean) == 0:
        st.error("Data kosong setelah membersihkan NaN.")
        st.stop()

    # ── KONVERSI KE PERSEN ─────────────────────
    inv_y_pct = to_pct(df_clean['y_orig'].values)

    # ── METRICS ────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Observations", f"{len(df_clean)}")

    c2.metric(
        "Period",
        f"{df_clean['ds'].min().strftime('%b %Y')} – {df_clean['ds'].max().strftime('%b %Y')}"
    )

    c3.metric("Average Inflation", f"{np.mean(inv_y_pct):.2f}%")
    c4.metric("Max Inflation",     f"{np.max(inv_y_pct):.2f}%")

    # ── CHART ──────────────────────────────────
    st.markdown("<br/><div class='section-label'>Historical Inflation Data</div>",
                unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_clean['ds'],
        y=inv_y_pct,
        name="Inflation (%)",
        mode="lines",
        line=dict(color=C_PRED, width=2),
        fill="tozeroy",
        fillcolor="rgba(22,163,74,0.08)"  # disesuaikan dengan theme baru
    ))

    fig.add_hline(
        y=2,
        line_color="rgba(22,163,74,0.25)",
        line_dash="dash"
    )

    fig.add_hline(
        y=4,
        line_color="rgba(251,146,60,0.25)",
        line_dash="dash"
    )

    fig.update_layout(**plotly_base("Monthly Inflation — % YoY", height=300))
    st.plotly_chart(fig, use_container_width=True)

    # ── HOW IT WORKS ───────────────────────────
    st.markdown("<br/><div class='section-label'>How It Works</div>",
                unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)

    steps = [
        ("01", "Upload Data",
         "Upload CSV/Excel berisi data inflasi dan variabel makro."),
        ("02", "Set Horizon",
         "Pilih 1–6 bulan ke depan dan isi variabel eksogen."),
        ("03", "Get Forecast",
         "Lihat hasil prediksi dalam bentuk chart & tabel.")
    ]

    for col, (num, title, desc) in zip([s1, s2, s3], steps):
        with col:
            st.markdown(f"""
            <div class='panel'>
              <div style='font-family:"DM Mono",monospace;
                          font-size:1.4rem;
                          color:#57534e;
                          font-weight:500;
                          margin-bottom:.5rem;'>
                {num}
              </div>

              <div style='font-weight:600;
                          color:#1c1917;
                          margin-bottom:.35rem;
                          font-size:.9rem;'>
                {title}
              </div>

              <div style='font-size:.82rem;
                          color:#6b7280;
                          line-height:1.7;'>
                {desc}
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PAGE: UPLOAD & FORECAST
# ══════════════════════════════════════════════════════════════════
elif nav == "📤 Upload & Forecast":

    st.markdown("""<div style='margin-bottom:2rem;'>
      <div class='hero-eyebrow'>Prediction Tool</div>
      <div style='font-size:1.75rem;font-weight:700;color:#1c1917;letter-spacing:-.3px;'>Upload & Forecast</div>
      <div style='font-size:.85rem;color:#6b7280;margin-top:.4rem;'>
        Upload historical data, set future exogenous values, and get your inflation forecast.
      </div>
    </div>""", unsafe_allow_html=True)

    if not MODEL_OK:
        st.markdown("<div class='alert-err'>⚠️ Model not loaded.</div>", unsafe_allow_html=True)
        st.stop()

    # ── STEP 1 ──────────────────────────────────────────────────────
    st.markdown("<div class='step-badge'>◆ STEP 1 — Upload Historical Data</div>",
                unsafe_allow_html=True)

    st.markdown("""<div class='alert-info'>
      Upload CSV/Excel with columns: 
      <code>ds</code>, <code>y</code>, <code>Harga Minyak Dunia</code>, 
      <code>BI Rate</code>, <code>Kurs USD/IDR</code><br/>
      Calendar dummies are optional — auto-computed if absent.<br/>
      <b>Column <code>y</code></b>: decimal (0.0372) or percent (3.72) — auto-detected.
    </div>""", unsafe_allow_html=True)

    col_dl, col_up = st.columns([1, 2])

    with col_dl:
        st.download_button(
            "⬇ Download Template",
            make_template_csv(),
            "template.csv",
            "text/csv",
            use_container_width=True
        )

    with col_up:
        uploaded = st.file_uploader(
            "Upload file",
            type=["csv", "xlsx"],
            label_visibility="collapsed"
        )

    hist_df = None  # penting: tetap di dalam block ini

    # ── HANDLE UPLOAD ──────────────────────────────────────────────
    if uploaded is not None:

        req = ['ds', 'y', 'Harga Minyak Dunia', 'BI Rate', 'Kurs USD/IDR']
        df_up, err = parse_upload(uploaded, req)

        if err:
            st.markdown(f"<div class='alert-err'>⚠️ {err}</div>", unsafe_allow_html=True)

        else:
            # ── VALIDASI MIN DATA ─────────────────────────────────
            if len(df_up) < 12:
                st.markdown(
                    "<div class='alert-err'>⚠️ Minimal 12 baris data diperlukan untuk lag features.</div>",
                    unsafe_allow_html=True
                )
                st.stop()

            # ── AUTO DUMMIES ─────────────────────────────────────
            for col in FUTR_EXOG:
                if col not in df_up.columns:
                    df_up[col] = df_up['ds'].apply(lambda d: auto_dummies(d)[col])

            # ── DETEKSI FORMAT y ────────────────────────────────
            y_raw = df_up['y'].values

            if np.percentile(np.abs(y_raw), 90) > 1:
                df_up['y'] = df_up['y'] / 100.0
                st.markdown(
                    "<div class='alert-warn'>ℹ️ Kolom <code>y</code> terdeteksi dalam persen → dikonversi ke desimal.</div>",
                    unsafe_allow_html=True
                )

            # ── SCALE TARGET ────────────────────────────────────
            df_up['y'] = scaler_y.transform(df_up[['y']]).flatten()

            # ── LAG FEATURES ───────────────────────────────────
            df_up['lag1']  = df_up['y'].shift(1)
            df_up['lag3']  = df_up['y'].shift(3)
            df_up['lag6']  = df_up['y'].shift(6)
            df_up['lag12'] = df_up['y'].shift(12)

            df_up = df_up.dropna().reset_index(drop=True)

            # ── SCALE EXOG (ORDER HARUS SAMA) ──────────────────
            EXOG_ORDER = [
                'Harga Minyak Dunia', 'BI Rate', 'Kurs USD/IDR',
                'lag1', 'lag3', 'lag6', 'lag12'
            ]

            df_up[EXOG_ORDER] = scaler_exog.transform(df_up[EXOG_ORDER])

            # ── FINAL FORMAT ───────────────────────────────────
            df_up['unique_id'] = 'inflasi'
            hist_df = df_up

            # ── INFO ──────────────────────────────────────────
            n  = len(hist_df)
            d0 = hist_df['ds'].min().strftime('%b %Y')
            d1 = hist_df['ds'].max().strftime('%b %Y')

            st.markdown(
                f"<div class='alert-info'>✓ <b>{n} valid rows</b> · Period: <b>{d0} – {d1}</b></div>",
                unsafe_allow_html=True
            )

            # ── PREVIEW ───────────────────────────────────────
            with st.expander("🔍 Preview (last 5 rows)", expanded=False):
                disp = hist_df[['ds'] + FUTR_EXOG].tail(5).copy()

                raw_y = scaler_y.inverse_transform(
                    hist_df[['y']].tail(5)
                ).flatten()

                disp['inflation (%)'] = to_pct(raw_y)

                st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── STEP 2 ──────────────────────────────────────────────────────
    if hist_df is not None:
        st.markdown("<br/><div class='step-badge'>◆ STEP 2 — Horizon & Future Values</div>",
                    unsafe_allow_html=True)

        last_ds = hist_df['ds'].max()

        # selalu generate 6 bulan (sesuai model h=6)
        next_months = pd.date_range(
            start=last_ds + pd.DateOffset(months=1),
            periods=6,
            freq='MS'
        )

        horizon = st.slider("Months ahead to forecast", 1, 6, 3)
        target_months = next_months[:horizon]

        st.markdown(f"""<div class='alert-info'>
        Forecasting: <b>{', '.join([m.strftime('%B %Y') for m in target_months])}</b>
        </div>""", unsafe_allow_html=True)

        future_rows = []

        for i, ds in enumerate(target_months):
            d = auto_dummies(ds)

            with st.expander(f"📅  {ds.strftime('%B %Y')}", expanded=(i == 0)):
                c1, c2, c3 = st.columns(3)

                minyak = c1.number_input(
                    "Oil Price (USD/bbl)", 0.0, 300.0, 80.0, 1.0, key=f"m_{i}"
                )

                bi = c2.number_input(
                    "BI Rate (%)", 0.0, 25.0, 6.0, 0.25, key=f"b_{i}"
                )

                kurs = c3.number_input(
                    "USD/IDR Exchange Rate", 5000.0, 30000.0, 15500.0, 50.0, key=f"k_{i}"
                )

                cd1, cd2, cd3, cd4 = st.columns(4)

                ram = cd1.checkbox("Ramadhan",  value=bool(d['Ramadhan']),  key=f"r_{i}")
                fit = cd2.checkbox("Idulfitri", value=bool(d['Idulfitri']), key=f"f_{i}")
                nat = cd3.checkbox("Natal",     value=bool(d['Natal']),     key=f"n_{i}")
                iml = cd4.checkbox("Imlek",     value=bool(d['Imlek']),     key=f"im_{i}")

            # ⚠️ SIMPAN RAW (BELUM DI-SCALE)
            future_rows.append({
                'ds': ds,
                'unique_id': 'inflasi',

                # numeric exog (RAW → nanti di-scale di STEP 3)
                'Harga Minyak Dunia': float(minyak),

                # ✅ FIX PENTING: BI Rate harus desimal
                'BI Rate': float(bi) / 100.0,

                'Kurs USD/IDR': float(kurs),

                # dummy tetap 0/1
                'Ramadhan': int(ram),
                'Idulfitri': int(fit),
                'Natal': int(nat),
                'Imlek': int(iml),
            })

       # ── STEP 3 ──────────────────────────────────────────────────
        st.markdown("<br/><div class='step-badge'>◆ STEP 3 — Run Forecast</div>",
                    unsafe_allow_html=True)

        run = st.button("▶  Run Forecast")

        if run:
            with st.spinner("Running model inference…"):
                try:
                    # ── 1. Future 6 bulan (WAJIB sesuai model h=6) ─────────
                    next_6 = pd.date_range(
                        start=hist_df['ds'].max() + pd.DateOffset(months=1),
                        periods=6,
                        freq='MS'
                    )

                    all_future_rows = []

                    for i, ds in enumerate(next_6):
                        if i < horizon:
                            row = future_rows[i].copy()
                        else:
                            d = auto_dummies(ds)
                            row = {
                                'ds': ds,
                                'unique_id': 'inflasi',
                                'Harga Minyak Dunia': future_rows[-1]['Harga Minyak Dunia'],
                                'BI Rate':            future_rows[-1]['BI Rate'],
                                'Kurs USD/IDR':       future_rows[-1]['Kurs USD/IDR'],
                                **d
                            }
                        all_future_rows.append(row)

                    futr_df = pd.DataFrame(all_future_rows)

                    # ── 2. LAG (pakai historis terakhir) ───────────────────
                    lag_vals = compute_lags(hist_df['y'])

                    for k in ['lag1','lag3','lag6','lag12']:
                        futr_df[k] = lag_vals[k]

                    # ⚠️ NOTE:
                    # ini masih "static lag"
                    # (lebih advanced: recursive update, tapi ini sudah cukup OK)

                    # ── 3. Scaling (WAJIB urutan sama training) ────────────
                    EXOG_ORDER = [
                        'Harga Minyak Dunia','BI Rate','Kurs USD/IDR',
                        'lag1','lag3','lag6','lag12'
                    ]

                    futr_df[EXOG_ORDER] = scaler_exog.transform(futr_df[EXOG_ORDER])

                    # ── 4. Final format ───────────────────────────────────
                    futr_df = futr_df[['unique_id','ds'] + ALL_EXOG]
                    hist_pred = hist_df[['unique_id','ds','y'] + ALL_EXOG].copy()

                    # ── 5. Predict ────────────────────────────────────────
                    preds = nf.predict(df=hist_pred, futr_df=futr_df)

                    pred_col = [c for c in preds.columns if c not in ['unique_id','ds']][0]
                    y_pred_sc = preds[pred_col].values[:horizon]

                    # ── 6. Inverse scaling ────────────────────────────────
                    y_pred_dec = scaler_y.inverse_transform(
                        y_pred_sc.reshape(-1,1)
                    ).flatten()

                    y_pred = to_pct(y_pred_dec)

                    # ── RESULTS ──────────────────────────────────────────
                    st.markdown("---")
                    st.markdown("<div class='section-label'>Forecast Results</div>",
                                unsafe_allow_html=True)

                    cols_res = st.columns(min(horizon, 3))

                    for i, (ds, yp) in enumerate(zip(target_months, y_pred)):
                        level, color, bg, badge_bg = inflation_level(yp)

                        with cols_res[i % 3]:
                            st.markdown(f"""<div class='result-block'
                            style='border-left-color:{color};background:{bg};'>
                            <div class='result-period'>{ds.strftime('%B %Y')}</div>
                            <div class='result-value'>{yp:.2f}<span class='result-pct'>%</span></div>
                            <span class='result-badge' style='background:{badge_bg};color:{color};'>
                                {level}
                            </span>
                            </div>""", unsafe_allow_html=True)

                    # ── CHART ─────────────────────────────────────────────
                    st.markdown("<br/><div class='section-label'>Historical + Forecast Chart</div>",
                                unsafe_allow_html=True)

                    last_24   = df_asli.tail(24)
                    last_24_y = to_pct(last_24['y_orig'].values)

                    fig2 = go.Figure()

                    # historis
                    fig2.add_trace(go.Scatter(
                        x=last_24['ds'], y=last_24_y,
                        name="Historical (24 mo)",
                        mode="lines",
                        line=dict(color=C_HIST, width=2),
                        fill="tozeroy",
                        fillcolor="rgba(87,83,78,0.12)"
                    ))

                    # garis penghubung (biar smooth)
                    fig2.add_trace(go.Scatter(
                        x=[last_24['ds'].iloc[-1], target_months[0]],
                        y=[last_24_y[-1], y_pred[0]],
                        mode="lines",
                        line=dict(dash="dot", width=1),
                        showlegend=False
                    ))

                    # forecast
                    fig2.add_trace(go.Scatter(
                        x=list(target_months),
                        y=list(y_pred),
                        name="Forecast",
                        mode="lines+markers",
                        line=dict(color=C_PRED, width=2.5),
                        marker=dict(size=8)
                    ))

                    fig2.update_layout(**plotly_base("Forecast (%)", height=360))
                    st.plotly_chart(fig2, use_container_width=True)

                    # ── TABLE ─────────────────────────────────────────────
                    st.markdown("<div class='section-label'>Forecast Table</div>",
                                unsafe_allow_html=True)

                    result_df = pd.DataFrame({
                        "Month":        [m.strftime("%B %Y") for m in target_months],
                        "Forecast (%)": np.round(y_pred, 4),
                    })

                    st.dataframe(result_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.markdown(f"<div class='alert-err'>❌ <b>Error:</b><br/><code>{e}</code></div>",
                                unsafe_allow_html=True)

                    import traceback
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════
#  PAGE: MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════
elif nav == "📊 Model Evaluation":
    st.markdown("""<div style='margin-bottom:2rem;'>
      <div class='hero-eyebrow'>Performance Metrics</div>
      <div style='font-size:1.75rem;font-weight:700;color:#16a34a;;'>Model Evaluation</div>
    </div>""", unsafe_allow_html=True)

    if not MODEL_OK:
        st.markdown("<div class='alert-err'>Model not loaded.</div>", unsafe_allow_html=True)
        st.stop()

    with st.spinner("Computing evaluation…"):
        try:
            # ── 1. SPLIT DATA ─────────────────────────────
            n_total = len(full_df)
            n_train = int(n_total * 0.8)

            train_part = full_df.iloc[:n_train].copy()
            test_part  = full_df.iloc[n_train:].copy()

            EXOG_ORDER = [
                'Harga Minyak Dunia','BI Rate','Kurs USD/IDR',
                'lag1','lag3','lag6','lag12'
            ]

            missing_cols = [c for c in EXOG_ORDER if c not in full_df.columns]
            if missing_cols:
                st.error(f"Kolom exogenous tidak ditemukan: {missing_cols}")
                st.stop()

            # ── 2. PREP DATA ──────────────────────────────
            futr_df = test_part[['unique_id','ds'] + EXOG_ORDER + FUTR_EXOG].copy()
            hist_df_eval = train_part[['unique_id','ds','y'] + EXOG_ORDER + FUTR_EXOG].copy()

            # ── 3. PREDICT ────────────────────────────────
            preds = nf.predict(df=hist_df_eval, futr_df=futr_df)

            pred_col = [c for c in preds.columns if c not in ['unique_id','ds']][0]

            n = min(len(preds), len(test_part))

            y_pred_sc = preds[pred_col].values[:n]
            y_true_sc = test_part['y'].values[:n]
            ds_vals   = test_part['ds'].values[:n]

            # ── 4. INVERSE ────────────────────────────────
            y_pred = to_pct(scaler_y.inverse_transform(y_pred_sc.reshape(-1,1)).flatten())
            y_true = to_pct(scaler_y.inverse_transform(y_true_sc.reshape(-1,1)).flatten())

            # ── 5. METRICS ────────────────────────────────
            mae     = np.mean(np.abs(y_true - y_pred))
            rmse    = np.sqrt(np.mean((y_true - y_pred) ** 2))
            smape_v = smape(y_true, y_pred)

            st.markdown("<div class='section-label'>Error Metrics — Pseudo Test Set</div>", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("MAE",   f"{mae:.4f}")
            m2.metric("RMSE",  f"{rmse:.4f}")
            m3.metric("sMAPE", f"{smape_v:.2f}%")

            # ── PLOT ───────────────────────────────────
            st.markdown("<br/><div class='section-label'>Actual vs Forecast</div>", unsafe_allow_html=True)

            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=ds_vals, y=y_true, name="Actual", mode="lines+markers"))
            fig4.add_trace(go.Scatter(x=ds_vals, y=y_pred, name="Forecast", mode="lines+markers", line=dict(dash="dot")))

            fig4.update_layout(**plotly_base("Actual vs Forecast (%)", height=340))
            st.plotly_chart(fig4, use_container_width=True)

            # ── RESIDUAL ───────────────────────────────
            residuals = y_true - y_pred

            from plotly.subplots import make_subplots

            st.markdown("<div class='section-label'>Residual Analysis</div>", unsafe_allow_html=True)

            fig5 = make_subplots(rows=1, cols=2)

            fig5.add_trace(go.Scatter(x=ds_vals, y=residuals, mode="lines+markers"), row=1, col=1)
            fig5.add_hline(y=0, line_dash="dash", row=1, col=1)

            fig5.add_trace(go.Histogram(x=residuals, nbinsx=15), row=1, col=2)

            fig5.update_layout(height=280)
            st.plotly_chart(fig5, use_container_width=True)

            # ── TABLE ──────────────────────────────────
            st.markdown("<div class='section-label'>Detail Table</div>", unsafe_allow_html=True)

            tbl = pd.DataFrame({
                "Month": pd.to_datetime(ds_vals).strftime("%b %Y"),
                "Actual (%)": np.round(y_true, 4),
                "Forecast (%)": np.round(y_pred, 4),
                "Residual": np.round(residuals, 4),
            })

            st.dataframe(tbl, use_container_width=True, hide_index=True)

        except Exception as e:
            st.markdown(f"<div class='alert-err'>❌ {e}</div>", unsafe_allow_html=True)

            import traceback
            with st.expander("Detail Error"):
                st.code(traceback.format_exc())

    # ── TABLE REFERENCE ─────────────────────────────────
    st.markdown("<br/><div class='section-label'>Column Reference</div>", unsafe_allow_html=True)

    kol_df = pd.DataFrame({
        "Column":   ["ds","y","Harga Minyak Dunia","BI Rate","Kurs USD/IDR",
                     "Ramadhan","Idulfitri","Natal","Imlek"],
        "Type":     ["Date","Float","Float","Float","Float","0/1","0/1","0/1","0/1"],
        "Required": ["✓","✓","✓","✓","✓","Optional","Optional","Optional","Optional"],
    })

    st.dataframe(kol_df, use_container_width=True, hide_index=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    st.download_button(
        "⬇ Download Template CSV",
        make_template_csv(),
        "template.csv",
        "text/csv"
    )

    # ── COLUMN TABLE ────────────────────────────────────
    st.markdown("<br/><div class='section-label'>Column Reference</div>", unsafe_allow_html=True)

    kol_df = pd.DataFrame({
        "Column": [
            "ds","y","Harga Minyak Dunia","BI Rate","Kurs USD/IDR",
            "Ramadhan","Idulfitri","Natal","Imlek"
        ],
        "Type": [
            "Date","Float","Float","Float","Float",
            "Binary (0/1)","Binary (0/1)","Binary (0/1)","Binary (0/1)"
        ],
        "Example": [
            "2025-01-01","0.0372 or 3.72","75.23","6.00","15650",
            "1","0","0","0"
        ],
        "Required": [
            "✓","✓","✓","✓","✓",
            "Optional","Optional","Optional","Optional"
        ],
    })

    st.dataframe(
        kol_df,
        use_container_width=True,
        hide_index=True
    )

    # ── IMPORTANT NOTES ─────────────────────────────────
    st.markdown("""
    <div class='alert-warn' style='margin-top:1rem;'>
    ⚠️ <b>Important:</b><br/>
    • Data harus <b>bulanan (monthly)</b><br/>
    • Tidak boleh ada <b>missing month</b><br/>
    • Minimal <b>12 observasi</b> (untuk lag features)<br/>
    • Jangan melakukan <b>scaling manual</b> — sistem akan handle otomatis<br/>
    • Pastikan urutan waktu <b>ascending</b>
    </div>
    """, unsafe_allow_html=True)

    # ── DOWNLOAD TEMPLATE ───────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)

    st.download_button(
        "⬇ Download Template CSV",
        make_template_csv(),
        file_name="template_inflasi.csv",
        mime="text/csv",
        use_container_width=True
    )