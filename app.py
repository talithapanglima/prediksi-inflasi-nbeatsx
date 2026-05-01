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
  --bg:        #f5f5f4;
  --surface:   #ffffff;
  --surface2:  #fafaf9;
  --border:    rgba(0,0,0,0.08);
  --border2:   rgba(0,0,0,0.04);

  --accent:    #16a34a;
  --accent-lt: rgba(22,163,74,0.08);
  --accent2:   #ea580c;

  --text:      #1c1917;
  --muted:     #6b7280;
  --muted-lt:  #9ca3af;

  --danger:    #dc2626;
  --warn:      #ea580c;
  --good:      #16a34a;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg) !important;
  color: var(--text);
}
.block-container { padding: 2rem 2.5rem 3rem 2.5rem !important; max-width: 1300px; }

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hero ── */
.hero {
  background: var(--surface);
  border: 1px solid var(--border);
  border-top: 3px solid var(--accent);
  padding: 2.5rem 3rem;
  margin-bottom: 2rem;
  border-radius: 6px;
}
.hero-eyebrow {
  font-family: 'DM Mono', monospace;
  font-size: .62rem;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: .75rem;
}
.hero-title {
  font-size: 2.4rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -.5px;
  line-height: 1.15;
  margin-bottom: .75rem;
}
.hero-sub {
  font-size: .88rem;
  color: var(--muted);
  max-width: 560px;
  line-height: 1.75;
  font-weight: 300;
}

/* ── Section label ── */
.section-label {
  font-family: 'DM Mono', monospace;
  font-size: .58rem;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: var(--muted-lt);
  border-bottom: 1px solid var(--border);
  padding-bottom: .5rem;
  margin-bottom: 1.25rem;
}

/* ── Step badge ── */
.step-badge {
  display: inline-flex;
  align-items: center;
  font-family: 'DM Mono', monospace;
  font-size: .6rem;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--accent);
  background: var(--accent-lt);
  border: 1px solid rgba(74,222,128,0.2);
  padding: .3rem .85rem;
  border-radius: 100px;
  margin-bottom: 1rem;
}

/* ── Panel ── */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1.5rem;
  margin-bottom: 1.25rem;
}

/* ── Result card ── */
.result-block {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  border-radius: 6px;
  padding: 1.5rem 1.75rem;
  margin-bottom: .75rem;
}
.result-period {
  font-family: 'DM Mono', monospace;
  font-size: .6rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: .4rem;
}
.result-value {
  font-family: 'DM Mono', monospace;
  font-size: 2.8rem;
  font-weight: 500;
  color: var(--text);
  line-height: 1;
}
.result-pct { font-size: 1.4rem; color: var(--muted); }
.result-badge {
  display: inline-block;
  margin-top: .6rem;
  padding: .2rem .75rem;
  border-radius: 100px;
  font-family: 'DM Mono', monospace;
  font-size: .6rem;
  font-weight: 500;
}

/* ── Alerts ── */
.alert-info {
  background: rgba(74,222,128,0.07);
  border-left: 3px solid var(--accent);
  padding: .85rem 1rem;
  border-radius: 0 6px 6px 0;
  font-size: .83rem;
  color: var(--text);
  margin: .75rem 0;
  line-height: 1.65;
}
.alert-warn {
  background: rgba(251,146,60,0.07);
  border-left: 3px solid var(--warn);
  padding: .85rem 1rem;
  border-radius: 0 6px 6px 0;
  font-size: .83rem;
  color: var(--text);
  margin: .75rem 0;
  line-height: 1.65;
}
.alert-err {
  background: rgba(248,113,113,0.07);
  border-left: 3px solid var(--danger);
  padding: .85rem 1rem;
  border-radius: 0 6px 6px 0;
  font-size: .83rem;
  color: var(--text);
  margin: .75rem 0;
  line-height: 1.65;
}

/* ── Template code ── */
.template-code {
  font-family: 'DM Mono', monospace;
  font-size: .72rem;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem 1.25rem;
  color: var(--muted);
  overflow-x: auto;
  line-height: 1.9;
}
.col-hi  { color: var(--accent);  font-weight: 500; }
.col-opt { color: var(--accent2); font-weight: 500; }

/* ── Streamlit overrides ── */
.stButton > button {
  background: var(--accent) !important;
  color: #1c1917 !important;
  border: none !important;
  border-radius: 6px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .85rem !important;
  font-weight: 700 !important;
  padding: .6rem 1.75rem !important;
  transition: opacity .15s !important;
}
.stButton > button:hover { opacity: .88 !important; }

div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1.1rem 1.25rem;
}
div[data-testid="stMetric"] label {
  color: var(--muted) !important;
  font-size: .68rem !important;
  font-family: 'DM Mono', monospace !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--accent) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 1.6rem !important;
}

hr { border-color: var(--border) !important; }
[data-testid="stFileUploader"] {
  border: 1.5px dashed var(--border) !important;
  border-radius: 6px !important;
  background: var(--surface) !important;
}
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

HIST_EXOG = ['lag1','lag3','lag6','lag12','Harga Minyak Dunia','BI Rate','Kurs USD/IDR']
FUTR_EXOG = ['Ramadhan','Idulfitri','Natal','Imlek']
ALL_EXOG  = HIST_EXOG + FUTR_EXOG

RAMADAN_MAP    = {2010:8,2011:8,2012:7,2013:7,2014:6,2015:6,2016:6,2017:5,
                  2018:5,2019:5,2020:4,2021:4,2022:4,2023:3,2024:3,2025:3,2026:2,2027:2}
IDUL_FITRI_MAP = {2010:9,2011:9,2012:8,2013:8,2014:7,2015:7,2016:7,2017:6,
                  2018:6,2019:6,2020:5,2021:5,2022:5,2023:4,2024:4,2025:3,2026:3,2027:3}

PLOT_BG    = "#ffffff"
PLOT_PAPER = "#ffffff"
PLOT_GRID  = "rgba(0,0,0,0.05)"

C_HIST     = "#6b7280"
C_PRED     = "#16a34a"
C_ACTUAL   = "#111827"


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
        title=dict(text=title, font=dict(size=11, color="#78716c", family="DM Mono"), x=0.01),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
        font=dict(family="DM Sans", color="#78716c", size=11),
        xaxis=dict(gridcolor=PLOT_GRID, linecolor="rgba(255,255,255,0.06)",
                   tickcolor="rgba(0,0,0,0)",
                   tickfont=dict(family="DM Mono", size=10, color="#78716c")),
        yaxis=dict(gridcolor=PLOT_GRID, linecolor="rgba(255,255,255,0.06)",
                   tickcolor="rgba(0,0,0,0)", ticksuffix="%",
                   tickfont=dict(family="DM Mono", size=10, color="#78716c")),
        legend=dict(bgcolor="rgba(40,33,30,0.9)",
                    bordercolor="rgba(255,255,255,0.08)", borderwidth=1,
                    font=dict(size=11, color="#e7e5e4")),
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified", height=height,
    )

def inflation_level(v):
    if v < 2.0:  return "RENDAH",            "#60a5fa", "rgba(96,165,250,0.08)",  "rgba(96,165,250,0.15)"
    if v <= 4.0: return "NORMAL — TARGET BI", "#4ade80", "rgba(74,222,128,0.08)",  "rgba(74,222,128,0.15)"
    if v <= 6.0: return "MODERAT TINGGI",     "#fb923c", "rgba(251,146,60,0.08)",  "rgba(251,146,60,0.15)"
    return             "TINGGI",              "#f87171", "rgba(248,113,113,0.08)", "rgba(248,113,113,0.15)"

def compute_lags(series: pd.Series) -> dict:
    s = series.values
    return {
        'lag1':  float(s[-1])  if len(s) >= 1  else 0.0,
        'lag3':  float(s[-3])  if len(s) >= 3  else 0.0,
        'lag6':  float(s[-6])  if len(s) >= 6  else 0.0,
        'lag12': float(s[-12]) if len(s) >= 12 else 0.0,
    }

def smape(y_true, y_pred):
    num   = np.abs(y_true - y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(num / (denom + 1e-9)) * 100

def to_pct(y):
    y = np.array(y)

    # gunakan threshold realistis inflasi
    # inflasi normal < 20%
    if np.max(np.abs(y)) < 1:
        return y * 100
    return y

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
        rows.append({'ds': ds.strftime('%Y-%m-%d'), 'y': '',
                     'Harga Minyak Dunia': '', 'BI Rate': '', 'Kurs USD/IDR': '', **d})
    return pd.DataFrame(rows).to_csv(index=False).encode()


# ══════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_everything():
    import torch
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)
    torch.load = patched_load

    nf = NeuralForecast.load(path=MODEL_PATH)

    from sklearn.preprocessing import StandardScaler
    sy_params  = np.load('saved_nbeatsx/scaler_y_params.npy',   allow_pickle=True).item()
    sex_params = np.load('saved_nbeatsx/scaler_exog_params.npy', allow_pickle=True).item()

    scaler_y = StandardScaler()
    scaler_y.mean_           = sy_params['mean']
    scaler_y.scale_          = sy_params['scale']
    scaler_y.var_            = sy_params['var']
    scaler_y.n_features_in_  = int(sy_params['n_features_in'])
    scaler_y.n_samples_seen_ = int(sy_params['n_samples_seen'])

    scaler_exog = StandardScaler()
    scaler_exog.mean_           = sex_params['mean']
    scaler_exog.scale_          = sex_params['scale']
    scaler_exog.var_            = sex_params['var']
    scaler_exog.n_features_in_  = int(sex_params['n_features_in'])
    scaler_exog.n_samples_seen_ = int(sex_params['n_samples_seen'])

    with open(PARAMS_PATH, 'rb') as f: bp = pickle.load(f)
    full_df = pd.read_parquet(FULL_PATH)
    df_asli = pd.read_parquet(ASLI_PATH)
    return nf, scaler_y, scaler_exog, bp, full_df, df_asli

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
    <div style='padding:1.75rem 0 1.25rem 0;'>
      <div style='font-family:"DM Mono",monospace;font-size:.58rem;letter-spacing:2.5px;
                  color:#4ade80;text-transform:uppercase;margin-bottom:.5rem;'>N-BEATSx · Bayesian Opt</div>
      <div style='font-size:1.2rem;font-weight:700;color:#e7e5e4;letter-spacing:-.3px;'>
        Inflation Forecasting</div>
    </div><hr/>""", unsafe_allow_html=True)

    nav = st.radio("Menu",
        ["🏠 Home", "📤 Upload & Forecast", "📊 Model Evaluation", "📋 Data Format Guide"],
        label_visibility="collapsed")

    st.markdown("""<hr/>
    <div style='font-family:"DM Mono",monospace;font-size:.58rem;color:#57534e;line-height:2.1;'>
      MODEL&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;N-BEATSx<br/>
      OPTIMIZER&nbsp;Bayesian / Optuna<br/>
      DATA&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;189 monthly obs<br/>
      HIST EXOG&nbsp;Lag1·3·6·12 · Oil · BI Rate · FX<br/>
      FUTR EXOG&nbsp;Ramadan · Eid · Christmas · CNY
    </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
    sc = "#4ade80" if MODEL_OK else "#f87171"
    st.markdown(f"<div style='font-family:\"DM Mono\",monospace;font-size:.58rem;color:{sc};'>● MODEL {'READY' if MODEL_OK else 'ERROR'}</div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if nav == "🏠 Home":
    st.write("DEBUG y_orig mean:", np.mean(df_asli['y_orig']))
    st.write("DEBUG y_orig sample:", df_asli['y_orig'].head())
    st.markdown("""<div class='hero'>
      <div class='hero-eyebrow'>Indonesia · Deep Learning Forecasting</div>
      <div class='hero-title'>Inflation Forecasting</div>
      <div class='hero-sub'>Upload your own historical data to forecast monthly inflation 
      using N-BEATSx optimized with Bayesian Optimization — combining macroeconomic 
      variables and calendar effects.</div>
    </div>""", unsafe_allow_html=True)

    if not MODEL_OK:
        st.markdown(f"<div class='alert-err'>⚠️ <b>Model failed to load:</b> {MODEL_ERR}</div>",
                    unsafe_allow_html=True)
        st.stop()

    # Fix: gunakan to_pct() agar deteksi otomatis desimal vs persen
    inv_y_pct = to_pct(df_asli['y_orig'].values)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observations", f"{len(df_asli)}")
    c2.metric("Period",
              f"{df_asli['ds'].min().strftime('%b %Y')} – {df_asli['ds'].max().strftime('%b %Y')}")
    c3.metric("Average Inflation", f"{inv_y_pct.mean():.2f}%")
    c4.metric("Max Inflation",     f"{inv_y_pct.max():.2f}%")

    st.markdown("<br/><div class='section-label'>Historical Inflation Data</div>",
                unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_asli['ds'], y=inv_y_pct,
        name="Inflation (%)", mode="lines",
        line=dict(color=C_PRED, width=2),
        fill="tozeroy", fillcolor="rgba(74,222,128,0.07)"
    ))
    fig.add_hline(y=2, line_color="rgba(74,222,128,0.25)", line_dash="dash")
    fig.add_hline(y=4, line_color="rgba(251,146,60,0.25)", line_dash="dash")
    fig.update_layout(**plotly_base("Monthly Inflation — % YoY", height=300))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br/><div class='section-label'>How It Works</div>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    for col, num, title, desc in [
        (s1,"01","Upload Data",   "Upload a CSV/Excel file with historical inflation data and macroeconomic variables."),
        (s2,"02","Set Horizon",   "Choose 1–6 months ahead and input future exogenous variable values."),
        (s3,"03","Get Forecast",  "View interactive charts and a detailed table of predicted inflation values."),
    ]:
        with col:
            st.markdown(f"""<div class='panel'>
              <div style='font-family:"DM Mono",monospace;font-size:1.5rem;color:#3c3330;
                          font-weight:500;margin-bottom:.6rem;'>{num}</div>
              <div style='font-weight:600;color:#e7e5e4;margin-bottom:.4rem;font-size:.92rem;'>{title}</div>
              <div style='font-size:.82rem;color:#78716c;line-height:1.75;'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE: UPLOAD & FORECAST
# ══════════════════════════════════════════════════════════════════
elif nav == "📤 Upload & Forecast":
    st.markdown("""<div style='margin-bottom:2rem;'>
      <div class='hero-eyebrow'>Prediction Tool</div>
      <div style='font-size:1.75rem;font-weight:700;color:#e7e5e4;letter-spacing:-.3px;'>Upload & Forecast</div>
      <div style='font-size:.85rem;color:#78716c;margin-top:.4rem;'>
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
      <b>Column <code>y</code></b>: decimal form e.g. 3.72% → <code>0.0372</code>, 
      OR percent form e.g. <code>3.72</code> — both are auto-detected.
    </div>""", unsafe_allow_html=True)

    col_dl, col_up = st.columns([1, 2])
    with col_dl:
        st.download_button("⬇ Download Template", make_template_csv(),
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
            for col in FUTR_EXOG:
                if col not in df_up.columns:
                    df_up[col] = df_up['ds'].apply(lambda d: auto_dummies(d)[col])

            # ── Auto-detect apakah y dalam desimal atau persen ──────
            y_raw = df_up['y'].values
            if np.mean(np.abs(y_raw)) > 1.0:
                # User upload dalam persen (misal 3.72) → konversi ke desimal dulu
                df_up['y'] = df_up['y'] / 100.0
                st.markdown("<div class='alert-warn'>ℹ️ Kolom <code>y</code> terdeteksi dalam bentuk <b>persen</b> — dikonversi otomatis ke desimal sebelum scaling.</div>",
                            unsafe_allow_html=True)

            # Scale y dulu
            df_up['y'] = scaler_y.transform(df_up[['y']]).flatten()

            # Hitung lag dari y yang sudah di-scale
            df_up['lag1']  = df_up['y'].shift(1)
            df_up['lag3']  = df_up['y'].shift(3)
            df_up['lag6']  = df_up['y'].shift(6)
            df_up['lag12'] = df_up['y'].shift(12)
            df_up = df_up.dropna().reset_index(drop=True)

            # Scale 7 kolom eksogen sekaligus
            EXOG_SCALE_COLS = ['Harga Minyak Dunia','BI Rate','Kurs USD/IDR',
                               'lag1','lag3','lag6','lag12']
            df_up[EXOG_SCALE_COLS] = scaler_exog.transform(df_up[EXOG_SCALE_COLS])

            df_up['unique_id'] = 'inflasi'
            hist_df = df_up

            n  = len(hist_df)
            d0 = hist_df['ds'].min().strftime('%b %Y')
            d1 = hist_df['ds'].max().strftime('%b %Y')
            st.markdown(f"<div class='alert-info'>✓ <b>{n} valid rows</b> · Period: <b>{d0} – {d1}</b></div>",
                        unsafe_allow_html=True)
            with st.expander("🔍 Preview (last 5 rows)", expanded=False):
                disp = hist_df[['ds'] + FUTR_EXOG].tail(5).copy()
                raw_y = scaler_y.inverse_transform(hist_df[['y']].tail(5)).flatten()
                disp['inflation (%)'] = to_pct(raw_y)
                st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── STEP 2 ──────────────────────────────────────────────────────
    if hist_df is not None:
        st.markdown("<br/><div class='step-badge'>◆ STEP 2 — Horizon & Future Values</div>",
                    unsafe_allow_html=True)

        last_ds      = hist_df['ds'].max()
        next_months  = pd.date_range(start=last_ds + pd.DateOffset(months=1),
                                     periods=6, freq='MS')
        horizon      = st.slider("Months ahead to forecast", 1, 6, 3)
        target_months = next_months[:horizon]

        st.markdown(f"""<div class='alert-info'>
          Forecasting: <b>{', '.join([m.strftime('%B %Y') for m in target_months])}</b>
        </div>""", unsafe_allow_html=True)

        future_rows = []
        for i, ds in enumerate(target_months):
            d = auto_dummies(ds)
            with st.expander(f"📅  {ds.strftime('%B %Y')}", expanded=(i==0)):
                c1, c2, c3 = st.columns(3)
                minyak = c1.number_input("Oil Price (USD/bbl)",   0.0,300.0, 80.0,1.0, key=f"m_{i}")
                bi     = c2.number_input("BI Rate (%)",           0.0, 25.0,  6.0,0.25,key=f"b_{i}")
                kurs   = c3.number_input("USD/IDR Exchange Rate",5000.0,30000.0,15500.0,50.0,key=f"k_{i}")
                cd1,cd2,cd3,cd4 = st.columns(4)
                ram = cd1.checkbox("Ramadhan",  value=bool(d['Ramadhan']),  key=f"r_{i}")
                fit = cd2.checkbox("Idulfitri", value=bool(d['Idulfitri']), key=f"f_{i}")
                nat = cd3.checkbox("Natal",     value=bool(d['Natal']),     key=f"n_{i}")
                iml = cd4.checkbox("Imlek",     value=bool(d['Imlek']),     key=f"im_{i}")
            future_rows.append({
                'ds': ds, 'unique_id': 'inflasi',
                'Harga Minyak Dunia': minyak, 'BI Rate': bi, 'Kurs USD/IDR': kurs,
                'Ramadhan': int(ram), 'Idulfitri': int(fit),
                'Natal': int(nat), 'Imlek': int(iml),
            })

        # ── STEP 3 ──────────────────────────────────────────────────
        st.markdown("<br/><div class='step-badge'>◆ STEP 3 — Run Forecast</div>",
                    unsafe_allow_html=True)
        run = st.button("▶  Run Forecast")

        if run:
            with st.spinner("Running model inference…"):
                try:
                    # Selalu buat 6 baris future (h=6)
                    next_6 = pd.date_range(
                        start=hist_df['ds'].max() + pd.DateOffset(months=1),
                        periods=6, freq='MS')

                    all_future_rows = []
                    for i, ds in enumerate(next_6):
                        if i < horizon:
                            row = future_rows[i].copy()
                        else:
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

                    futr_df   = futr_df[['unique_id','ds'] + ALL_EXOG]
                    hist_pred = hist_df[['unique_id','ds','y'] + ALL_EXOG].copy()

                    
                    preds     = nf.predict(df=hist_pred, futr_df=futr_df)
                    pred_col  = [c for c in preds.columns if c not in ['unique_id','ds']][0]
                    y_pred_sc = preds[pred_col].values[:horizon]

                    # Inverse transform → desimal → persen
                    y_pred_dec = scaler_y.inverse_transform(y_pred_sc.reshape(-1,1)).flatten()
                    y_pred     = to_pct(y_pred_dec)

                    # ── RESULTS ──────────────────────────────────────
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

                    # Chart historis + prediksi
                    st.markdown("<br/><div class='section-label'>Historical + Forecast Chart</div>",
                                unsafe_allow_html=True)
                    last_24   = df_asli.tail(24)
                    last_24_y = to_pct(last_24['y_orig'].values)

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=last_24['ds'], y=last_24_y,
                        name="Historical (24 mo)", mode="lines",
                        line=dict(color=C_HIST, width=2),
                        fill="tozeroy", fillcolor="rgba(87,83,78,0.12)"
                    ))
                    fig2.add_trace(go.Scatter(
                        x=[last_24['ds'].iloc[-1], target_months[0]],
                        y=[last_24_y[-1], y_pred[0]],
                        mode="lines",
                        line=dict(color="#3c3330", width=1.5, dash="dot"),
                        showlegend=False
                    ))
                    fig2.add_trace(go.Scatter(
                        x=list(target_months), y=list(y_pred),
                        name="N-BEATSx Forecast", mode="lines+markers",
                        line=dict(color=C_PRED, width=2.5),
                        marker=dict(color=C_PRED, size=9, symbol="diamond",
                                    line=dict(color="#1c1917", width=1.5)),
                        fill="tozeroy", fillcolor="rgba(74,222,128,0.07)"
                    ))
                    fig2.add_hline(y=2, line_color="rgba(74,222,128,0.2)", line_dash="dash",
                                   annotation_text="BI lower (2%)",
                                   annotation_font_color="#78716c", annotation_font_size=10)
                    fig2.add_hline(y=4, line_color="rgba(251,146,60,0.2)", line_dash="dash",
                                   annotation_text="BI upper (4%)",
                                   annotation_font_color="#78716c", annotation_font_size=10)
                    fig2.update_layout(**plotly_base("Last 24 Months + Forecast (%)", height=360))
                    st.plotly_chart(fig2, use_container_width=True)

                    # Bar chart
                    st.markdown("<div class='section-label'>Forecast Comparison</div>",
                                unsafe_allow_html=True)
                    fig3 = go.Figure(go.Bar(
                        x=[m.strftime("%b %Y") for m in target_months],
                        y=list(y_pred),
                        marker_color=[inflation_level(v)[1] for v in y_pred],
                        marker_line_color="rgba(255,255,255,0.1)",
                        marker_line_width=1,
                        text=[f"{v:.2f}%" for v in y_pred],
                        textposition="outside",
                        textfont=dict(family="DM Mono", size=11, color="#e7e5e4"),
                        width=0.5,
                    ))
                    fig3.add_hline(y=2, line_color="rgba(74,222,128,0.25)", line_dash="dash")
                    fig3.add_hline(y=4, line_color="rgba(251,146,60,0.25)", line_dash="dash")
                    fig3.update_layout(**plotly_base("Forecast per Month (%)", height=300))
                    st.plotly_chart(fig3, use_container_width=True)

                    # Tabel
                    st.markdown("<div class='section-label'>Forecast Table</div>",
                                unsafe_allow_html=True)
                    result_df = pd.DataFrame({
                        "Month":               [m.strftime("%B %Y") for m in target_months],
                        "Forecast (%)":        np.round(y_pred, 4),
                        "Oil Price (USD/bbl)": [r['Harga Minyak Dunia'] for r in future_rows[:horizon]],
                        "BI Rate (%)":         [r['BI Rate']   for r in future_rows[:horizon]],
                        "USD/IDR":             [r['Kurs USD/IDR'] for r in future_rows[:horizon]],
                        "Ramadhan":            [r['Ramadhan']  for r in future_rows[:horizon]],
                        "Idulfitri":           [r['Idulfitri'] for r in future_rows[:horizon]],
                        "Natal":               [r['Natal']     for r in future_rows[:horizon]],
                        "Imlek":               [r['Imlek']     for r in future_rows[:horizon]],
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
      <div style='font-size:1.75rem;font-weight:700;color:#e7e5e4;letter-spacing:-.3px;'>Model Evaluation</div>
    </div>""", unsafe_allow_html=True)

    if not MODEL_OK:
        st.markdown("<div class='alert-err'>Model not loaded.</div>", unsafe_allow_html=True)
        st.stop()

    with st.spinner("Computing evaluation…"):
        try:
            n_total    = len(full_df)
            n_train    = int(n_total * 0.8)
            train_part = full_df.iloc[:n_train].copy()
            test_part  = full_df.iloc[n_train:].copy()

            preds    = nf.predict(df=train_part,
                                  futr_df=test_part[['unique_id','ds'] + ALL_EXOG])
            pred_col = [c for c in preds.columns if c not in ['unique_id','ds']][0]
            n        = min(len(preds), len(test_part))
            y_pred_sc = preds[pred_col].values[:n]
            y_true_sc = test_part['y'].values[:n]
            ds        = test_part['ds'].values[:n]

            y_pred_dec = scaler_y.inverse_transform(y_pred_sc.reshape(-1,1)).flatten()
            y_true_dec = scaler_y.inverse_transform(y_true_sc.reshape(-1,1)).flatten()
            y_pred = to_pct(y_pred_dec)
            y_true = to_pct(y_true_dec)

            mae      = np.mean(np.abs(y_true - y_pred))
            rmse     = np.sqrt(np.mean((y_true - y_pred)**2))
            smape_v  = smape(y_true, y_pred)
            r2       = 1 - np.sum((y_true-y_pred)**2)/(np.sum((y_true-np.mean(y_true))**2)+1e-9)

            st.markdown("<div class='section-label'>Error Metrics — Pseudo Test Set (last 20%)</div>",
                        unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("MAE",   f"{mae:.4f}",   delta="Mean Absolute Error")
            m2.metric("RMSE",  f"{rmse:.4f}",  delta="Root Mean Squared Error")
            m3.metric("sMAPE", f"{smape_v:.2f}%", delta="Symmetric MAPE")
            m4.metric("R²",    f"{r2:.4f}",    delta="Coefficient of Determination")

            st.markdown("<br/><div class='section-label'>Actual vs Forecast</div>",
                        unsafe_allow_html=True)
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=ds, y=y_true, name="Actual",
                mode="lines+markers", line=dict(color=C_ACTUAL, width=2),
                marker=dict(size=5, color=C_ACTUAL)))
            fig4.add_trace(go.Scatter(x=ds, y=y_pred, name="N-BEATSx Forecast",
                mode="lines+markers", line=dict(color=C_PRED, width=2, dash="dot"),
                marker=dict(size=6, symbol="diamond", color=C_PRED)))
            fig4.update_layout(**plotly_base("Actual vs Forecast (%)", height=340))
            st.plotly_chart(fig4, use_container_width=True)

            residuals = y_true - y_pred
            st.markdown("<div class='section-label'>Residual Analysis</div>",
                        unsafe_allow_html=True)
            fig5 = make_subplots(rows=1, cols=2,
                                 subplot_titles=["Residuals over Time","Distribution"])
            fig5.add_trace(go.Scatter(x=ds, y=residuals, mode="lines+markers",
                line=dict(color=C_HIST, width=1.5),
                marker=dict(size=4, color=C_PRED)), row=1, col=1)
            fig5.add_hline(y=0, line_color="rgba(74,222,128,0.35)",
                           line_dash="dash", row=1, col=1)
            fig5.add_trace(go.Histogram(x=residuals, nbinsx=14,
                marker_color="rgba(74,222,128,0.25)",
                marker_line_color=C_PRED, marker_line_width=1), row=1, col=2)
            fig5.update_layout(
                plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
                font=dict(family="DM Sans", color="#78716c", size=11),
                height=280, showlegend=False,
                margin=dict(l=10,r=10,t=40,b=10)
            )
            fig5.update_xaxes(gridcolor=PLOT_GRID)
            fig5.update_yaxes(gridcolor=PLOT_GRID)
            st.plotly_chart(fig5, use_container_width=True)

            st.markdown("<div class='section-label'>Detail Table</div>", unsafe_allow_html=True)
            tbl = pd.DataFrame({
                "Month":        pd.to_datetime(ds).strftime("%b %Y"),
                "Actual (%)":  np.round(y_true, 4),
                "Forecast (%)":np.round(y_pred, 4),
                "Error":       np.round(residuals, 4),
                "APE (%)":     np.round(np.abs(residuals/(y_true+1e-9))*100, 2),
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        except Exception as e:
            st.markdown(f"<div class='alert-err'>❌ {e}</div>", unsafe_allow_html=True)
            import traceback
            with st.expander("Detail"):
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════
#  PAGE: DATA FORMAT GUIDE
# ══════════════════════════════════════════════════════════════════
elif nav == "📋 Data Format Guide":
    st.markdown("""<div style='margin-bottom:2rem;'>
      <div class='hero-eyebrow'>Documentation</div>
      <div style='font-size:1.75rem;font-weight:700;color:#e7e5e4;letter-spacing:-.3px;'>Data Format Guide</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Accepted File Formats</div>", unsafe_allow_html=True)
    fa, fb = st.columns(2)
    with fa:
        st.markdown("""<div class='panel'>
          <div style='color:#4ade80;font-weight:600;margin-bottom:.4rem;'>✓ CSV (.csv)</div>
          <div style='font-size:.83rem;color:#78716c;'>Comma-separated, UTF-8, header on first row.</div>
        </div>""", unsafe_allow_html=True)
    with fb:
        st.markdown("""<div class='panel'>
          <div style='color:#4ade80;font-weight:600;margin-bottom:.4rem;'>✓ Excel (.xlsx)</div>
          <div style='font-size:.83rem;color:#78716c;'>First sheet used, header on first row.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/><div class='section-label'>Expected Format</div>", unsafe_allow_html=True)
    st.markdown("""<div class='template-code'>
<span class='col-hi'>ds</span>,<span class='col-hi'>y</span>,<span class='col-hi'>Harga Minyak Dunia</span>,<span class='col-hi'>BI Rate</span>,<span class='col-hi'>Kurs USD/IDR</span>,<span class='col-opt'>Ramadhan</span>,<span class='col-opt'>Idulfitri</span>,<span class='col-opt'>Natal</span>,<span class='col-opt'>Imlek</span>
2025-01-01,0.0257,75.23,6.00,15650,0,0,0,1
2025-02-01,0.0281,76.80,6.00,15700,0,0,0,0
2025-03-01,0.0305,81.10,6.00,15800,1,0,0,0
</div>
<div class='alert-info' style='margin-top:.5rem;'>
  ℹ️ Kolom <code>y</code> bisa dalam <b>desimal</b> (0.0257) <b>atau persen</b> (2.57) — keduanya auto-detected.
</div>""", unsafe_allow_html=True)

    st.markdown("<br/><div class='section-label'>Column Reference</div>", unsafe_allow_html=True)
    kol_df = pd.DataFrame({
        "Column":   ["ds","y","Harga Minyak Dunia","BI Rate","Kurs USD/IDR",
                     "Ramadhan","Idulfitri","Natal","Imlek"],
        "Type":     ["Date","Float","Float","Float","Float","0/1","0/1","0/1","0/1"],
        "Example":  ["2025-01-01","0.0372 or 3.72","75.23","6.00","15650","1","0","0","0"],
        "Required": ["✓","✓","✓","✓","✓","Optional","Optional","Optional","Optional"],
    })
    st.dataframe(kol_df, use_container_width=True, hide_index=True)

    st.markdown("""<div class='alert-warn' style='margin-top:1.25rem;'>
      ⚠️ <b>Important:</b><br/>
      • Data must be <b>monthly, sequential, and gap-free</b>.<br/>
      • At least <b>12 rows</b> needed for lag features.<br/>
      • Scaling is handled automatically — do not pre-scale your data.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.download_button("⬇ Download Template CSV", make_template_csv(),
                       "template.csv", "text/csv")