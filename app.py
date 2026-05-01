import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import os
from neuralforecast import NeuralForecast

# ── PATH CONFIG (AMAN) ─────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH   = os.path.join(BASE_DIR, "saved_nbeatsx/nf_model")
SCALERY_PATH = os.path.join(BASE_DIR, "saved_nbeatsx/scaler_y.pkl")
SCALERX_PATH = os.path.join(BASE_DIR, "saved_nbeatsx/scaler_exog.pkl")
PARAMS_PATH  = os.path.join(BASE_DIR, "saved_nbeatsx/best_params_v2.pkl")
FULL_PATH    = os.path.join(BASE_DIR, "saved_nbeatsx/full_df.parquet")

# ── CONFIG UI ──────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Inflasi Indonesia",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Prediksi Inflasi Indonesia")
st.markdown("**Model: N-BEATSx (Pretrained)**")
st.markdown("---")

# ── LOAD ASSETS ───────────────────────────────────────
@st.cache_resource
def load_assets():
    with open(SCALERY_PATH, 'rb') as f:
        scaler_y = pickle.load(f)

    with open(SCALERX_PATH, 'rb') as f:
        scaler_exog = pickle.load(f)

    with open(PARAMS_PATH, 'rb') as f:
        best_v2 = pickle.load(f)

    full_df = pd.read_parquet(FULL_PATH)

    # 🔥 LOAD MODEL (NO RETRAIN)
    nf = NeuralForecast.load(path=MODEL_PATH)

    return scaler_y, scaler_exog, best_v2, full_df, nf

scaler_y, scaler_exog, best_v2, full_df, nf = load_assets()

# ── SIDEBAR ───────────────────────────────────────────
st.sidebar.header("⚙️ Input Variabel")

last_ds = full_df['ds'].max()

bulan_depan = pd.date_range(
    start=last_ds + pd.DateOffset(months=1),
    periods=6,
    freq='MS'
)

bulan_label = [b.strftime('%b %Y') for b in bulan_depan]

harga_minyak, bi_rate, kurs = [], [], []
ramadhan, idulfitri, natal, imlek = [], [], [], []

for i, b in enumerate(bulan_label):
    harga_minyak.append(st.sidebar.number_input(f"Oil {b}", 0.0, 300.0, 75.0, key=f"m{i}"))
    bi_rate.append(st.sidebar.number_input(f"BI Rate {b}", 0.0, 20.0, 6.0, key=f"b{i}"))
    kurs.append(st.sidebar.number_input(f"Kurs {b}", 0.0, 30000.0, 15500.0, key=f"k{i}"))

    ramadhan.append(int(st.sidebar.checkbox(f"Ramadhan {b}", key=f"r{i}")))
    idulfitri.append(int(st.sidebar.checkbox(f"Idulfitri {b}", key=f"f{i}")))
    natal.append(int(st.sidebar.checkbox(f"Natal {b}", key=f"n{i}")))
    imlek.append(int(st.sidebar.checkbox(f"Imlek {b}", key=f"im{i}")))

predict_btn = st.sidebar.button("🔮 Prediksi")

# ── HISTORICAL ───────────────────────────────────────
st.subheader("📊 Data Historis")

hist_data = full_df.copy()
hist_data['y_orig'] = scaler_y.inverse_transform(hist_data[['y']]).flatten()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_data['ds'],
    y=hist_data['y_orig'] * 100,
    mode='lines+markers',
    name='Inflasi'
))
st.plotly_chart(fig, use_container_width=True)

# ── PREDICT ──────────────────────────────────────────
if predict_btn:
    with st.spinner("Running forecast..."):
        try:
            # 🔥 LAG FIX
            last_y = hist_data['y'].values.copy()

            lags = []
            for i in range(6):
                lag_dict = {
                    'lag1': last_y[-1],
                    'lag3': last_y[-3] if len(last_y) >= 3 else last_y[-1],
                    'lag6': last_y[-6] if len(last_y) >= 6 else last_y[-1],
                    'lag12': last_y[-12] if len(last_y) >= 12 else last_y[-1],
                }
                lags.append(lag_dict)

            # FUTURE DF
            future_df = pd.DataFrame({
                'unique_id': ['inflasi'] * 6,
                'ds': bulan_depan,
                'Harga Minyak Dunia': harga_minyak,
                'BI Rate': bi_rate,
                'Kurs USD/IDR': kurs,
                'lag1': [l['lag1'] for l in lags],
                'lag3': [l['lag3'] for l in lags],
                'lag6': [l['lag6'] for l in lags],
                'lag12': [l['lag12'] for l in lags],
                'Ramadhan': ramadhan,
                'Idulfitri': idulfitri,
                'Natal': natal,
                'Imlek': imlek
            })

            # SCALE
            scale_cols = [
                'Harga Minyak Dunia','BI Rate','Kurs USD/IDR',
                'lag1','lag3','lag6','lag12'
            ]
            future_df[scale_cols] = scaler_exog.transform(future_df[scale_cols])

            futr_df = future_df[[
                'unique_id','ds',
                'Harga Minyak Dunia','BI Rate','Kurs USD/IDR',
                'lag1','lag3','lag6','lag12',
                'Ramadhan','Idulfitri','Natal','Imlek'
            ]]

            # 🔥 PREDICT (NO FIT)
            preds = nf.predict(futr_df=futr_df)

            # INVERSE
            pred_col = [c for c in preds.columns if c not in ['unique_id','ds']][0]

            preds['inflasi_%'] = scaler_y.inverse_transform(
                preds[[pred_col]]
            ).flatten() * 100

            # RESULT
            st.subheader("🎯 Hasil Prediksi")

            result_df = pd.DataFrame({
                "Bulan": bulan_label,
                "Inflasi (%)": preds['inflasi_%'].round(4)
            })

            st.dataframe(result_df, use_container_width=True)

            # PLOT
            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=hist_data['ds'].tail(18),
                y=hist_data['y_orig'].tail(18) * 100,
                name='Historis'
            ))

            fig2.add_trace(go.Scatter(
                x=preds['ds'],
                y=preds['inflasi_%'],
                name='Forecast'
            ))

            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

else:
    st.info("Isi input di sidebar lalu klik Prediksi")