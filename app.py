import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from neuralforecast import NeuralForecast
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── KONFIGURASI HALAMAN ───────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Inflasi Indonesia — N-BEATSx",
    page_icon="📈",
    layout="wide"
)

# ── LOAD ASET ─────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    with open('saved_nbeatsx/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    with open('saved_nbeatsx/scaler_exog.pkl', 'rb') as f:
        scaler_exog = pickle.load(f)
    with open('saved_nbeatsx/best_params_v2.pkl', 'rb') as f:
        best_v2 = pickle.load(f)
    full_df = pd.read_parquet('saved_nbeatsx/full_df.parquet')
    df_asli = pd.read_parquet('saved_nbeatsx/df_asli.parquet')
    return scaler_y, scaler_exog, best_v2, full_df, df_asli

scaler_y, scaler_exog, best_v2, full_df, df_asli = load_assets()

# ── HEADER ────────────────────────────────────────────────────────
st.title("📈 Prediksi Inflasi Indonesia")
st.markdown("**Model: N-BEATSx + Bayesian Optimization** | "
            "Data: Jan 2009 – Sep 2025 | Horizon: 6 Bulan")
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Data Historis", "ℹ️ Info Model"])

# ════════════════════════════════════════════════════════════════
# TAB 1: PREDIKSI
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Input Variabel untuk Prediksi 6 Bulan ke Depan")

    last_date    = pd.to_datetime(full_df['ds'].max())
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=6, freq='MS'
    )
    bulan_label = [d.strftime('%B %Y') for d in future_dates]

    # ── INPUT MAKROEKONOMI ────────────────────────────────────────
    st.markdown("### 📊 Variabel Makroekonomi")
    st.caption("Masukkan nilai perkiraan untuk setiap bulan prediksi")

    col_header = st.columns([2, 1, 1, 1])
    col_header[0].markdown("**Bulan**")
    col_header[1].markdown("**Harga Minyak (USD/barrel)**")
    col_header[2].markdown("**BI Rate (%)**")
    col_header[3].markdown("**Kurs USD/IDR**")

    harga_minyak_input = []
    bi_rate_input      = []
    kurs_input         = []

    for i, b in enumerate(bulan_label):
        cols   = st.columns([2, 1, 1, 1])
        cols[0].markdown(f"**{b}**")
        minyak = cols[1].number_input("", min_value=0.0, max_value=300.0,
                                       value=75.0, step=0.5,
                                       key=f"minyak_{i}",
                                       label_visibility="collapsed")
        birate = cols[2].number_input("", min_value=0.0, max_value=20.0,
                                       value=5.75, step=0.25,
                                       key=f"birate_{i}",
                                       label_visibility="collapsed")
        kurs   = cols[3].number_input("", min_value=0.0, max_value=30000.0,
                                       value=15800.0, step=100.0,
                                       key=f"kurs_{i}",
                                       label_visibility="collapsed")
        harga_minyak_input.append(minyak)
        bi_rate_input.append(birate / 100)
        kurs_input.append(kurs)

    # ── INPUT DUMMY KALENDER ──────────────────────────────────────
    st.markdown("### 📅 Dummy Kalender")
    st.caption("Centang jika bulan tersebut termasuk momen kalender")

    cal_cols = st.columns(4)
    ramadhan_input  = []
    idulfitri_input = []
    natal_input     = []
    imlek_input     = []

    with cal_cols[0]:
        st.markdown("**🌙 Ramadhan**")
        for i, b in enumerate(bulan_label):
            ramadhan_input.append(int(st.checkbox(b, key=f"ram_{i}")))
    with cal_cols[1]:
        st.markdown("**🕌 Idulfitri**")
        for i, b in enumerate(bulan_label):
            idulfitri_input.append(int(st.checkbox(b, key=f"idul_{i}")))
    with cal_cols[2]:
        st.markdown("**🎄 Natal**")
        for i, b in enumerate(bulan_label):
            natal_input.append(int(st.checkbox(b, key=f"nat_{i}")))
    with cal_cols[3]:
        st.markdown("**🧧 Imlek**")
        for i, b in enumerate(bulan_label):
            imlek_input.append(int(st.checkbox(b, key=f"imlek_{i}")))

    st.markdown("---")
    predict_btn = st.button("🔮 Jalankan Prediksi",
                             type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("⏳ Model sedang dilatih... estimasi 3–5 menit"):
            try:
                num_cols = ['Harga Minyak Dunia', 'BI Rate', 'Kurs USD/IDR',
                            'lag1', 'lag3', 'lag6', 'lag12']

                # Ambil nilai lag dari data terakhir (unscaled)
                last_vals = df_asli.sort_values('ds')['y_orig'].values
                lag1_val  = last_vals[-1]
                lag3_val  = last_vals[-3]
                lag6_val  = last_vals[-6]
                lag12_val = last_vals[-12]

                # Buat future exog & scale
                future_exog_raw = pd.DataFrame({
                    'Harga Minyak Dunia': harga_minyak_input,
                    'BI Rate'           : bi_rate_input,
                    'Kurs USD/IDR'      : kurs_input,
                    'lag1'              : [lag1_val]  * 6,
                    'lag3'              : [lag3_val]  * 6,
                    'lag6'              : [lag6_val]  * 6,
                    'lag12'             : [lag12_val] * 6,
                })

                # futr_df untuk predict
                futr_df = pd.DataFrame({
                    'unique_id': 'inflasi',
                    'ds'       : future_dates,
                    'Ramadhan' : ramadhan_input,
                    'Idulfitri': idulfitri_input,
                    'Natal'    : natal_input,
                    'Imlek'    : imlek_input,
                })

                # Bangun & fit model
                model_app = NBEATSx(
                    h=6,
                    input_size=best_v2['input_size'],
                    stack_types=['trend', 'seasonality'],
                    n_blocks=best_v2['n_blocks'],
                    mlp_units=[[best_v2['hidden_size'], best_v2['hidden_size']],
                               [best_v2['hidden_size'], best_v2['hidden_size']]],
                    learning_rate=best_v2['lr'],
                    max_steps=best_v2['max_steps'],
                    dropout_prob_theta=best_v2['dropout'],
                    hist_exog_list=['lag1', 'lag3', 'lag6', 'lag12',
                                    'Harga Minyak Dunia', 'BI Rate', 'Kurs USD/IDR'],
                    futr_exog_list=['Ramadhan', 'Idulfitri', 'Natal', 'Imlek'],
                    scaler_type=None
                )

                nf_app = NeuralForecast(models=[model_app], freq='MS')
                nf_app.fit(df=full_df)
                preds  = nf_app.predict(futr_df=futr_df)

                preds['inflasi_pct'] = scaler_y.inverse_transform(
                    preds[['NBEATSx']]
                ).flatten() * 100

                # ── HASIL ─────────────────────────────────────────
                st.success("✅ Prediksi berhasil!")
                st.markdown("### 🎯 Hasil Prediksi Inflasi")

                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    hasil_df = pd.DataFrame({
                        'Bulan'              : bulan_label,
                        'Prediksi Inflasi (%)': preds['inflasi_pct'].round(4).values
                    })
                    st.dataframe(hasil_df, hide_index=True,
                                 use_container_width=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rata-rata", f"{preds['inflasi_pct'].mean():.4f}%")
                    m2.metric("Min",       f"{preds['inflasi_pct'].min():.4f}%")
                    m3.metric("Max",       f"{preds['inflasi_pct'].max():.4f}%")

                with res_col2:
                    hist_plot = df_asli.sort_values('ds').tail(24)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist_plot['ds'],
                        y=hist_plot['y_orig'] * 100,
                        mode='lines+markers',
                        name='Historis Aktual',
                        line=dict(color='#2563EB', width=2),
                        marker=dict(size=5)
                    ))
                    fig.add_trace(go.Scatter(
                        x=preds['ds'],
                        y=preds['inflasi_pct'],
                        mode='lines+markers',
                        name='Prediksi Masa Depan',
                        line=dict(color='#F59E0B', width=2.5, dash='dash'),
                        marker=dict(size=8, symbol='diamond')
                    ))
                    fig.add_trace(go.Scatter(
                        x=list(preds['ds']) + list(preds['ds'])[::-1],
                        y=list(preds['inflasi_pct'] * 1.15) +
                          list(preds['inflasi_pct'] * 0.85)[::-1],
                        fill='toself',
                        fillcolor='rgba(245,158,11,0.15)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Interval ±15%'
                    ))
                    fig.add_vline(x=str(future_dates[0]),
                                  line_dash='dot', line_color='gray',
                                  annotation_text='Mulai Prediksi')
                    fig.update_layout(
                        title='Prediksi Inflasi 6 Bulan ke Depan',
                        xaxis_title='Tanggal',
                        yaxis_title='Inflasi (%)',
                        hovermode='x unified',
                        height=420,
                        legend=dict(orientation='h', y=-0.2)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    else:
        st.info("👆 Isi input di atas lalu klik **Jalankan Prediksi**")

# ════════════════════════════════════════════════════════════════
# TAB 2: DATA HISTORIS
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Data Historis Inflasi Indonesia (2009–2025)")

    hist_full = df_asli.sort_values('ds').copy()
    hist_full['Inflasi (%)'] = (hist_full['y_orig'] * 100).round(4)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hist_full['ds'],
        y=hist_full['Inflasi (%)'],
        mode='lines+markers',
        name='Inflasi (%)',
        line=dict(color='#2563EB', width=2),
        marker=dict(size=4),
        hovertemplate='%{x|%b %Y}: %{y:.4f}%<extra></extra>'
    ))
    fig2.update_layout(
        title='Inflasi Indonesia Jan 2009 – Sep 2025',
        xaxis_title='Tanggal',
        yaxis_title='Inflasi (%)',
        hovermode='x unified',
        height=450
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        hist_full[['ds', 'Inflasi (%)']].rename(
            columns={'ds': 'Tanggal'}
        ).sort_values('Tanggal', ascending=False),
        use_container_width=True, hide_index=True
    )

# ════════════════════════════════════════════════════════════════
# TAB 3: INFO MODEL
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ Informasi Model N-BEATSx")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🏆 Hyperparameter Terbaik")
        params_df = pd.DataFrame({
            'Hyperparameter': ['input_size', 'learning_rate', 'max_steps',
                               'hidden_size', 'n_blocks', 'dropout'],
            'Nilai'         : [best_v2['input_size'],
                               f"{best_v2['lr']:.6f}",
                               best_v2['max_steps'],
                               best_v2['hidden_size'],
                               str(best_v2['n_blocks']),
                               best_v2['dropout']]
        })
        st.dataframe(params_df, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("### 📋 Spesifikasi Model")
        st.info("""
        - **Model**: N-BEATSx
        - **Horizon**: 6 bulan ke depan
        - **Stack**: Trend + Seasonality
        - **Variabel Historis**: lag1, lag3, lag6, lag12,
          Harga Minyak, BI Rate, Kurs USD/IDR
        - **Variabel Future**: Ramadhan, Idulfitri, Natal, Imlek
        - **Optimasi**: Bayesian Optimization (Optuna) — 40 trials
        - **Data**: Jan 2009 – Sep 2025
        """)

    st.markdown("### 📈 Performa Model")
    perf_df = pd.DataFrame({
        'Split'  : ['Validasi', 'Test'],
        'MAE'    : [0.005910, 0.006035],
        'RMSE'   : [0.007546, 0.006980],
        'SMAPE'  : ['20.16%', '48.01%*']
    })
    st.dataframe(perf_df, hide_index=True, use_container_width=True)
    st.caption("*sMAPE Test dipengaruhi deflasi Februari 2025")