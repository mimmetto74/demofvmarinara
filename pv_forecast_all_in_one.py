# =========================
# 📊 STORICO vs PREVISIONE
# =========================
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def _coerce_date(s):
    """Forza YYYY-MM-DD e rimuove timezone/ora se presente."""
    s = pd.to_datetime(s, errors="coerce", utc=True).tz_convert(None)
    return s.normalize()

def _ensure_daily(df, date_col="date", value_col="kwh"):
    """Rende il DataFrame giornaliero con colonne ['date','kwh']."""
    if date_col not in df.columns:
        # prova a indovinare la colonna della data
        for c in df.columns:
            if str(c).lower() in ("date", "data", "giorno", "day"):
                date_col = c
                break
    if value_col not in df.columns:
        # prova a indovinare la produzione
        for c in df.columns:
            cl = str(c).lower()
            if "kwh" in cl and ("e_int" in cl or "energia" in cl or "prod" in cl):
                value_col = c
                break

    tmp = df[[date_col, value_col]].copy()
    tmp.columns = ["date", "kwh"]
    tmp["date"] = _coerce_date(tmp["date"])
    # se ci sono 15-min → raggruppa
    tmp = tmp.groupby("date", as_index=False)["kwh"].sum()
    tmp = tmp.sort_values("date")
    return tmp

def _aggregate_15min_to_daily(df_15min, date_col="ts", power_col="kwh"):
    """Aggrega curva 15min in giornaliero."""
    tmp = df_15min[[date_col, power_col]].copy()
    tmp[date_col] = _coerce_date(tmp[date_col])
    tmp = tmp.groupby(date_col, as_index=False)[power_col].sum()
    tmp.columns = ["date", "kwh"]
    return tmp

def _metrics(real, pred):
    """Calcola MAE, RMSE, MAPE su valori allineati (serie pandas)."""
    err = pred - real
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err / np.where(real==0, np.nan, real))) * 100.0
    return mae, rmse, mape

def page_storico_vs_previsione():
    st.markdown("## 📊 Confronto **Storico vs Previsione**")
    st.caption("Carica un CSV reale oppure usa quello predefinito. Le curve a 15 minuti vengono aggregate su base giornaliera.")

    # --- lateral tools (uguali alla tua V4) ---
    with st.expander("Opzioni grafico", expanded=False):
        smooth = st.checkbox("Mostra media mobile (7 giorni)", value=True)
        show_points = st.checkbox("Mostra markers", value=False)

    col_upl1, col_upl2 = st.columns([1,1])
    with col_upl1:
        st.write("### Produzione reale (CSV giornaliero)")
        real_file = st.file_uploader(
            "Carica CSV reale (colonne attese: Date, E_INT_Daily_kWh)",
            type=["csv"], key="real_csv_upl")
        if real_file is None:
            # fallback: dataset pre-caricato nella repo
            # (lo carichi dal path locale del progetto)
            try:
                real_df = pd.read_csv("Dataset_Daily_EnergiaSeparata_2020_2025.csv")
            except Exception:
                st.warning("CSV predefinito non trovato. Carica un file.")
                real_df = None
        else:
            real_df = pd.read_csv(real_file)

    with col_upl2:
        st.write("### Curva previsione (15 min)")
        st.caption("Se non hai eseguito una previsione, puoi caricare una curva 15-min (colonne attese: ts, kwh).")
        pred_15m_file = st.file_uploader("Carica curva 15-min (opzionale)", type=["csv"], key="pred15_csv_upl")

    # --- costruisci daily reale ---
    if real_df is None:
        st.stop()
    real_daily = _ensure_daily(real_df, value_col="E_INT_Daily_kWh")

    # --- costruisci daily previsto ---
    # 1) se utente ha caricato curva 15-min
    pred_daily = None
    if pred_15m_file is not None:
        df15 = pd.read_csv(pred_15m_file)
        # prova a inferire nomi
        ts_col = None
        kwh_col = None
        for c in df15.columns:
            cl = c.lower()
            if cl in ("ts", "timestamp", "time", "datetime"):
                ts_col = c
            if "kwh" in cl:
                kwh_col = c
        if ts_col is None:
            ts_col = df15.columns[0]
        if kwh_col is None:
            kwh_col = df15.columns[1]
        pred_daily = _aggregate_15min_to_daily(df15, ts_col, kwh_col)

    # 2) altrimenti se la tua app ha già le curve previsionali in sessione (come V4/V6)
    if pred_daily is None and "forecast_curves" in st.session_state:
        # st.session_state["forecast_curves"] = dict con chiavi "ieri", "oggi", "domani", "dopodomani"
        # ogni valore: DataFrame con colonne ["ts","kwh"]
        all_days = []
        for k, df15 in st.session_state["forecast_curves"].items():
            if isinstance(df15, pd.DataFrame) and {"ts", "kwh"}.issubset(set(df15.columns)):
                all_days.append(df15)
        if len(all_days):
            pred_daily = _aggregate_15min_to_daily(pd.concat(all_days, ignore_index=True))

    if pred_daily is None:
        st.info("Nessuna curva di previsione disponibile. Esegui prima una previsione o carica una curva 15-min.")
        # Mostra almeno storico reale
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=real_daily["date"], y=real_daily["kwh"],
            mode="lines+markers" if show_points else "lines",
            name="Reale (kWh)", line=dict(color="#2ecc71")
        ))
        if smooth and len(real_daily) >= 7:
            fig.add_trace(go.Scatter(
                x=real_daily["date"], y=real_daily["kwh"].rolling(7, min_periods=1).mean(),
                mode="lines", name="Reale (MM 7d)", line=dict(color="#27ae60", dash="dot")
            ))
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10),
                          xaxis_title="Data", yaxis_title="kWh")
        st.plotly_chart(fig, use_container_width=True)
        return

    # --- allinea le date e confronta ---
    merged = pd.merge(real_daily, pred_daily, on="date", how="inner", suffixes=("_real", "_pred"))
    merged.rename(columns={"kwh_real": "real_kwh", "kwh_pred": "pred_kwh"}, inplace=True)
    if merged.empty:
        st.warning("Non ci sono date in comune tra storico e previsione.")
        return

    mae, rmse, mape = _metrics(merged["real_kwh"].values, merged["pred_kwh"].values)

    # --- GRAFICO ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged["date"], y=merged["real_kwh"],
        mode="lines+markers" if show_points else "lines",
        name="Reale (kWh)", line=dict(color="#2ecc71")
    ))
    fig.add_trace(go.Scatter(
        x=merged["date"], y=merged["pred_kwh"],
        mode="lines+markers" if show_points else "lines",
        name="Previsto (kWh)", line=dict(color="#3498db")
    ))
    if smooth and len(merged) >= 7:
        fig.add_trace(go.Scatter(
            x=merged["date"], y=merged["real_kwh"].rolling(7, min_periods=1).mean(),
            mode="lines", name="Reale (MM 7d)", line=dict(color="#27ae60", dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=merged["date"], y=merged["pred_kwh"].rolling(7, min_periods=1).mean(),
            mode="lines", name="Previsto (MM 7d)", line=dict(color="#2980b9", dash="dot")
        ))
    fig.update_layout(height=460, margin=dict(l=10,r=10,t=10,b=10),
                      xaxis_title="Data", yaxis_title="kWh")
    st.plotly_chart(fig, use_container_width=True)

    # --- METRICHE ---
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (kWh)", f"{mae:,.1f}")
    c2.metric("RMSE (kWh)", f"{rmse:,.1f}")
    c3.metric("MAPE (%)", f"{mape:,.1f}")

    # --- TAB CON ERRORI ---
    merged["err_kwh"] = merged["pred_kwh"] - merged["real_kwh"]
    merged["err_%"] = np.where(merged["real_kwh"]==0, np.nan, merged["err_kwh"]/merged["real_kwh"]*100)
    st.dataframe(merged.rename(columns={
        "date":"Data",
        "real_kwh":"Reale (kWh)",
        "pred_kwh":"Previsto (kWh)",
        "err_kwh":"Errore (kWh)",
        "err_%":"Errore (%)"
    }), use_container_width=True, height=300)

    # --- DOWNLOAD CSV CONFRONTO ---
    out = io.StringIO()
    merged.to_csv(out, index=False)
    st.download_button(
        "⬇️ Scarica CSV confronto (reale vs previsto)",
        data=out.getvalue().encode("utf-8"),
        file_name="confronto_reale_previsto.csv",
        mime="text/csv"
    )

# ===== Aggancio nel menù principale =====
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📂 Storico", "🧩 Modello", "📈 Previsioni 4 giorni (15m)", "🗺️ Mappa", "📊 Storico vs Previsione"]
)

with tab1:
    page_storico()  # già esistente
with tab2:
    page_modello()  # già esistente
with tab3:
    page_previsioni_4g_pt15m()  # già esistente
with tab4:
    page_mappa()  # già esistente
with tab5:
    page_storico_vs_previsione()  # <-- nuovo

