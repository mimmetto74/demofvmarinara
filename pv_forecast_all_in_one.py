
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

CLOUD_DATA = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
CLOUD_DATA_GZ = "Dataset_Daily_EnergiaSeparata_2020_2025.csv.gz"
WIN_DATA = r"C:\Users\cpign\Desktop\demo fabio\Dataset_Daily_EnergiaSeparata_2020_2025.csv"

CLOUD_MODEL = "pv_model.joblib"
WIN_MODEL = r"C:\Users\cpign\Desktop\demo fabio\pv_model.joblib"

def pick_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

DATA_PATH = pick_existing_path([CLOUD_DATA_GZ, CLOUD_DATA, WIN_DATA])
MODEL_PATH = pick_existing_path([CLOUD_MODEL, WIN_MODEL]) or CLOUD_MODEL

DEFAULT_LAT = 40.6432780
DEFAULT_LON = 16.9860830

def load_dataset():
    if DATA_PATH is None:
        st.error("⚠️ Dataset non trovato. Carica il file nella repo o verifica il percorso locale.")
        return None
    try:
        return pd.read_csv(DATA_PATH, parse_dates=["Date"])
    except Exception as e:
        st.error(f"Errore nel caricamento del dataset: {e}")
        return None

def train_model(df):
    df = df.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"])
    train = df[df["Date"] < "2025-01-01"]
    test = df[df["Date"] >= "2025-01-01"]

    X_train, y_train = train[["G_M0_Wm2"]], train["E_INT_Daily_kWh"]
    X_test, y_test = test[["G_M0_Wm2"]], test["E_INT_Daily_kWh"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    return mae, r2

def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        st.error("Modello non trovato! Esegui prima l'addestramento.")
        return None

def forecast_day_ahead(irradiance_forecast: float) -> float:
    model = load_model()
    if model is None:
        return None
    X_future = pd.DataFrame({"G_M0_Wm2": [irradiance_forecast]})
    return float(model.predict(X_future)[0])

def get_forecast_irradiance(lat: float, lon: float, days_ahead: int = 1):
    target_date = (date.today() + timedelta(days=days_ahead)).isoformat()
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=shortwave_radiation&timezone=auto"
        f"&start_date={target_date}&end_date={target_date}"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    irr_values = data["hourly"]["shortwave_radiation"]
    hours = pd.date_range(start=target_date + " 00:00", periods=len(irr_values), freq="H")
    df_irr = pd.DataFrame({"Ora": hours, "Irraggiamento": irr_values}).set_index("Ora")
    df_irr = df_irr.resample("15T").interpolate()
    return float(df_irr["Irraggiamento"].mean()), df_irr

def estimate_power_curve(irr_series, daily_prod_forecast):
    irr_array = irr_series.values
    total_irr = irr_array.sum()
    if total_irr == 0:
        return pd.Series(np.zeros_like(irr_array), index=irr_series.index)
    kwh_curve = (irr_array / total_irr) * daily_prod_forecast
    return pd.Series(kwh_curve, index=irr_series.index)

st.set_page_config(page_title="PV Forecast Dashboard", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: orange;'>☀️ Solar Forecast - TESEO-RX for IMEPOWER</h1>",
    unsafe_allow_html=True
)
st.write("---")

st.sidebar.title("Menu")
st.sidebar.markdown("Seleziona le opzioni:")

st.header("📊 Analisi Storica")
df = load_dataset()
if df is not None:
    df = df.rename(columns={
        "E_INT_kWh": "E_INT_Daily_kWh",
        "E_Z_EVU_kWh": "E_Z_EVU_Daily_kWh"
    })
    start_date, end_date = st.date_input(
        "Intervallo date",
        [df["Date"].min().date(), df["Date"].max().date()]
    )
    mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    df_filtered = df.loc[mask]

    st.line_chart(df_filtered.set_index("Date")[["E_INT_Daily_kWh", "G_M0_Wm2"]])

    c1, c2, c3 = st.columns(3)
    c1.metric("Produzione media [kWh]", f"{df_filtered['E_INT_Daily_kWh'].mean():.1f}")
    c2.metric("Produzione max [kWh]",   f"{df_filtered['E_INT_Daily_kWh'].max():.1f}")
    c3.metric("Irraggiamento medio [W/m²]", f"{df_filtered['G_M0_Wm2'].mean():.1f}")

    st.download_button(
        "⬇️ Scarica CSV filtrato",
        df_filtered.to_csv(index=False).encode("utf-8"),
        file_name="pv_data_filtered.csv",
        mime="text/csv"
    )

st.header("🛠️ Addestramento modello")
if st.button("Addestra modello con dati storici"):
    if df is not None:
        mae, r2 = train_model(df)
        st.success(f"Modello salvato in: {MODEL_PATH}")
        st.write(f"Prestazioni su test 2025 → MAE: {mae:.2f} kWh • R²: {r2:.3f}")
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                st.download_button("⬇️ Scarica modello allenato", f, file_name="pv_model.joblib")

st.header("🔮 Previsione FV")
lat = st.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
lon = st.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")
giorno = st.selectbox("Giorno di previsione", ["Domani", "Dopodomani"])
days_ahead = 1 if giorno == "Domani" else 2

if st.button("Calcola previsione"):
    irr_mean, df_irr = get_forecast_irradiance(lat, lon, days_ahead=days_ahead)
    prod_forecast = forecast_day_ahead(irr_mean)
    if prod_forecast is not None:
        st.subheader(f"📅 Risultati ({giorno})")
        st.metric("Irraggiamento medio previsto", f"{irr_mean:.1f} W/m²")
        st.metric("Produzione stimata", f"{prod_forecast:.1f} kWh")

        prod_curve = estimate_power_curve(df_irr["Irraggiamento"], prod_forecast)
        df_plot = pd.DataFrame({
            "Irraggiamento [W/m²]": df_irr["Irraggiamento"],
            "Produzione stimata [kWh/15min]": prod_curve
        })
        st.subheader("📈 Andamento previsto ogni 15 minuti")
        st.line_chart(df_plot)
