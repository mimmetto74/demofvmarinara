import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from datetime import date, timedelta

# ===============================
# 🔒 Login semplice
# ===============================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    st.title("🔒 Accesso richiesto")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "FVMANAGER" and password == "admin2025":
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("❌ Credenziali non valide")

    return False

if not check_password():
    st.stop()

# ===============================
# Config
# ===============================
DATA_FILE = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_FILE = "pv_model.joblib"
DEFAULT_LAT, DEFAULT_LON = 40.643278, 16.986083

# ===============================
# Open-Meteo API
# ===============================
def get_openmeteo_forecast(lat, lon, days_ahead=1):
    start = date.today() + timedelta(days=days_ahead)
    end = start

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=global_radiation_sum,cloudcover"
        f"&timezone=auto"
        f"&start_date={start}&end_date={end}"
    )

    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["daily"]

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    return df

# ===============================
# Training modello
# ===============================
def train_model(df):
    df = df.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2", "cloudcover"])
    X = df[["G_M0_Wm2", "cloudcover"]]
    y = df["E_INT_Daily_kWh"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    joblib.dump(model, MODEL_FILE)
    return mae, r2

# ===============================
# Previsione
# ===============================
def forecast_day_ahead(lat, lon, days_ahead=1):
    df_forecast = get_openmeteo_forecast(lat, lon, days_ahead)
    model = joblib.load(MODEL_FILE)

    X_new = df_forecast[["global_radiation_sum", "cloudcover"]].rename(
        columns={"global_radiation_sum": "G_M0_Wm2"}
    )
    yhat = model.predict(X_new)
    return df_forecast, yhat

# ===============================
# UI Streamlit
# ===============================
st.set_page_config(page_title="Solar Forecast - ROBOTRONIX", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: orange;'>☀️ Solar Forecast - ROBOTRONIX for IMEPOWER</h1>",
    unsafe_allow_html=True
)
st.write("---")

menu = st.sidebar.radio("📂 Menu", ["📊 Analisi Storica", "🛠️ Training modello", "🔮 Previsione FV"])

# --- Analisi Storica ---
if menu == "📊 Analisi Storica":
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
        st.success("✅ Dataset caricato")
        st.line_chart(df.set_index("Date")[["E_INT_Daily_kWh", "G_M0_Wm2"]])
        st.dataframe(df.head())
    else:
        st.error("❌ Dataset non trovato")

# --- Training ---
elif menu == "🛠️ Training modello":
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
        if "cloudcover" not in df.columns:
            st.warning("⚠️ Il dataset storico non ha cloudcover → serve aggiungerla manualmente o stimarla")
        else:
            mae, r2 = train_model(df)
            st.success(f"✅ Modello addestrato - MAE={mae:.2f}, R²={r2:.3f}")
    else:
        st.error("❌ Dataset non trovato per training")

# --- Previsione ---
elif menu == "🔮 Previsione FV":
    lat = st.number_input("Latitudine", value=DEFAULT_LAT)
    lon = st.number_input("Longitudine", value=DEFAULT_LON)
    giorno = st.selectbox("Seleziona giorno", ["Domani", "Dopodomani"])
    days_ahead = 1 if giorno == "Domani" else 2

    if st.button("Calcola previsione"):
        df_forecast, yhat = forecast_day_ahead(lat, lon, days_ahead)
        df_forecast["forecast_kWh"] = yhat

        st.subheader(f"📅 Risultati previsione {giorno}")
        st.metric("Irraggiamento previsto", f"{df_forecast['global_radiation_sum'].iloc[0]:.1f} Wh/m²")
        st.metric("Nuvolosità prevista", f"{df_forecast['cloudcover'].iloc[0]:.1f} %")
        st.metric("Produzione stimata", f"{df_forecast['forecast_kWh'].iloc[0]:.1f} kWh")

        st.line_chart(df_forecast.set_index("time")[["forecast_kWh"]])
