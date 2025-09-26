import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ======================
# LOGIN
# ======================
def check_password():
    if "password_ok" not in st.session_state:
        st.session_state["password_ok"] = False
    if st.session_state["password_ok"]:
        return True
    st.title("🔐 Accesso richiesto")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "FVMANAGER" and pwd == "admin2025":
            st.session_state["password_ok"] = True
            st.success("✅ Accesso consentito")
            st.rerun()
        else:
            st.error("❌ Credenziali errate")
    return False

if not check_password():
    st.stop()

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="🌞 Solar Forecast - ROBOTRONIX", layout="wide")
st.title("🌞 Solar Forecast - ROBOTRONIX for IMEPOWER")

DEFAULT_LAT = 40.643278
DEFAULT_LON = 16.986083
DATA_PATH = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
MODEL_PATH = "pv_model.joblib"

# ======================
# TRAIN MODEL
# ======================
def train_model():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"])
    X = df[["G_M0_Wm2"]]
    y = df["E_INT_Daily_kWh"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mae, r2

# ======================
# OPEN-METEO API
# ======================
def get_openmeteo_forecast(lat, lon, days_ahead=1):
    start_date = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    end_date = start_date
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=shortwave_radiation_sum,cloudcover_mean"
        f"&timezone=auto&start_date={start_date}&end_date={end_date}"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "radiation": data["daily"]["shortwave_radiation_sum"],
        "cloud": data["daily"]["cloudcover_mean"]
    })
    return df

def forecast_day(lat, lon, days_ahead):
    model = joblib.load(MODEL_PATH)
    df_forecast = get_openmeteo_forecast(lat, lon, days_ahead)
    yhat = model.predict(df_forecast[["radiation"]])
    df_forecast["forecast_kWh"] = yhat
    return df_forecast

# ======================
# UI
# ======================
st.sidebar.header("☀️ Menu")
menu = st.sidebar.radio("Seleziona:", ["📊 Analisi Storica", "🛠️ Addestramento modello", "🔮 Previsione FV"])

if menu == "📊 Analisi Storica":
    st.subheader("📊 Analisi Storica")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        st.write(df.head())
        st.line_chart(df.set_index("Date")[["E_INT_Daily_kWh", "G_M0_Wm2"]])
    else:
        st.error("Dataset non trovato!")

elif menu == "🛠️ Addestramento modello":
    st.subheader("🛠️ Addestramento modello")
    if os.path.exists(DATA_PATH):
        mae, r2 = train_model()
        st.success(f"✅ Modello addestrato. MAE={mae:.2f}, R²={r2:.3f}")
    else:
        st.error("Dataset non trovato per training")

elif menu == "🔮 Previsione FV":
    st.subheader("🔮 Previsione FV con Open-Meteo")
    lat = st.number_input("Latitudine", value=DEFAULT_LAT)
    lon = st.number_input("Longitudine", value=DEFAULT_LON)
    option = st.selectbox("Quando?", ["Domani", "Dopodomani"])
    days_ahead = 1 if option == "Domani" else 2
    if st.button("Calcola previsione"):
        try:
            df_forecast = forecast_day(lat, lon, days_ahead)
            st.write(df_forecast)
            st.line_chart(df_forecast.set_index("date")[["forecast_kWh", "radiation", "cloud"]])
        except Exception as e:
            st.error(f"Errore durante la previsione: {e}")
