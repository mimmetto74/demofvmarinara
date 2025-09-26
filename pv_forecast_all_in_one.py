import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from datetime import date, timedelta, datetime

# =================================
# Config
# =================================
USERNAME = "xensoramasrl_pignatelli_cosimo"
PASSWORD = "m7jfJRWcqN09lr88YWYs"

DEFAULT_LAT, DEFAULT_LON = 40.643278, 16.986083
MODEL_FILE = "pv_model.joblib"
DATA_FILE = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"

# Parametri Meteomatics disponibili con account trial
METEO_PARAMS = [
    "t_2m:C",             # Temperatura
    "wind_speed_10m:ms",  # Vento
    "precip_1h:mm",       # Precipitazioni
    "cloud_cover:frac",   # Copertura nuvolosa (frazione 0–1)
    "global_rad:W"        # Radiazione globale (se disponibile sul trial)
]

# =================================
# Funzioni
# =================================
def get_meteomatics_forecast(lat, lon, days_ahead=1):
    start = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%dT00:00:00Z")
    end = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%dT23:00:00Z")
    url = f"https://api.meteomatics.com/{start}--{end}:PT1H/{','.join(METEO_PARAMS)}/{lat},{lon}/json"

    try:
        response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD))
        response.raise_for_status()
        data = response.json()

        dfs = []
        for param in data.get("data", []):
            param_name = param["parameter"]
            series = {
                "date": [d["date"] for d in param["coordinates"][0]["dates"]],
                param_name: [d["value"] for d in param["coordinates"][0]["dates"]]
            }
            dfs.append(pd.DataFrame(series))

        if not dfs:
            return pd.DataFrame()

        df = dfs[0]
        for d in dfs[1:]:
            df = df.merge(d, on="date", how="outer")

        df["date"] = pd.to_datetime(df["date"])
        return df

    except Exception as e:
        st.error(f"Errore Meteomatics: {e}")
        return pd.DataFrame()


def train_model(df):
    features = ["G_M0_Wm2", "cloud_cover", "temperature"]
    available_features = [f for f in features if f in df.columns]

    df = df.dropna(subset=["E_INT_Daily_kWh"] + available_features)

    if len(available_features) == 0:
        st.error("❌ Nessuna feature meteo disponibile per addestrare il modello")
        return None, None

    X = df[available_features]
    y = df["E_INT_Daily_kWh"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    joblib.dump((model, available_features), MODEL_FILE)
    return mae, r2


def predict_energy(df_forecast, model_path=MODEL_FILE):
    if not os.path.exists(model_path):
        return None

    model, features = joblib.load(model_path)
    available_features = [f for f in features if f in df_forecast.columns]
    if not available_features:
        return None

    X_new = df_forecast[available_features].fillna(0)
    return model.predict(X_new)


# =================================
# Streamlit UI
# =================================
st.set_page_config(page_title="Solar Forecast - ROBOTRONIX", layout="wide")
st.title("☀️ Solar Forecast - ROBOTRONIX for IMEPOWER")

menu = st.sidebar.radio("📂 Menu", ["Analisi Storica", "Addestramento modello", "Previsione FV"])

# --- Analisi Storica ---
if menu == "Analisi Storica":
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
        st.success("✅ Dataset caricato")
        st.dataframe(df.head())
    else:
        st.error("❌ Dataset non trovato")

# --- Addestramento ---
elif menu == "Addestramento modello":
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
        mae, r2 = train_model(df)
        if mae is not None:
            st.success(f"✅ Modello multivariato addestrato - MAE={mae:.2f}, R2={r2:.2f}")
    else:
        st.error("❌ Dataset non trovato per training")

# --- Previsione ---
elif menu == "Previsione FV":
    lat = st.number_input("Latitudine", value=DEFAULT_LAT)
    lon = st.number_input("Longitudine", value=DEFAULT_LON)
    days_ahead = st.selectbox("Seleziona giorno", {"Domani": 1, "Dopodomani": 2})

    if st.button("Calcola previsione con Meteomatics"):
        df_forecast = get_meteomatics_forecast(lat, lon, days_ahead)
        if not df_forecast.empty:
            st.write("📊 Dati Meteomatics")
            st.dataframe(df_forecast.head())

            yhat = predict_energy(df_forecast)
            if yhat is not None:
                df_forecast["forecast_kWh"] = yhat
                st.line_chart(df_forecast.set_index("date")[["forecast_kWh"]])
            else:
                st.warning("⚠️ Nessun modello addestrato o variabili meteo mancanti.")
