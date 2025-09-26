import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =============================
# Config - Percorsi
# =============================
MODEL_PATH = "pv_model.joblib"

# =============================
# Funzione autenticazione
# =============================
def check_password():
    def password_entered():
        if (
            st.session_state["username"] == "FVMANAGER"
            and st.session_state["password"] == "admin2025"
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("❌ Username o password errati")
        return False
    else:
        return True

# =============================
# Funzioni modello
# =============================
def train_model(df):
    df = df.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"])
    X = df[["G_M0_Wm2", "cloud_cover"]]
    y = df["E_INT_Daily_kWh"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mae, r2

def forecast_day_ahead(lat, lon, days_ahead):
    url = "https://api.open-meteo.com/v1/forecast"
    start = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    end = start
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "global_radiation_sum,cloudcover",
        "timezone": "auto",
        "start_date": start,
        "end_date": end,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame({
        "Date": data["daily"]["time"],
        "G_M0_Wm2": data["daily"]["global_radiation_sum"],
        "cloud_cover": data["daily"]["cloudcover"]
    })
    model = joblib.load(MODEL_PATH)
    yhat = model.predict(df[["G_M0_Wm2", "cloud_cover"]])
    df["Forecast_kWh"] = yhat
    return df

# =============================
# Streamlit App
# =============================
if check_password():
    st.title("🌞 Solar Forecast - ROBOTRONIX for IMEPOWER")

    st.header("📊 Addestramento modello")
    uploaded = st.file_uploader("Carica dataset CSV con radiazione, cloud_cover e produzione", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=["Date"])
        st.dataframe(df.head())
        if st.button("🔧 Addestra modello"):
            mae, r2 = train_model(df)
            st.success(f"✅ Modello addestrato - MAE: {mae:.2f}, R²: {r2:.3f}")

    st.header("🔮 Previsione FV con Open-Meteo")
    lat = st.number_input("Latitudine", value=40.643278)
    lon = st.number_input("Longitudine", value=16.986083)
    days = st.selectbox("Quando?", ["Domani", "Dopodomani"])
    days_ahead = 1 if days == "Domani" else 2

    if st.button("Calcola previsione"):
        try:
            df_forecast = forecast_day_ahead(lat, lon, days_ahead)
            st.dataframe(df_forecast)
            st.line_chart(df_forecast.set_index("Date")[["Forecast_kWh"]])
        except Exception as e:
            st.error(f"Errore durante la previsione: {e}")
