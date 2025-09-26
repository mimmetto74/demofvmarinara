
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import date, timedelta, datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ===============================
# Autenticazione semplice
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
# Config dataset e modello
# ===============================
CLOUD_DATA = "Dataset_Daily_EnergiaSeparata_2020_2025.csv"
CLOUD_DATA_GZ = "Dataset_Daily_EnergiaSeparata_2020_2025.csv.gz"
CLOUD_MODEL = "pv_model.joblib"

def pick_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

DATA_PATH = pick_existing_path([CLOUD_DATA_GZ, CLOUD_DATA])
MODEL_PATH = pick_existing_path([CLOUD_MODEL]) or CLOUD_MODEL

# Coordinate default impianto Marinara (Taranto)
DEFAULT_LAT = 40.6432780
DEFAULT_LON = 16.9860830

# ===============================
# Config Meteomatics (parametri corretti)
# ===============================
MM_USER = "xensoramasrl_pignatelli_cosimo"
MM_PASS = "m7jfJRWcqN09lr88YWYs"
BASE_URL = "https://api.meteomatics.com"

PARAMS = [
    "solar_rad:mean:W",   # Irraggiamento solare medio
    "cloud_cover:tot:p",  # Copertura nuvolosa totale (%)
    "t_2m:C",             # Temperatura a 2m
    "wind_speed_10m:ms"   # Vento a 10m
]

def get_meteomatics_forecast(lat, lon, days_ahead=1):
    start = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%dT00:00:00Z")
    end   = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%dT23:59:00Z")
    url = f"{BASE_URL}/{start}--{end}:PT1H/{','.join(PARAMS)}/{lat},{lon}/json"

    r = requests.get(url, auth=(MM_USER, MM_PASS))
    r.raise_for_status()
    data = r.json()["data"]

    df = pd.DataFrame()
    for var in data:
        name = var["parameter"]
        df[name] = [float(e["value"]) for e in var["coordinates"][0]["dates"]]
    df["time"] = [pd.to_datetime(e["date"]) for e in data[0]["coordinates"][0]["dates"]]
    df = df.set_index("time")

    return df

# ===============================
# Funzioni modello
# ===============================
def load_dataset():
    if DATA_PATH is None:
        st.error("⚠️ Dataset non trovato. Carica il file nella repo.")
        return None
    try:
        return pd.read_csv(DATA_PATH, parse_dates=["Date"])
    except Exception as e:
        st.error(f"Errore nel caricamento del dataset: {e}")
        return None

def train_model(df):
    # Training solo su G_M0_Wm2 (lo storico non ha altre variabili meteo)
    df = df.dropna(subset=["E_INT_Daily_kWh", "G_M0_Wm2"])
    train = df[df["Date"] < "2025-01-01"]
    test  = df[df["Date"] >= "2025-01-01"]

    X_train = train[["G_M0_Wm2"]]
    y_train = train["E_INT_Daily_kWh"]
    X_test  = test[["G_M0_Wm2"]]
    y_test  = test["E_INT_Daily_kWh"]

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

def forecast_day_ahead(features: pd.DataFrame) -> float:
    model = load_model()
    if model is None:
        return None
    return float(model.predict(features)[0])

# ===============================
# UI Streamlit
# ===============================
st.set_page_config(page_title="PV Forecast Dashboard", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: orange;'>☀️ Solar Forecast - ROBOTRONIX for IMEPOWER</h1>",
    unsafe_allow_html=True
)
st.write("---")

st.sidebar.title("☀️ Menu")
st.sidebar.markdown("Seleziona le opzioni:")

# --- Analisi Storica ---
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

# --- Addestramento ---
st.header("🛠️ Addestramento modello")
if st.button("Addestra modello con dati storici"):
    if df is not None:
        mae, r2 = train_model(df)
        st.success(f"Modello salvato in: {MODEL_PATH}")
        st.write(f"Prestazioni su test 2025 → MAE: {mae:.2f} kWh • R²: {r2:.3f}")
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                st.download_button("⬇️ Scarica modello allenato", f, file_name="pv_model.joblib")

# --- Previsione ---
st.header("🔮 Previsione FV")
lat = st.number_input("Latitudine", value=DEFAULT_LAT, format="%.6f")
lon = st.number_input("Longitudine", value=DEFAULT_LON, format="%.6f")
giorno = st.selectbox("Giorno di previsione", ["Domani", "Dopodomani"])
days_ahead = 1 if giorno == "Domani" else 2

if st.button("Calcola previsione con Meteomatics"):
    df_forecast = get_meteomatics_forecast(lat, lon, days_ahead=days_ahead)
    features_mean = pd.DataFrame({
        "G_M0_Wm2": [df_forecast["solar_rad:mean:W"].mean()]
    })
    prod_forecast = forecast_day_ahead(features_mean)
    if prod_forecast is not None:
        st.subheader(f"📅 Risultati ({giorno})")
        st.metric("Irraggiamento medio previsto", f"{features_mean['G_M0_Wm2'][0]:.1f} W/m²")
        st.metric("Copertura nuvolosa media", f"{df_forecast['cloud_cover:tot:p'].mean():.1f} %")
        st.metric("Temperatura media", f"{df_forecast['t_2m:C'].mean():.1f} °C")
        st.metric("Vento medio", f"{df_forecast['wind_speed_10m:ms'].mean():.1f} m/s")
        st.metric("Produzione stimata", f"{prod_forecast:.1f} kWh")
