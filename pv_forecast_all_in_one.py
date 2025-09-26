import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ===============================
# Config
# ===============================
PARAMS = [
    "global_rad:W",      # Irraggiamento solare globale
    "cloud_cover:tot:p", # Copertura nuvolosa totale (%)
    "t_2m:C",            # Temperatura a 2m
    "wind_speed_10m:ms"  # Vento a 10m
]

st.title("☀️ Solar Forecast - TESEO-RX for IMEPOWER")

st.write("Versione con parametri Meteomatics corretti (global_rad, cloud_cover, t_2m, wind_speed_10m).")
