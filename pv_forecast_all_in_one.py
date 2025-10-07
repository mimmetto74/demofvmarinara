# -*- coding: utf-8 -*-
# ROBOTRONIX - Solar Forecast (V7)
# UI stile V4 con migliorie: Meteomatics "silenzioso", PT15M tilt/orient, CSV logging, mappa folium
# Note: credenziali Meteomatics inserite su richiesta (puoi mettere ENV in produzione).

import os
import io
import json
import math
import time
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_folium import st_folium
import folium

# -----------------------------
# CONFIG & COSTANTI
# -----------------------------

APP_TITLE = "Solar Forecast - ROBOTRONIX for IMEPOWER"
LOCAL_TZ = "Europe/Rome"  # adattabile

# >>>> CREDENZIALI METEOMATICS (su richiesta dell’utente)
METEO_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
METEO_PASS = "6S8KTHPbrUlp6523T9Xd"

# In alternativa in produzione:
# METEO_USER = os.getenv("METEO_USER", "")
# METEO_PASS = os.getenv("METEO_PASS", "")

DATA_DIR = "logs"
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_LAT = 40.643278
DEFAULT_LON = 16.986083
DEFAULT_TILT = 20          # gradi
DEFAULT_AZ = 180           # 0=N, 90=E, 180=S, 270=W
DEFAULT_POWER_KW = 947.32  # potenza di targa in kW
DEFAULT_PR = 0.85          # performance ratio

PROVIDERS = ["Meteomatics", "Open-Meteo"]
STEPS_ISO = "PT15M"  # 15 minuti

# -----------------------------
# UTILS
# -----------------------------

def utcnow():
    return datetime.now(timezone.utc)

def to_iso_z(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def daterange_iso(day, days=1):
    """range ISO [day 00:00Z, day+days 00:00Z)"""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=days)
    return to_iso_z(start), to_iso_z(end)

def to_local(df, col="time", tz=LOCAL_TZ):
    return df.assign(**{
        col: pd.to_datetime(df[col], utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    })

def safe_round(x, n=1):
    try:
        return round(float(x), n)
    except Exception:
        return x

def ensure_cols(df, names):
    for n in names:
        if n not in df.columns:
            df[n] = np.nan
    return df

# -----------------------------
# FETCH PROVIDERS
# -----------------------------

def fetch_meteomatics(lat, lon, tilt, az, day_utc, step=STEPS_ISO):
    """
    Ritorna df con colonne:
      time (TZ local), gti_wm2, cloud_p
    Query: global_rad_tilt_{tilt}_orientation_{az}:W, total_cloud_cover:p
    """
    start_iso, end_iso = daterange_iso(day_utc)
    param = f"global_rad_tilt_{int(tilt)}_orientation_{int(az)}:W,total_cloud_cover:p"
    url = f"https://api.meteomatics.com/{start_iso}--{end_iso}:{step}/{param}/{lat},{lon}/json"

    resp = requests.get(url, auth=(METEO_USER, METEO_PASS), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # struttura: data["data"] è lista di parametri
    # each param -> {"parameter":"global_rad_tilt_...:W", "coordinates":[{"dates":[{"date":"...","value":...}, ...]}]}
    recs = {}
    for par in data.get("data", []):
        pname = par.get("parameter", "")
        # standardizzo
        if pname.startswith("global_rad_tilt_"):
            key = "gti_wm2"
        elif pname.startswith("total_cloud_cover"):
            key = "cloud_p"
        else:
            continue
        dates = par["coordinates"][0]["dates"]
        recs[key] = pd.DataFrame(dates).rename(columns={"date": "time", "value": key})

    df = None
    for k, d in recs.items():
        df = d if df is None else df.merge(d, on="time", how="outer")
    if df is None:
        # vuoto
        df = pd.DataFrame({"time": [], "gti_wm2": [], "cloud_p": []})

    df = to_local(df, "time")
    df = df.sort_values("time").reset_index(drop=True)
    df = ensure_cols(df, ["gti_wm2", "cloud_p"])
    return df

def fetch_open_meteo(lat, lon, day_utc):
    """
    Fallback con Open-Meteo: direct_radiation (W/m2), cloudcover (%)
    Simulazione GTI = direct * 0.9 (approssimazione), poi resampling a 15 min.
    """
    start = day_utc.strftime("%Y-%m-%d")
    end = (day_utc + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=direct_radiation,cloudcover"
        f"&start_date={start}&end_date={end}&timezone=UTC"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()

    hours = js.get("hourly", {}).get("time", [])
    direct = js.get("hourly", {}).get("direct_radiation", [])
    cloud = js.get("hourly", {}).get("cloudcover", [])

    if not hours:
        return pd.DataFrame({"time": [], "gti_wm2": [], "cloud_p": []})

    dfh = pd.DataFrame({"time": pd.to_datetime(hours, utc=True),
                        "direct": direct,
                        "cloud_p": cloud})
    # “simulo” GTI (global sul piano) in modo conservativo
    dfh["gti_wm2"] = dfh["direct"] * 0.90
    # porta a locale e resample a 15 min
    dfh = dfh.set_index("time").tz_convert(LOCAL_TZ)
    dfr = dfh[["gti_wm2", "cloud_p"]].resample("15min").interpolate().reset_index()
    dfr["time"] = dfr["time"].dt.tz_localize(None)
    return dfr

# -----------------------------
# MODELLO FV (semplice ma coerente)
# -----------------------------

def pv_curve(df, power_kw=DEFAULT_POWER_KW, pr=DEFAULT_PR):
    """PVout_kW = Pdc_rated * PR * (GTI/1000)"""
    df = df.copy()
    df["pv_kw"] = power_kw * pr * (df["gti_wm2"] / 1000.0)
    df["pv_kw"] = df["pv_kw"].clip(lower=0)
    # energia su 15 min:
    df["kwh_15min"] = df["pv_kw"] * 0.25
    return df

def day_metrics(df):
    if df.empty:
        return dict(energy_kwh=0, peak_kw=0, pct=0, cloud_avg=np.nan)
    energy = df["kwh_15min"].sum()
    peak = df["pv_kw"].max()
    return dict(
        energy_kwh=energy,
        peak_kw=peak,
        cloud_avg=float(np.nanmean(df["cloud_p"])) if "cloud_p" in df.columns else np.nan
    )

# -----------------------------
# LOGGING
# -----------------------------

def save_logs(day_iso, df_curve, agg_row, prefix="meteo"):
    """Salva: curve_YYYY-MM-DD.csv, agg_YYYY-MM-DD.csv"""
    # curva
    p_curve = os.path.join(DATA_DIR, f"{prefix}_curve_{day_iso}.csv")
    df_curve.to_csv(p_curve, index=False)
    # aggregato (append)
    p_agg = os.path.join(DATA_DIR, f"{prefix}_agg.csv")
    header = not os.path.exists(p_agg)
    pd.DataFrame([agg_row]).to_csv(p_agg, mode="a", index=False, header=header)

# -----------------------------
# CHART
# -----------------------------

def chart_curve(df, title):
    dfp = df[["time", "pv_kw"]].rename(columns={"pv_kw": "kW"})
    base = alt.Chart(dfp).mark_line().encode(
        x=alt.X("time:T", title="Ora"),
        y=alt.Y("kW:Q", title="Potenza (kW)")
    ).properties(height=260, title=title)
    return base.interactive(bind_y=False)

# -----------------------------
# UI
# -----------------------------

st.set_page_config(APP_TITLE, layout="wide", page_icon="🔆")

with st.sidebar:
    st.header("Impostazioni")
    provider = st.selectbox("Fonte meteo:", PROVIDERS, index=0)
    power_kw = st.number_input("Potenza di targa impianto (kW)", min_value=1.0, step=1.0, value=float(DEFAULT_POWER_KW))
    auto_save = st.toggle("Salvataggio automatico CSV (curva + aggregato)", value=True)
    st.markdown("---")
    st.subheader("📍 Posizione & Piano")
    lat = st.number_input("Latitudine", value=float(DEFAULT_LAT), step=0.0001, format="%.6f")
    lon = st.number_input("Longitudine", value=float(DEFAULT_LON), step=0.0001, format="%.6f")
    tilt = st.slider("Tilt (°)", 0, 60, DEFAULT_TILT)
    az = st.slider("Orientazione (°, 0=N, 90=E, 180=S, 270=W)", 0, 360, DEFAULT_AZ, step=1)

st.title(APP_TITLE)

tabs = st.tabs(["🗃 Storico", "🛠 Modello", "📈 Previsioni 4 giorni (15m)", "🗺 Mappa"])

# -----------------------------
# TAB: PREVISIONI
# -----------------------------

with tabs[2]:
    st.subheader("Previsioni (PT15M, tilt/orient, provider toggle)")

    btn = st.button("Calcola previsioni (Ieri/Oggi/Domani/Dopodomani)")
    if btn:
        st.session_state["run_forecast"] = True

    # date base (UTC)
    today_utc = utcnow().date()

    days = [
        ("Ieri", today_utc - timedelta(days=1)),
        ("Oggi", today_utc),
        ("Domani", today_utc + timedelta(days=1)),
        ("Dopodomani", today_utc + timedelta(days=2)),
    ]

    if st.session_state.get("run_forecast"):
        for label, d in days:
            col_box = st.container()
            with col_box:
                st.markdown(f"### {label} – {d.isoformat()}")

                # tenta Meteomatics ma non mostra URL
                used_provider = provider
                try:
                    if provider == "Meteomatics" and METEO_USER and METEO_PASS:
                        df = fetch_meteomatics(lat, lon, tilt, az, datetime(d.year, d.month, d.day, tzinfo=timezone.utc))
                        used_provider = "Meteomatics"
                    else:
                        raise RuntimeError("Meteomatics non disponibile, uso Open-Meteo")
                except Exception:
                    df = fetch_open_meteo(lat, lon, datetime(d.year, d.month, d.day, tzinfo=timezone.utc))
                    used_provider = "Open-Meteo"

                df = pv_curve(df, power_kw=power_kw, pr=DEFAULT_PR)
                m = day_metrics(df)
                pct = (m["peak_kw"] / power_kw * 100.0) if power_kw > 0 else 0.0

                met_left, met_mid, met_right, met_cloud = st.columns([1,1,1,1])
                met_left.metric("Energia stimata giorno", f"{safe_round(m['energy_kwh'],1):,} kWh".replace(",", " "))
                met_mid.metric("Picco stimato", f"{safe_round(m['peak_kw'],1):,} kW".replace(",", " "))
                met_right.metric("% della targa", f"{safe_round(pct,1)}%")
                met_cloud.metric("Nuvolosità media", f"{safe_round(m['cloud_avg'],0)}%")

                st.altair_chart(chart_curve(df, f"Curva 15-min ({used_provider})"), use_container_width=True)

                # salvataggi
                day_iso = d.isoformat()
                if auto_save:
                    agg_row = {
                        "day": day_iso,
                        "provider": used_provider,
                        "lat": lat, "lon": lon,
                        "tilt": tilt, "az": az,
                        "energy_kwh": m["energy_kwh"],
                        "peak_kw": m["peak_kw"],
                        "pct_of_nameplate": pct,
                        "cloud_avg": m["cloud_avg"]
                    }
                    prefix = "meteo" if used_provider == "Meteomatics" else "openmeteo"
                    save_logs(day_iso, df[["time", "gti_wm2", "cloud_p", "pv_kw", "kwh_15min"]], agg_row, prefix=prefix)

                # download curva locale
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Scarica curva 15-min (CSV)",
                    data=csv,
                    file_name=f"curve_15m_{label.lower()}_{day_iso}.csv",
                    mime="text/csv"
                )

# -----------------------------
# TAB: STORICO (placeholder)
# -----------------------------
with tabs[0]:
    st.info("Storico: carica un CSV di produzione reale per confronti e validazione (facoltativo).")
    up = st.file_uploader("Carica produzione reale giornaliera (CSV con colonne: date, kWh)", type=["csv"])
    if up:
        dfr = pd.read_csv(up)
        st.dataframe(dfr.head())

# -----------------------------
# TAB: MODELLO (placeholder semplice)
# -----------------------------
with tabs[1]:
    st.info("Modello: stima semplice PV = P_nom * PR * (GTI/1000). In una fase successiva si può allenare un regressore con dati storici.")

# -----------------------------
# TAB: MAPPA
# -----------------------------
with tabs[3]:
    st.subheader("Mappa impianto (satellitare)")

    m = folium.Map(location=[lat, lon], zoom_start=16, tiles="Esri.WorldImagery")
    folium.Marker([lat, lon], icon=folium.Icon(color="green", icon="sun")).add_to(m)
    # box descrittivo separato (non popup)
    st.markdown(
        """
        <div style="padding:8px 12px; background:#111; border:1px solid #333; border-radius:12px; width: 420px; margin-bottom:10px;">
        <b>Impianto:</b> ROBOTRONIX – Lat/Lon: {:.6f}, {:.6f} – Tilt: {}° – Orientazione: {}°
        </div>
        """.format(lat, lon, tilt, az),
        unsafe_allow_html=True
    )
    st_folium(m, height=520, width=None)

