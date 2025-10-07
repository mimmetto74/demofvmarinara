
# pv_forecast_all_in_one.py
# Streamlit app: produzione FV - storico, previsioni e confronto
# V7 - pronto per Railway/GitHub
import io
import os
import json
import math
import time
import base64
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None

# ---------------- Meteomatics: credenziali + fetch ----------------
# (richieste dall'utente per esecuzione diretta su Railway/GitHub)
METEO_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
METEO_PASS = "6S8KTHPbrUlp6523T9Xd"

SHOW_DEBUG_URL = False  # lascia False: non mostriamo la URL in UI

def _orient_cardinal(deg: int) -> str:
    m = {0: "N", 90: "E", 180: "S", 270: "W"}
    if deg in m:
        return m[deg]
    nearest = min(m.keys(), key=lambda k: abs(k - deg))
    return m[nearest]

def get_meteomatics_json(lat: float, lon: float, start_iso: str, end_iso: str,
                         tilt_deg: int, orient_deg: int) -> dict:
    orient = _orient_cardinal(int(orient_deg))
    param = f"global_rad_tilt_{int(tilt_deg)}_orientation_{orient}:W,total_cloud_cover:p"
    url = (
        f"https://api.meteomatics.com/"
        f"{start_iso}--{end_iso}:PT15M/{param}/{lat:.6f},{lon:.6f}/json"
    )
    if SHOW_DEBUG_URL:
        print("DEBUG MM URL:", url)
    r = requests.get(url, auth=(METEO_USER, METEO_PASS), timeout=25)
    r.raise_for_status()
    return r.json()

def meteomatics_to_df(payload: dict) -> pd.DataFrame:
    if "data" not in payload or not payload["data"]:
        return pd.DataFrame(columns=["ts", "global_rad_Wm2", "tcc_pct"])
    data_by_param = {entry["parameter"]: entry["coordinates"][0]["dates"] for entry in payload["data"]}
    ts = [pd.to_datetime(d["date"]) for d in next(iter(data_by_param.values()))]
    df = pd.DataFrame({"ts": ts})
    # global rad
    k_glob = next((k for k in data_by_param if k.startswith("global_rad")), None)
    if k_glob:
        df["global_rad_Wm2"] = [d["value"] for d in data_by_param[k_glob]]
    else:
        df["global_rad_Wm2"] = np.nan
    # cloud cover
    if "total_cloud_cover" in data_by_param:
        df["tcc_pct"] = [d["value"] for d in data_by_param["total_cloud_cover"]]
    else:
        df["tcc_pct"] = np.nan
    return df

# ---------------- Open-Meteo fallback (gratuito) ------------------
def get_openmeteo_df(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    # Nota: API pubblica; qui usiamo la stima di radiazione solare globale (shortwave_radiation)
    # Aggregazione oraria; ricampioniamo a 15 minuti con step interposto.
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=shortwave_radiation,cloud_cover&timezone=UTC"
        f"&start_date={start.date()}&end_date={end.date()}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    js = r.json()
    if "hourly" not in js:
        return pd.DataFrame(columns=["ts", "global_rad_Wm2", "tcc_pct"])
    t = pd.to_datetime(js["hourly"]["time"])
    rad = js["hourly"].get("shortwave_radiation", [np.nan]*len(t))
    cc = js["hourly"].get("cloud_cover", [np.nan]*len(t))
    df = pd.DataFrame({"ts": t, "global_rad_Wm2": rad, "tcc_pct": cc})
    # Ricampiona a 15 min (interpolazione)
    df = df.set_index("ts").resample("15T").interpolate().reset_index()
    return df

# ---------------- Utility produzione stimata ----------------------
def radiation_to_power_w(df: pd.DataFrame, kwp: float, system_eff: float = 0.85) -> pd.DataFrame:
    """Converte la radiazione globale (W/m2) in potenza FV stimata (kW).
    Modello semplice: P = G * area_eff * rendimento; area_eff ~ kwp / 0.2 (ipotesi 200 W/m2).
    """
    if df.empty:
        return df.assign(p_kW=np.nan, e_kWh=np.nan)
    area_eff = kwp / 0.2  # m2 equivalenti (ipotesi moduli 20%)
    p_w = df["global_rad_Wm2"].clip(lower=0) * area_eff * system_eff
    p_kw = p_w / 1000.0
    df_out = df.copy()
    df_out["p_kW"] = p_kw
    # Energia su intervallo (15 min): kWh = kW * 0.25
    df_out["e_kWh"] = df_out["p_kW"] * 0.25
    return df_out

def daily_agg(df15: pd.DataFrame, ts_col="ts", e_col="e_kWh") -> pd.DataFrame:
    if df15.empty:
        return pd.DataFrame(columns=["date", "energy_kWh"])
    d = df15.copy()
    d["date"] = pd.to_datetime(d[ts_col]).dt.date
    g = d.groupby("date")[e_col].sum().reset_index()
    g.rename(columns={e_col: "energy_kWh"}, inplace=True)
    return g

# ---------------- UI Helpers -------------------------------------
def page_header():
    st.set_page_config(page_title="Robotronix Solar V7", layout="wide")
    st.title("Robotronix Solar • V7")
    st.caption("Storico, Previsioni e Confronto • Meteomatics + Open‑Meteo (fallback)")

def sidebar_controls():
    st.sidebar.header("Impostazioni")
    lat = st.sidebar.number_input("Latitudine", value=40.836, format="%.6f")
    lon = st.sidebar.number_input("Longitudine", value=14.305, format="%.6f")
    tilt = st.sidebar.slider("Tilt (°)", 0, 60, 25, step=5)
    orient = st.sidebar.slider("Orientamento (° da Nord, senso orario)", 0, 359, 180, step=15)
    kwp = st.sidebar.number_input("Potenza di picco (kWp)", value=50.0, step=1.0)
    eff = st.sidebar.slider("Rendimento di sistema", 0.6, 0.98, 0.85, step=0.01)
    days_back = st.sidebar.slider("Giorni storici da scaricare", 1, 14, 7)
    return lat, lon, tilt, orient, kwp, eff, days_back

def load_csv(label, default_path):
    st.write(f"**{label}**")
    up = st.file_uploader(f"Carica CSV {label}", type=["csv"], key=label)
    if up is not None:
        df = pd.read_csv(up)
        st.success("CSV caricato ✅")
        return df
    if os.path.exists(default_path):
        try:
            df = pd.read_csv(default_path)
            st.info(f"Caricato file di default: `{default_path}`")
            return df
        except Exception as e:
            st.warning(f"Impossibile leggere `{default_path}`: {e}")
    st.warning("Nessun CSV fornito. Uso un esempio minimale.")
    return pd.DataFrame({"Date": [str(datetime.utcnow().date())], "E_INT_Daily_kWh": [0.0]})

def normalize_real_df(df: pd.DataFrame) -> pd.DataFrame:
    # accetta due schemi: (1) Date + E_INT_Daily_kWh ; (2) ts + value ; (3) arbitrary con inferenza
    cols = [c.lower() for c in df.columns]
    df = df.copy()
    df.columns = cols
    if "date" in df.columns and "e_int_daily_kwh" in df.columns:
        out = df[["date", "e_int_daily_kwh"]].copy()
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out.rename(columns={"e_int_daily_kwh": "energy_kWh"}, inplace=True)
        return out
    # tentativo generico
    for a in df.columns:
        if "date" in a or "data" in a:
            date_col = a
            break
    else:
        date_col = df.columns[0]
    # cerca col energia
    energy_col = None
    for a in df.columns:
        if "kwh" in a:
            energy_col = a
            break
    if energy_col is None:
        # fallback su seconda colonna numerica
        num_cols = df.select_dtypes(include=[float, int]).columns.tolist()
        energy_col = num_cols[0] if num_cols else df.columns[1] if len(df.columns)>1 else df.columns[0]
    out = df[[date_col, energy_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.date
    out.columns = ["date", "energy_kWh"]
    return out

def fetch_forecast(lat, lon, tilt, orient, kwp, eff, days_back):
    tz = timezone.utc
    end = datetime.now(tz).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days_back)
    start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    # tenta Meteomatics
    try:
        mm = get_meteomatics_json(lat, lon, start_iso, end_iso, tilt, orient)
        st.toast("Dati Meteo ricevuti ✅ (Meteomatics)", icon="✅")
        d15 = meteomatics_to_df(mm)
    except Exception as e:
        st.toast("Fallback Open‑Meteo", icon="⚠️")
        d15 = get_openmeteo_df(lat, lon, start, end)

    d15 = radiation_to_power_w(d15, kwp, system_eff=eff)
    daily = daily_agg(d15)
    return d15, daily

def mae(a, f):
    a = np.array(a, dtype=float)
    f = np.array(f, dtype=float)
    m = np.nanmean(np.abs(a - f))
    return float(m)

def mape(a, f):
    a = np.array(a, dtype=float)
    f = np.array(f, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pe = np.abs((a - f) / np.where(a==0, np.nan, a)) * 100.0
    return float(np.nanmean(pe))

# ---------------- Pagine ------------------------------------------
def page_storico(df_real):
    st.subheader("Andamento Storico")
    if df_real.empty:
        st.info("Carica un CSV con produzione reale.")
        return
    df_r = normalize_real_df(df_real)
    # Filtri
    years = sorted({pd.to_datetime(x).year for x in pd.to_datetime(df_r["date"])})
    year = st.multiselect("Filtra per anno", years, default=years)
    df_f = df_r[df_r["date"].apply(lambda d: pd.to_datetime(d).year in set(year))]
    c1, c2 = st.columns([2,1])
    with c1:
        st.line_chart(df_f.set_index("date")["energy_kWh"])
    with c2:
        st.metric("Totale (kWh)", f"{df_f['energy_kWh'].sum():,.0f}")
        st.metric("Media giornaliera (kWh)", f"{df_f['energy_kWh'].mean():,.1f}")

def page_previsioni(lat, lon, tilt, orient, kwp, eff, days_back):
    st.subheader("Previsioni Meteo → Produzione FV stimata")
    d15, daily = fetch_forecast(lat, lon, tilt, orient, kwp, eff, days_back)
    st.write("**Produzione stimata giornaliera (kWh)**")
    st.bar_chart(daily.set_index("date")["energy_kWh"])
    st.download_button("Scarica CSV giornaliero", data=daily.to_csv(index=False), file_name="forecast_daily.csv")
    st.write("**Serie a 15 minuti (kW)**")
    st.line_chart(d15.set_index("ts")["p_kW"])
    st.download_button("Scarica CSV 15 min", data=d15.to_csv(index=False), file_name="forecast_15min.csv")

def page_storico_vs_previsione(df_real, lat, lon, tilt, orient, kwp, eff, days_back):
    st.subheader("Storico vs Previsione")
    df_real_n = normalize_real_df(df_real) if not df_real.empty else pd.DataFrame(columns=["date","energy_kWh"])
    d15, daily = fetch_forecast(lat, lon, tilt, orient, kwp, eff, days_back)
    # allinea per data
    m = pd.merge(df_real_n, daily, on="date", how="inner", suffixes=("_real", "_pred"))
    if m.empty:
        st.warning("Nessuna sovrapposizione di date tra storico e previsione.")
        return
    c1, c2, c3 = st.columns(3)
    mae_val = mae(m["energy_kWh_real"], m["energy_kWh_pred"])
    mape_val = mape(m["energy_kWh_real"], m["energy_kWh_pred"])
    bias = (m["energy_kWh_pred"].mean() - m["energy_kWh_real"].mean()) / (m["energy_kWh_real"].mean()+1e-9) * 100
    c1.metric("MAE (kWh)", f"{mae_val:,.1f}")
    c2.metric("Scostamento medio (%)", f"{bias:+.1f}%")
    c3.metric("MAPE (%)", f"{mape_val:.1f}%")

    st.write("**Confronto giornaliero (kWh)**")
    plot_df = m[["date", "energy_kWh_real", "energy_kWh_pred"]].set_index("date")
    st.line_chart(plot_df)

    st.dataframe(m.rename(columns={
        "energy_kWh_real": "Reale (kWh)",
        "energy_kWh_pred": "Stimato (kWh)"
    }))

def page_mappa(lat, lon):
    st.subheader("Mappa impianto")
    if folium is None:
        st.info("Installare folium e streamlit-folium per la mappa.")
        return
    m = folium.Map(location=[lat, lon], zoom_start=13, control_scale=True)
    folium.Marker([lat, lon], tooltip="Impianto FV").add_to(m)
    st_folium(m, width=None, height=500)

def main():
    page_header()
    lat, lon, tilt, orient, kwp, eff, days_back = sidebar_controls()

    # Menu pagine (centralizzato)
    page = st.sidebar.radio(
        "Pagina",
        ("storico", "previsioni", "storico_vs_previsione", "mappa"),
        captions=[
            "Grafici storici da CSV",
            "Previsioni Meteo → Produzione",
            "Confronto reale vs stimato",
            "Posizione impianto"
        ],
        index=2
    )

    # Caricamento CSV storici (anche multipli)
    with st.expander("Dati storici (carica uno o più CSV)"):
        df_main = load_csv("Dataset_Daily_EnergiaSeparata_2020_2025.csv", "Dataset_Daily_EnergiaSeparata_2020_2025.csv")
        df_alt = load_csv("Marinara.csv", "Marinara.csv")
        if not df_main.empty and not df_alt.empty:
            # se hanno stessa struttura, unisci evitando duplicati
            try:
                df_hist = pd.concat([df_main, df_alt], ignore_index=True).drop_duplicates()
            except Exception:
                df_hist = df_main
        else:
            df_hist = df_main if not df_main.empty else df_alt

    # Routing pagine
    if page == "storico":
        page_storico(df_hist)
    elif page == "previsioni":
        page_previsioni(lat, lon, tilt, orient, kwp, eff, days_back)
    elif page == "storico_vs_previsione":
        page_storico_vs_previsione(df_hist, lat, lon, tilt, orient, kwp, eff, days_back)
    elif page == "mappa":
        page_mappa(lat, lon)

if __name__ == "__main__":
    main()
