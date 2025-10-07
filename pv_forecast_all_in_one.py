
# pv_forecast_all_in_one.py
# Robotronix Solar V8.1 - Meteomatics fixed + Open-Meteo fallback + previsioni a DOMANI
import os
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

# ---------------- Credenziali Meteomatics ----------------
METEO_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
METEO_PASS = "6S8KTHPbrUlp6523T9Xd"
SHOW_DEBUG_URL = False

# ---------------- Funzioni Meteo ----------------
def _orient_cardinal(deg: int) -> str:
    if 45 <= deg < 135:
        return "east"
    elif 135 <= deg < 225:
        return "south"
    elif 225 <= deg < 315:
        return "west"
    else:
        return "north"

def get_meteomatics_json(lat: float, lon: float, start_iso: str, end_iso: str,
                         tilt_deg: int, orient_deg: int) -> dict:
    orient = _orient_cardinal(int(orient_deg))
    param = f"global_rad_tilt_{int(tilt_deg)}_orientation_{orient}:W,total_cloud_cover:p"
    url = f"https://api.meteomatics.com/{start_iso}--{end_iso}:PT15M/{param}/{lat:.6f},{lon:.6f}/json"
    if SHOW_DEBUG_URL:
        st.write("DEBUG URL:", url)
    r = requests.get(url, auth=(METEO_USER, METEO_PASS), timeout=25)
    r.raise_for_status()
    return r.json()

def get_openmeteo_df(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
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
    df = df.set_index("ts").resample("15T").interpolate().reset_index()
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("UTC").dt.tz_convert("Europe/Rome")
    return df

def meteomatics_to_df(payload: dict) -> pd.DataFrame:
    if "data" not in payload or not payload["data"]:
        return pd.DataFrame(columns=["ts", "global_rad_Wm2", "tcc_pct"])
    data_by_param = {entry["parameter"]: entry["coordinates"][0]["dates"] for entry in payload["data"]}
    ts = [pd.to_datetime(d["date"]) for d in next(iter(data_by_param.values()))]
    df = pd.DataFrame({"ts": ts})
    k_glob = next((k for k in data_by_param if k.startswith("global_rad")), None)
    df["global_rad_Wm2"] = [d["value"] for d in data_by_param.get(k_glob, [])] if k_glob else np.nan
    if "total_cloud_cover" in data_by_param:
        df["tcc_pct"] = [d["value"] for d in data_by_param["total_cloud_cover"]]
    else:
        df["tcc_pct"] = np.nan
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("UTC").dt.tz_convert("Europe/Rome")
    return df

# ---------------- Conversione energia ----------------
def radiation_to_power(df: pd.DataFrame, kwp: float, system_eff: float = 0.85) -> pd.DataFrame:
    if df.empty:
        return df.assign(p_kW=np.nan, e_kWh=np.nan)
    area_eff = kwp / 0.2
    p_kw = df["global_rad_Wm2"].clip(lower=0) * area_eff * system_eff / 1000.0
    out = df.copy()
    out["p_kW"] = p_kw
    out["e_kWh"] = out["p_kW"] * 0.25
    return out

def daily_agg(df15: pd.DataFrame) -> pd.DataFrame:
    if df15.empty:
        return pd.DataFrame(columns=["date","energy_kWh"])
    d = df15.copy()
    d["date"] = d["ts"].dt.tz_convert("Europe/Rome").dt.date
    g = d.groupby("date")["e_kWh"].sum().reset_index().rename(columns={"e_kWh":"energy_kWh"})
    return g

# ---------------- Training modello ----------------
def train_model_with_history(df_hist_daily: pd.DataFrame, lat, lon, tilt, orient, kwp, eff):
    if df_hist_daily is None or df_hist_daily.empty:
        return {"alpha":1.0,"beta":0.0,"n_days":0,"rmse":np.nan}
    df = df_hist_daily.copy()
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        date_col = next((c for c in df.columns if "date" in c or "data" in c), df.columns[0])
        df["date"] = pd.to_datetime(df[date_col]).dt.date
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    if "energy_kwh" not in df.columns:
        kwh_col = next((c for c in df.columns if "kwh" in c), None)
        if kwh_col is None:
            num = df.select_dtypes(include=[float,int]).columns.tolist()
            kwh_col = num[0] if num else df.columns[1]
        df["energy_kWh"] = df[kwh_col].astype(float)
    else:
        df["energy_kWh"] = df["energy_kwh"].astype(float)
    df = df[["date","energy_kWh"]].dropna()

    # Limita al periodo recente (ultimi 10 giorni)
    end = pd.to_datetime(max(df["date"]))
    start = end - pd.Timedelta(days=10)
    start_iso = start.strftime("%Y-%m-%dT00:00:00Z")
    end_iso = end.strftime("%Y-%m-%dT23:45:00Z")
    try:
        js = get_meteomatics_json(lat, lon, start_iso, end_iso, tilt, orient)
        d15 = meteomatics_to_df(js)
        d15 = radiation_to_power(d15, kwp, eff)
        pred_daily = daily_agg(d15)
    except Exception as e:
        st.warning(f"⚠️ Meteomatics non disponibile: uso Open-Meteo ({e})")
        d15 = get_openmeteo_df(lat, lon, start, end)
        d15 = radiation_to_power(d15, kwp, eff)
        pred_daily = daily_agg(d15)

    m = pd.merge(df, pred_daily, on="date", how="inner", suffixes=("_real","_pred"))
    if m.empty:
        return {"alpha":1.0,"beta":0.0,"n_days":0,"rmse":np.nan}
    y = m["energy_kWh_real"].to_numpy(float)
    x = m["energy_kWh_pred"].to_numpy(float)
    if np.var(x) < 1e-9:
        alpha = 1.0; beta = 0.0
    else:
        alpha = float(np.cov(x, y, bias=True)[0,1] / np.var(x))
        beta  = float(np.mean(y) - alpha*np.mean(x))
    rmse = float(np.sqrt(np.mean((y - (alpha*x + beta))**2)))
    return {"alpha":alpha, "beta":beta, "n_days":int(len(m)), "rmse":rmse}

# ---------------- UI ----------------
def page_header():
    st.set_page_config(page_title="Robotronix Solar V8.1", layout="wide")
    st.title("Robotronix Solar • V8.1")
    st.caption("Meteomatics FIX + fallback Open-Meteo • Previsioni DOMANI")

def sidebar_controls():
    st.sidebar.header("Impostazioni impianto")
    lat = st.sidebar.number_input("Latitudine", value=40.836, format="%.6f")
    lon = st.sidebar.number_input("Longitudine", value=14.305, format="%.6f")
    tilt = st.sidebar.slider("Tilt (°)", 0, 60, 25, step=5)
    orient = st.sidebar.slider("Orientamento (° da Nord, senso orario)", 0, 359, 180, step=15)
    kwp = st.sidebar.number_input("Potenza di picco (kWp)", value=50.0, step=1.0)
    eff = st.sidebar.slider("Rendimento di sistema", 0.6, 0.98, 0.85, step=0.01)
    return lat, lon, tilt, orient, kwp, eff

def page_modello(lat, lon, tilt, orient, kwp, eff):
    st.subheader("Modello • Training con dati storici")
    up = st.file_uploader("Carica CSV storico giornaliero (Date, E_INT_Daily_kWh)", type=["csv"], key="hist")
    if up is None:
        st.info("Carica il tuo CSV per calibrare alpha e beta.")
        return
    df_hist = pd.read_csv(up)
    st.success("Storico caricato ✅")
    model = train_model_with_history(df_hist, lat, lon, tilt, orient, kwp, eff)
    st.session_state["calibration_model"] = model
    c1, c2, c3 = st.columns(3)
    c1.metric("Giorni usati", f"{model['n_days']}")
    c2.metric("RMSE (kWh)", f"{model['rmse']:.2f}" if model['rmse']==model['rmse'] else "—")
    c3.metric("Fattori", f"α={model['alpha']:.3f}, β={model['beta']:.2f}")

def fetch_tomorrow(lat, lon, tilt, orient):
    import pytz
    rome = pytz.timezone("Europe/Rome")
    tomorrow = datetime.now(rome).date() + timedelta(days=1)
    start_loc = rome.localize(datetime.combine(tomorrow, datetime.min.time()))
    end_loc   = rome.localize(datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=23, minutes=45))
    start_iso = start_loc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso   = end_loc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        js = get_meteomatics_json(lat, lon, start_iso, end_iso, tilt, orient)
        d15 = meteomatics_to_df(js)
        st.toast("✅ Dati Meteomatics ricevuti")
    except Exception as e:
        st.warning(f"⚠️ Meteomatics non disponibile, uso Open-Meteo ({e})")
        d15 = get_openmeteo_df(lat, lon, start_loc, end_loc)
    return d15

def page_previsioni(lat, lon, tilt, orient, kwp, eff):
    st.subheader("Previsioni • DOMANI")
    d15 = fetch_tomorrow(lat, lon, tilt, orient)
    d15 = radiation_to_power(d15, kwp, eff)
    model = st.session_state.get("calibration_model", {"alpha":1.0,"beta":0.0})
    alpha = model.get("alpha",1.0); beta = model.get("beta",0.0)
    d15["e_kWh_cal"] = d15["e_kWh"]*alpha + beta/96.0
    d15["p_kW_cal"]  = d15["p_kW"]*alpha
    daily_cal = daily_agg(d15.rename(columns={"e_kWh":"e_kWh_cal"}).assign(e_kWh=d15["e_kWh_cal"]))
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Potenza prevista (kW)**")
        st.line_chart(d15.set_index("ts")["p_kW_cal"])
    with c2:
        st.write("**Energia giornaliera prevista (kWh)**")
        st.metric("kWh (calibrata)", f"{daily_cal['energy_kWh'].sum():.1f}")
    st.download_button("Scarica CSV 15min", data=d15.to_csv(index=False), file_name="forecast_15min_tomorrow.csv")
    st.download_button("Scarica CSV giornaliero", data=daily_cal.to_csv(index=False), file_name="forecast_daily_tomorrow.csv")

def page_mappa(lat, lon):
    st.subheader("Mappa impianto")
    if folium is None:
        st.info("Installa folium e streamlit-folium per la mappa.")
        return
    m = folium.Map(location=[lat, lon], zoom_start=13, control_scale=True)
    folium.Marker([lat, lon], tooltip="Impianto FV").add_to(m)
    st_folium(m, width=None, height=500)

def main():
    page_header()
    lat, lon, tilt, orient, kwp, eff = sidebar_controls()
    page = st.sidebar.radio("Menu", ("modello", "previsioni", "mappa"),
                            captions=["Training storico", "Previsioni domani", "Mappa impianto"], index=0)
    if page == "modello":
        page_modello(lat, lon, tilt, orient, kwp, eff)
    elif page == "previsioni":
        page_previsioni(lat, lon, tilt, orient, kwp, eff)
    elif page == "mappa":
        page_mappa(lat, lon)

if __name__ == "__main__":
    main()
