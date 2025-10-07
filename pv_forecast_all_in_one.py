
# pv_forecast_all_in_one.py
# Robotronix Solar V8 (menu stile V4, senza "storico", con TRAIN modello da storici e previsioni a DOMANI)
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

# ---------------- Meteomatics: credenziali + fetch ----------------
METEO_USER = "teseospa-eiffageenergiesystemesitaly_daniello_fabio"
METEO_PASS = "6S8KTHPbrUlp6523T9Xd"
SHOW_DEBUG_URL = False  # non mostrare URL/credenziali in UI

def _orient_cardinal(deg: int) -> str:
    m = {0: "N", 90: "E", 180: "S", 270: "W"}
    if deg in m:
        return m[deg]
    return m[min(m.keys(), key=lambda k: abs(k - deg))]

def get_meteomatics_json(lat: float, lon: float, start_iso: str, end_iso: str,
                         tilt_deg: int, orient_deg: int) -> dict:
    orient = _orient_cardinal(int(orient_deg))
    param = f"global_rad_tilt_{int(tilt_deg)}_orientation_{orient}:W,total_cloud_cover:p"
    url = (
        f"https://api.meteomatics.com/"
        f"{start_iso}--{end_iso}:PT15M/{param}/{lat:.6f},{lon:.6f}/json"
    )
    if SHOW_DEBUG_URL:
        st.write("DEBUG MM URL:", url)
    r = requests.get(url, auth=(METEO_USER, METEO_PASS), timeout=25)
    r.raise_for_status()
    return r.json()

def meteomatics_to_df(payload: dict) -> pd.DataFrame:
    if "data" not in payload or not payload["data"]:
        return pd.DataFrame(columns=["ts", "global_rad_Wm2", "tcc_pct"])
    data_by_param = {entry["parameter"]: entry["coordinates"][0]["dates"] for entry in payload["data"]}
    ts = [pd.to_datetime(d["date"]) for d in next(iter(data_by_param.values()))]
    df = pd.DataFrame({"ts": ts})
    # rad
    k_glob = next((k for k in data_by_param if k.startswith("global_rad")), None)
    df["global_rad_Wm2"] = [d["value"] for d in data_by_param.get(k_glob, [])] if k_glob else np.nan
    # cloud cover
    if "total_cloud_cover" in data_by_param:
        df["tcc_pct"] = [d["value"] for d in data_by_param["total_cloud_cover"]]
    else:
        df["tcc_pct"] = np.nan
    # UTC -> Europe/Rome
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("UTC").dt.tz_convert("Europe/Rome")
    return df

# ---------------- Conversione radiazione -> potenza/energia -------
def radiation_to_power(df: pd.DataFrame, kwp: float, system_eff: float = 0.85) -> pd.DataFrame:
    if df.empty:
        return df.assign(p_kW=np.nan, e_kWh=np.nan)
    area_eff = kwp / 0.2  # m2 equivalenti con rendimento pannello 20%
    p_kw = df["global_rad_Wm2"].clip(lower=0) * area_eff * system_eff / 1000.0
    out = df.copy()
    out["p_kW"] = p_kw
    out["e_kWh"] = out["p_kW"] * 0.25  # 15 minuti = 0.25 h
    return out

def daily_agg(df15: pd.DataFrame) -> pd.DataFrame:
    if df15.empty:
        return pd.DataFrame(columns=["date","energy_kWh"])
    d = df15.copy()
    d["date"] = d["ts"].dt.tz_convert("Europe/Rome").dt.date
    g = d.groupby("date")["e_kWh"].sum().reset_index().rename(columns={"e_kWh":"energy_kWh"})
    return g

# ---------------- Training modello (alpha, beta, shift) -----------
def train_model_with_history(df_hist_daily: pd.DataFrame, lat, lon, tilt, orient, kwp, eff):
    """Allena un modello lineare y = alpha * y_pred + beta.
       df_hist_daily: colonne attese ['Date','E_INT_Daily_kWh'] oppure ['date','energy_kWh']
    """
    if df_hist_daily is None or df_hist_daily.empty:
        return {"alpha":1.0,"beta":0.0,"n_days":0,"rmse":np.nan}
    # normalizza colonne
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

    # scarica Meteomatics per l'intervallo coperto dallo storico (margine 1g)
    start = pd.to_datetime(min(df["date"])) - pd.Timedelta(days=1)
    end   = pd.to_datetime(max(df["date"])) + pd.Timedelta(days=1)
    start_iso = start.strftime("%Y-%m-%dT00:00:00Z")
    end_iso   = end.strftime("%Y-%m-%dT23:45:00Z")
    try:
        js = get_meteomatics_json(lat, lon, start_iso, end_iso, tilt, orient)
        d15 = meteomatics_to_df(js)
        d15 = radiation_to_power(d15, kwp, eff)
        pred_daily = daily_agg(d15)
    except Exception as e:
        st.error(f"Errore fetch Meteomatics per training: {e}")
        return {"alpha":1.0,"beta":0.0,"n_days":0,"rmse":np.nan}

    m = pd.merge(df, pred_daily, on="date", how="inner", suffixes=("_real","_pred"))
    if m.empty:
        st.warning("Nessuna sovrapposizione date tra storico e previsione per training.")
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

# ---------------- UI (menu stile V4: modello, previsioni, mappa) --
def page_header():
    st.set_page_config(page_title="Robotronix Solar V8", layout="wide")
    st.title("Robotronix Solar • V8")
    st.caption("Menu stile V4 • Modello (training) • Previsioni DOMANI • Mappa")

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
        st.info("Suggerimento: puoi caricare il tuo CSV per addestrare la calibrazione (alpha, beta).")
        return
    df_hist = pd.read_csv(up)
    st.success("Storico caricato ✅")
    model = train_model_with_history(df_hist, lat, lon, tilt, orient, kwp, eff)
    st.session_state["calibration_model"] = model

    c1, c2, c3 = st.columns(3)
    c1.metric("Giorni usati", f"{model['n_days']}")
    c2.metric("RMSE (kWh)", f"{model['rmse']:.2f}" if model['rmse']==model['rmse'] else "—")
    c3.metric("Fattori", f"alpha={model['alpha']:.3f}, beta={model['beta']:.2f}")
    st.caption("Il modello stima: energia_reale ≈ alpha·energia_stimata + beta")

def fetch_tomorrow(lat, lon, tilt, orient):
    # Intervallo: 00:00 → 23:45 del giorno successivo in Europe/Rome, poi in UTC
    import pytz
    rome = pytz.timezone("Europe/Rome")
    tomorrow = datetime.now(rome).date() + timedelta(days=1)
    start_loc = rome.localize(datetime.combine(tomorrow, datetime.min.time()))
    end_loc   = rome.localize(datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=23, minutes=45))
    start_iso = start_loc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso   = end_loc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    js = get_meteomatics_json(lat, lon, start_iso, end_iso, tilt, orient)
    d15 = meteomatics_to_df(js)
    return d15

def page_previsioni(lat, lon, tilt, orient, kwp, eff):
    st.subheader("Previsioni • DOMANI")
    d15 = fetch_tomorrow(lat, lon, tilt, orient)
    d15 = radiation_to_power(d15, kwp, eff)

    # Calibrazione da training
    model = st.session_state.get("calibration_model", {"alpha":1.0,"beta":0.0})
    alpha = model.get("alpha",1.0); beta = model.get("beta",0.0)
    d15["e_kWh_cal"] = d15["e_kWh"]*alpha + beta/96.0
    d15["p_kW_cal"]  = d15["p_kW"]*alpha

    daily_cal = daily_agg(d15.rename(columns={"e_kWh":"e_kWh_cal"}).assign(e_kWh=d15["e_kWh_cal"]))

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Potenza prevista (kW) – calibrata**")
        st.line_chart(d15.set_index("ts")["p_kW_cal"])
    with c2:
        st.write("**Energia giornaliera prevista (kWh)**")
        st.metric("kWh (calibrata)", f"{daily_cal['energy_kWh'].sum():.1f}")

    st.download_button("Scarica previsione 15 min (CSV)", data=d15.to_csv(index=False), file_name="forecast_15min_tomorrow.csv")
    st.download_button("Scarica previsione giornaliera (CSV)", data=daily_cal.to_csv(index=False), file_name="forecast_daily_tomorrow.csv")

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

    # === MENU stile V4 (senza STORICO): modello • previsioni • mappa ===
    page = st.sidebar.radio("Menu", ("modello", "previsioni", "mappa"),
                            captions=["Training con storico", "Previsione per domani", "Posizione impianto"],
                            index=0)

    if page == "modello":
        page_modello(lat, lon, tilt, orient, kwp, eff)
    elif page == "previsioni":
        page_previsioni(lat, lon, tilt, orient, kwp, eff)
    elif page == "mappa":
        page_mappa(lat, lon)

if __name__ == "__main__":
    main()
