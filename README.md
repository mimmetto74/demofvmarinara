# 🌞 Solar Forecast - ROBOTRONIX for IMEPOWER

Demo Streamlit con addestramento modello su dati storici e previsione produzione FV
usando **Open-Meteo** (`shortwave_radiation_sum` + `cloudcover_mean`).

## ▶️ Avvio locale
```bash
pip install -r requirements.txt
streamlit run pv_forecast_all_in_one.py
```
Login: `FVMANAGER` / `admin2025`

## 🚀 Deploy su Railway
- Connetti la repo GitHub a Railway
- Il Procfile avvia Streamlit su porta 8080
