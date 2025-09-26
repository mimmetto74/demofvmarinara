# ☀️ Solar Forecast - ROBOTRONIX for IMEPOWER

Applicazione **Streamlit** per la previsione della produzione fotovoltaica,
basata su dati storici + API gratuite Open-Meteo (irradianza e nuvolosità).

---

## 🚀 Deploy su Railway

1. Carica questa repo su GitHub
2. Collega la repo a [Railway](https://railway.app)
3. Railway userà automaticamente:
   - `requirements.txt` per installare le dipendenze
   - `Procfile` per avviare Streamlit
   - `runtime.txt` per fissare la versione Python

---

## ▶️ Avvio locale

```bash
pip install -r requirements.txt
streamlit run pv_forecast_all_in_one.py
```

Poi apri [http://localhost:8501](http://localhost:8501).

---

## 🔑 Open-Meteo API

Il modello integra dati meteo (radiazione solare e nuvolosità)
tramite Open-Meteo, che è gratuito e non richiede API key.

---

## 📂 Dataset

La demo utilizza il file:

```
Dataset_Daily_EnergiaSeparata_2020_2025.csv
```

Se vuoi aggiornare i dati, sostituisci questo CSV nella root del progetto.
