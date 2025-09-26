# ☀️ Solar Forecast - ROBOTRONIX for IMEPOWER

Applicazione **Streamlit** per la previsione della produzione fotovoltaica,
basata su dati storici + Meteomatics API.

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

## 🔑 Meteomatics API

Il modello integra dati meteo (radiazione, nuvolosità, vento, temperatura)
tramite Meteomatics API.

Aggiorna le tue credenziali nello script se necessario:

```python
USERNAME = "xxxxx"
PASSWORD = "xxxxx"
```

---

## 📂 Dataset

La demo utilizza il file:

```
Dataset_Daily_EnergiaSeparata_2020_2025.csv
```

Se vuoi aggiornare i dati, sostituisci questo CSV nella root del progetto.
