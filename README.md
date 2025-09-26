# ☀️ Solar Forecast - TESEO-RX for IMEPOWER

Applicazione **Streamlit** per la previsione della produzione di un impianto fotovoltaico, basata su:  
- **Dati storici** di produzione e irraggiamento  
- **Previsioni Meteomatics API** (irraggiamento, copertura nuvolosa, temperatura e vento)

## 📂 Struttura repository

```
/ (root)
├── pv_forecast_all_in_one.py     # Script principale Streamlit
├── requirements.txt              # Dipendenze Python
├── Procfile                      # Istruzioni avvio su Railway
├── runtime.txt                   # Versione Python
├── .streamlit/config.toml        # Configurazione interfaccia
└── Dataset_Daily_EnergiaSeparata_2020_2025.csv.gz   # Dataset storico
```

## 🚀 Funzionalità

- 🔑 Login sicuro (utente: `FVMANAGER`, password: `admin2025`)
- 📊 Analisi storica della produzione FV
- 🛠️ Addestramento modello con regressione lineare (storico 2020–2024, test 2025)
- 🔮 Previsioni day-ahead e two-day-ahead usando Meteomatics API
- 🌦️ Parametri Meteomatics integrati:
  - `solar_rad:mean:W` → Irraggiamento solare medio
  - `cloud_cover:tot:p` → Copertura nuvolosa totale (%)
  - `t_2m:C` → Temperatura a 2m
  - `wind_speed_10m:ms` → Vento a 10m

## 📦 Installazione locale

```bash
git clone https://github.com/TUO_USER/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt
streamlit run pv_forecast_all_in_one.py
```

## ☁️ Deploy su Railway

1. Carica la repo su GitHub
2. Collega la repo a [Railway](https://railway.app/)
3. Railway userà `Procfile` per avviare Streamlit automaticamente

## 🔑 Credenziali

- Username: `FVMANAGER`
- Password: `admin2025`
