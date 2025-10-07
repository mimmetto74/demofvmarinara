# Robotronix Solar V7

Dashboard Streamlit per storico produzione FV, previsioni Meteo → produzione, confronto Storico vs Previsione e mappa.

## Deploy rapido su Railway

1. **Forka/Carica** questa cartella su GitHub.
2. Su Railway: New Project → Deploy from GitHub → seleziona il repo.
3. Railway rileverà `Procfile` e lancerà Streamlit su `$PORT`.

> Le credenziali Meteomatics sono **incluse nel codice** (su richiesta). In produzione ti consigliamo di passare a variabili d'ambiente.

## Esecuzione locale

```bash
pip install -r requirements.txt
streamlit run pv_forecast_all_in_one.py
```

## Dati richiesti

- `Dataset_Daily_EnergiaSeparata_2020_2025.csv` (schema consigliato: `Date,E_INT_Daily_kWh`)
- `Marinara.csv` (alternativo/equivalente, verrà unito se presente)

Se non presenti, l'app carica un esempio minimo.

## Funzionalità principali

- **Storico**: grafici multi‑anno, filtri, totali.
- **Previsioni**: Meteomatics (con fallback Open‑Meteo), stima kW/kWh a 15 minuti e aggregazione giornaliera, download CSV.
- **Storico vs Previsione**: doppia curva, MAE/MAPE/scostamento medio %.
- **Mappa**: marker su coordinate impostate.

## Note sicurezza

- Le credenziali non vengono mai mostrate nella UI.
- L'URL completo delle chiamate Meteomatics non è stampato (a meno di `SHOW_DEBUG_URL=True`).

