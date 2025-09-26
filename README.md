# 🌞 Solar Forecast - ROBOTRONIX for IMEPOWER

Questa è una demo Streamlit per la previsione della produzione fotovoltaica.

## Funzionalità
- Login (utente **FVMANAGER**, password **admin2025**)
- Addestramento modello da CSV con radiazione (`G_M0_Wm2`), copertura nuvolosa (`cloud_cover`) e produzione (`E_INT_Daily_kWh`)
- Previsione giorno successivo o dopodomani tramite API **Open-Meteo**

## Deploy
1. Carica i file su **GitHub**
2. Connetti il repo a **Railway**
3. Imposta il comando di avvio:  
   ```bash
   web: streamlit run pv_forecast_all_in_one.py
   ```
