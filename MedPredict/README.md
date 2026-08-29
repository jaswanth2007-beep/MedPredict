# MedPredict — Predictive Maintenance Demo

A portfolio-ready predictive-maintenance project inspired by the NASA C-MAPSS benchmark.
It predicts Remaining Useful Life (RUL), assigns risk levels, and provides a Streamlit dashboard.

> Note: C-MAPSS is simulated turbofan-engine data, not GE HealthCare equipment data.

## Run

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python download_data.py
python train.py
streamlit run app.py
```

The dashboard runs at http://localhost:8501.

## Project structure

- `download_data.py` downloads the public NASA C-MAPSS archive.
- `train.py` preprocesses FD001 and trains an XGBoost RUL model.
- `app.py` provides an interactive dashboard.
- `src/` contains reusable preprocessing and prediction functions.
- `data/raw/` stores downloaded data.
- `models/` stores the trained model.

## Interview positioning

This is a portfolio/research prototype. It should not be presented as a clinically validated or production maintenance system.
