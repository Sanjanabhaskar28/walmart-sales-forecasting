# Multivariate Time Series Forecasting - Walmart Weekly Sales

Complete end-to-end Machine Learning & Deep Learning project built for Walmart Weekly Sales forecasting.

## Folder Architecture
- `data/` : Place `train.csv`, `stores.csv`, `features.csv`, and `test.csv` here.
- `backend/preprocessing.py` : Data loading, merging, cleaning, and feature engineering.
- `backend/train_models.py` : Trains 5 models (Seasonal Naive, Random Forest, XGBoost, PyTorch LSTM, SARIMA), computes WMAE/MAE/RMSE metrics, and saves model artifacts & `backend/models/metrics.json`.
- `backend/app.py` : Flask REST API running on `http://127.0.0.1:5000`.
- `frontend/` : Web Dashboard (`index.html`, `styles.css`, `app.js`) rendering Chart.js visualizations.

## Quick Execution Steps
1. Place raw CSV files into `data/`.
2. Install packages: `pip install -r requirements.txt`
3. Train models: `python backend/train_models.py`
4. Run server: `python backend/app.py`
5. Open `frontend/index.html` in your web browser.
