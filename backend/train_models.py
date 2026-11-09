import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn

# Ensure local imports work regardless of execution folder
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from preprocessing import load_and_merge_data

def compute_wmae(y_true, y_pred, is_holiday):
    """
    Weighted Mean Absolute Error (WMAE) computation.
    Holiday weeks receive weight 5; regular weeks receive weight 1.
    """
    weights = np.where(is_holiday == 1, 5, 1)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))

class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out

def train_and_evaluate():
    df = load_and_merge_data()
    
    # Chronological Split: 80% Train, 20% Test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()
    
    feature_cols = ["Store", "Dept", "Size", "Temperature", "Fuel_Price", 
                    "CPI", "Unemployment", "Year", "Month", "Week", "IsHolidayInt"]
    
    X_train, y_train = train_df[feature_cols].values, train_df["Weekly_Sales"].values
    X_test, y_test = test_df[feature_cols].values, test_df["Weekly_Sales"].values
    holidays_test = test_df["IsHolidayInt"].values
    
    metrics = {}
    models_dir = os.path.join(backend_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Baseline: Seasonal Naive
    print("Training 1/5: Seasonal Naive Baseline...")
    test_df["Naive_Pred"] = train_df.groupby(["Store", "Dept"])["Weekly_Sales"].transform("mean")
    naive_preds = test_df["Naive_Pred"].fillna(y_train.mean()).values
    metrics["Seasonal Naive"] = {
        "MAE": round(float(mean_absolute_error(y_test, naive_preds)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, naive_preds))), 2),
        "WMAE": round(compute_wmae(y_test, naive_preds, holidays_test), 2)
    }
    
    # 2. Random Forest Regressor
    print("Training 2/5: Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    metrics["Random Forest"] = {
        "MAE": round(float(mean_absolute_error(y_test, rf_preds)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, rf_preds))), 2),
        "WMAE": round(compute_wmae(y_test, rf_preds, holidays_test), 2)
    }
    with open(os.path.join(models_dir, "random_forest.pkl"), "wb") as f:
        pickle.dump(rf, f)
        
    # 3. XGBoost Regressor
    print("Training 3/5: XGBoost Regressor...")
    xgb = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=6, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    metrics["XGBoost"] = {
        "MAE": round(float(mean_absolute_error(y_test, xgb_preds)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, xgb_preds))), 2),
        "WMAE": round(compute_wmae(y_test, xgb_preds, holidays_test), 2)
    }
    with open(os.path.join(models_dir, "xgboost.pkl"), "wb") as f:
        pickle.dump(xgb, f)

    # 4. Deep Learning: PyTorch LSTM Network
    print("Training 4/5: LSTM Neural Network...")
    train_size = min(30000, len(X_train))
    test_size = min(10000, len(X_test))
    
    X_tr_t = torch.tensor(X_train[:train_size], dtype=torch.float32).unsqueeze(1)
    y_tr_t = torch.tensor(y_train[:train_size], dtype=torch.float32).unsqueeze(1)
    X_te_t = torch.tensor(X_test[:test_size], dtype=torch.float32).unsqueeze(1)
    
    lstm = LSTMModel(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
    
    lstm.train()
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = lstm(X_tr_t)
        loss = criterion(outputs, y_tr_t)
        loss.backward()
        optimizer.step()
        
    lstm.eval()
    with torch.no_grad():
        lstm_preds = lstm(X_te_t).squeeze().numpy()
        
    metrics["LSTM Neural Network"] = {
        "MAE": round(float(mean_absolute_error(y_test[:test_size], lstm_preds)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test[:test_size], lstm_preds))), 2),
        "WMAE": round(compute_wmae(y_test[:test_size], lstm_preds, holidays_test[:test_size]), 2)
    }
    torch.save(lstm.state_dict(), os.path.join(models_dir, "lstm_model.pt"))

    # 5. Time Series: SARIMA
    print("Training 5/5: SARIMA Time Series...")
    weekly_agg = train_df.groupby("Date")["Weekly_Sales"].sum()
    sarima = SARIMAX(weekly_agg.values, order=(1,1,0)).fit(disp=False)
    sarima_pred_val = float(sarima.forecast(steps=1)[0]) / 2500
    sarima_preds = np.full_like(y_test, sarima_pred_val)
    metrics["SARIMA"] = {
        "MAE": round(float(mean_absolute_error(y_test, sarima_preds)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, sarima_preds))), 2),
        "WMAE": round(compute_wmae(y_test, sarima_preds, holidays_test), 2)
    }

    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Model training complete! Evaluation matrix saved to: {metrics_path}")

if __name__ == "__main__":
    train_and_evaluate()
