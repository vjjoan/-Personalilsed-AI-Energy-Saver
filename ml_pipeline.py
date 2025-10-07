
import os
import glob
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------------
# Config
# -------------------------------
DATA_DIR = os.getenv("DATA_DIR", "./data")  # put house_*.csv here
FORECAST_MODEL_DIR = os.getenv("FORECAST_MODEL_DIR", "./models/forecast")
AE_MODEL_DIR = os.getenv("AE_MODEL_DIR", "./models/autoencoder")
SEQ_LEN = int(os.getenv("SEQ_LEN", "24"))  # hours per sequence window
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "7"))  # days to predict by default

# -------------------------------
# Data Loading
# -------------------------------
def load_house_files(data_dir: str = DATA_DIR) -> Dict[str, pd.DataFrame]:
    files = glob.glob(os.path.join(data_dir, "house_*_1month.csv"))
    house_map = {}
    for path in files:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        # Basic quality checks
        df = df.dropna(subset=["timestamp", "appliance", "usage_kWh"]).copy()
        df["house_id"] = path.split("/")[-1].split("_")[1]  # crude: "house_3_1month.csv" -> "3"
        house_map[f"house_{df['house_id'].iloc[0]}"] = df
    return house_map

# -------------------------------
# Utilities
# -------------------------------
def daily_total(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df["timestamp"].dt.date)["usage_kWh"].sum().reset_index()
    g = g.rename(columns={"timestamp":"date", "usage_kWh":"total_kwh"})
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date").reset_index(drop=True)
    return g

def make_sequences(series: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

def daily_matrix(df: pd.DataFrame, appliance: str) -> pd.DataFrame:
    """Return daily kWh time series for a single appliance."""
    sub = df[df["appliance"] == appliance].copy()
    sub = sub.set_index("timestamp").sort_index()
    ts = sub["usage_kWh"].resample("1D").sum().fillna(0.0)
    return ts.to_frame(name="kwh")

# -------------------------------
# Hourly matrix for anomaly detection
# -------------------------------
def hourly_matrix(df: pd.DataFrame, appliance: str) -> pd.DataFrame:
    """Return hourly kWh time series for a single appliance."""
    sub = df[df["appliance"] == appliance].copy()
    if sub.empty:
        # Handle case with no data
        return pd.DataFrame({"kwh": []})
    sub = sub.set_index("timestamp").sort_index()
    ts = sub["usage_kWh"].resample("1H").sum().fillna(0.0)
    return ts.to_frame(name="kwh")


# -------------------------------
# Deep Anomaly Detection (Autoencoder) on 24h windows per appliance
# -------------------------------
import json

def train_autoencoder(house_id: str, df_house: pd.DataFrame, appliances: List[str]) -> Dict:
    os.makedirs(os.path.join(AE_MODEL_DIR, house_id), exist_ok=True)
    report = {}
    thresholds = {}  # keep thresholds for this house

    # Normalize appliance names from dataset
    available_apps = {a.lower(): a for a in df_house['appliance'].unique()}
    
    for app in appliances:
        clean_app = app.strip("[]").lower()
        if clean_app not in available_apps:
            report[app] = {"status": "skipped", "reason": "appliance not found in dataset"}
            continue
        real_app = available_apps[clean_app]
        
        ts = hourly_matrix(df_house, real_app)["kwh"].values.astype(np.float32)
        print(f"[DEBUG] Appliance {real_app} has {len(ts)} hourly points")
        if len(ts) < SEQ_LEN:
            report[real_app] = {"status": "skipped", "reason": f"only {len(ts)} points"}
            continue

        # Build 24h windows
        X = []
        for i in range(len(ts) - SEQ_LEN + 1):
            X.append(ts[i:i+SEQ_LEN])
        X = np.array(X)

        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)

        ae = build_autoencoder(SEQ_LEN)
        ae.fit(Xs, Xs, epochs=30, batch_size=32, verbose=0)

        app_dir = os.path.join(AE_MODEL_DIR, house_id, real_app.replace(" ", "_"))
        os.makedirs(app_dir, exist_ok=True)
        ae.save(os.path.join(app_dir, "ae.keras"))
        np.save(os.path.join(app_dir, "scaler_min_.npy"), scaler.min_)
        np.save(os.path.join(app_dir, "scaler_scale_.npy"), scaler.scale_)

        # 🔑 Calculate reconstruction error
        recon = ae.predict(Xs, verbose=0)
        mse = np.mean((recon - Xs) ** 2, axis=1)
        suggested_thr = float(np.quantile(mse, 0.99))  # 99th percentile

        thresholds[real_app] = suggested_thr
        report[real_app] = {
            "status": "ok",
            "samples": int(len(X)),
            "suggested_threshold": suggested_thr
        }

    #  Save all thresholds for this house into a JSON file
    thr_file = os.path.join(AE_MODEL_DIR, house_id, "thresholds.json")
    with open(thr_file, "w") as f:
        json.dump(thresholds, f, indent=2)

    return report


# -------------------------------
# Forecasting (LSTM) on daily totals
# -------------------------------
def train_forecast_model(house_id: str, df_house: pd.DataFrame) -> Dict:
    ddf = daily_total(df_house)
    values = ddf["total_kwh"].values.astype(np.float32).reshape(-1, 1)
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    X, y = make_sequences(values_scaled.flatten(), seq_len=7)  # use 7-day window for daily forecasting
    if len(X) < 10:
        return {"status":"skipped", "reason":"not enough data to train"}

    X = X[..., np.newaxis]  # (n, seq, 1)

    model = models.Sequential([
        layers.Input(shape=(X.shape[1], 1)),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=30, batch_size=16, verbose=0)

    # Save model + scaler
    os.makedirs(os.path.join(FORECAST_MODEL_DIR, house_id), exist_ok=True)
    model.save(os.path.join(FORECAST_MODEL_DIR, house_id, "lstm.keras"))
    np.save(os.path.join(FORECAST_MODEL_DIR, house_id, "scaler_min_.npy"), scaler.min_)
    np.save(os.path.join(FORECAST_MODEL_DIR, house_id, "scaler_scale_.npy"), scaler.scale_)
    return {"status":"ok", "samples": int(len(X))}

def predict_future_usage(house_id: str, df_house: pd.DataFrame, days: int = FORECAST_HORIZON) -> Dict:
    # Load
    model_path = os.path.join(FORECAST_MODEL_DIR, house_id, "lstm.keras")
    if not os.path.exists(model_path):
        # train on the fly
        train_forecast_model(house_id, df_house)
    model = models.load_model(model_path)
    scaler = MinMaxScaler()
    # restore scaler
    scaler.min_ = np.load(os.path.join(FORECAST_MODEL_DIR, house_id, "scaler_min_.npy"))
    scaler.scale_ = np.load(os.path.join(FORECAST_MODEL_DIR, house_id, "scaler_scale_.npy"))
    scaler.data_min_ = np.array([0.0])  # dummy
    scaler.data_max_ = (1 - scaler.min_) / scaler.scale_
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.n_samples_seen_ = 1
    scaler.n_features_in_ = 1
    scaler.feature_names_in_ = None

    ddf = daily_total(df_house)
    last_date = ddf["date"].max()
    series = ddf["total_kwh"].values.astype(np.float32)
    scaled = scaler.transform(series.reshape(-1,1)).flatten()
    window = scaled[-7:].tolist()
    preds = []
    for _ in range(days):
        x = np.array(window)[np.newaxis, :, np.newaxis]
        yhat = model.predict(x, verbose=0)[0,0]
        preds.append(yhat)
        window = window[1:] + [yhat]

    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten().tolist()
    future_dates = [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(days)]
    return {"house": house_id, "predictions": dict(zip(future_dates, [float(p) for p in preds_inv]))}

# -------------------------------
# Deep Anomaly Detection (Autoencoder) on 24h windows per appliance
# -------------------------------
def build_autoencoder(input_dim: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(input_dim, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def load_scaler_from_dir(d):
    scaler = MinMaxScaler()
    scaler.min_ = np.load(os.path.join(d, "scaler_min_.npy"))
    scaler.scale_ = np.load(os.path.join(d, "scaler_scale_.npy"))
    scaler.data_min_ = np.zeros_like(scaler.min_)
    scaler.data_max_ = (1 - scaler.min_) / scaler.scale_
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.n_samples_seen_ = 1
    scaler.n_features_in_ = scaler.min_.shape[0]
    scaler.feature_names_in_ = None
    return scaler

def detect_anomalies(house_id: str, df_house: pd.DataFrame, appliances: List[str], threshold: float = 0.01) -> Dict:
    results = {}
    # Map dataset appliances to real names
    available_apps = {a.lower(): a for a in df_house['appliance'].unique()}

    # 🔑 Load saved thresholds if available
    thr_file = os.path.join(AE_MODEL_DIR, house_id, "thresholds.json")
    if os.path.exists(thr_file):
        with open(thr_file, "r") as f:
            saved_thresholds = json.load(f)
    else:
        saved_thresholds = {}

    for app in appliances:
        clean_app = app.strip("[]").lower()
        if clean_app not in available_apps:
            results[app] = []
            continue
        real_app = available_apps[clean_app]

        app_dir = os.path.join(AE_MODEL_DIR, house_id, real_app.replace(" ", "_"))
        model_path = os.path.join(app_dir, "ae.keras")
        if not os.path.exists(model_path):
            # train on the fly for this appliance
            train_autoencoder(house_id, df_house, [real_app])

        ae = tf.keras.models.load_model(model_path)
        scaler = load_scaler_from_dir(app_dir)

        ts = hourly_matrix(df_house, real_app)
        values = ts["kwh"].values.astype(np.float32)
        if len(values) < SEQ_LEN + 1:
            results[real_app] = []
            continue

        # Build windows + keep end timestamps
        X, ends = [], []
        for i in range(len(values) - SEQ_LEN + 1):
            X.append(values[i:i+SEQ_LEN])
            ends.append(ts.index[i+SEQ_LEN-1])
        X = np.array(X)
        Xs = scaler.transform(X)
        recon = ae.predict(Xs, verbose=0)
        mse = np.mean((recon - Xs) ** 2, axis=1)

        # 🔑 Use saved threshold if available, else fallback to percentile
        if real_app in saved_thresholds:
            thr = float(saved_thresholds[real_app])
        else:
            thr = np.quantile(mse, 1 - threshold)

        anomalies = [str(ends[i]) for i, e in enumerate(mse) if e >= thr]
        results[real_app] = anomalies

    return results



# -------------------------------
# Recommendations (Hybrid)
# -------------------------------
def recommendations(house_id: str, df_house: pd.DataFrame) -> List[Dict]:
    recs = []
    # Simple rules + learned stats
    ddf = daily_total(df_house)
    last7 = ddf.tail(7)["total_kwh"]
    if len(last7) >= 7 and last7.mean() > ddf["total_kwh"].mean() * 1.2:
        recs.append({"tip":"Your last 7 days usage is ~20% higher than your average. Check AC schedules and lighting habits."})
    # Peak hour
    df_house["hour"] = df_house["timestamp"].dt.hour
    peak_hour = int(df_house.groupby("hour")["usage_kWh"].sum().idxmax())
    if peak_hour in [18,19,20,21]:
        recs.append({"tip":"Evening peak detected. Consider delaying washing/dishwasher to late-night or morning."})
    # Appliances baseline
    top_app = df_house.groupby("appliance")["usage_kWh"].sum().idxmax()
    recs.append({"tip": f"Top energy consumer: {top_app}. Optimizing its runtime can reduce your bill the most."})
    return recs

def daily_totals(df):
    if "usage_kWh" not in df.columns:
        raise ValueError("Missing 'usage_kWh' column in CSV")
    return df.groupby(df["timestamp"].dt.date)["usage_kWh"].sum().to_dict()

def hourly_totals(df):
    if "usage_kWh" not in df.columns:
        raise ValueError("Missing 'usage_kWh' column in CSV")
    return df.groupby(df["timestamp"].dt.hour)["usage_kWh"].sum().to_dict()

def per_appliance_data(df_house):
    data = {}
    if "appliance" not in df_house.columns or "usage_kWh" not in df_house.columns:
        return data
    for app in df_house["appliance"].unique():
        sub = df_house[df_house["appliance"] == app].copy()
        ts = sub.set_index("timestamp").resample("H")["usage_kWh"].sum().fillna(0)
        data[app] = [{"timestamp": str(idx), "kwh": float(val)} for idx, val in ts.items()]
    return data


def peak_min_hours(df):
    hourly = hourly_totals(df)
    peak_hour = max(hourly, key=hourly.get)
    min_hour = min(hourly, key=hourly.get)
    return peak_hour, min_hour


