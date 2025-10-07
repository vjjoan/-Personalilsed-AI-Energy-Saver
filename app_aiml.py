import os
import json
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi.responses import JSONResponse
import plotly.express as px
import plotly.io as pio
from enum import Enum

class ThresholdOptions(str, Enum):
    very_high = "0.1"
    high      = "0.05"
    medium    = "0.01"
    low       = "0.005"


# Import your ML pipeline functions
from ml_pipeline import (
    load_house_files,
    train_forecast_model,
    predict_future_usage,
    train_autoencoder,
    detect_anomalies,
    recommendations,
    daily_totals,
    hourly_totals,
    peak_min_hours,
    per_appliance_data
)

# -------------------------------
# Config
# -------------------------------
DATA_DIR = os.getenv("DATA_DIR", "./data")

# -------------------------------
# App
# -------------------------------
app = FastAPI(
    title="Personalized AI Energy Saver",
    description="Backend for anomaly detection, forecasting, and recommendations",
    version="1.0.0",
)
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Alert Manager for WebSockets
# -------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            await connection.send_json(message)

alert_manager = ConnectionManager()

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    """Clients (frontend apps) connect here to receive real-time anomaly alerts"""
    await alert_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        alert_manager.disconnect(websocket)

# -------------------------------
# Root → redirect to /docs
# -------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")   


# -------------------------------
# Endpoints
# -------------------------------

@app.get("/houses")
def list_houses():
    """List all available house IDs from dataset"""
    house_map = load_house_files(DATA_DIR)
    return {"houses": list(house_map.keys())}

@app.post("/train/forecast")
def train_forecast(house_id: str = Query(..., description="House ID like house_1")):
    """Train forecasting model for a house"""
    house_map = load_house_files(DATA_DIR)
    if house_id not in house_map:
        return {"error": "Unknown house"}
    return train_forecast_model(house_id, house_map[house_id])

@app.get("/predict/forecast")
def forecast_usage(
    house_id: str = Query(..., description="House ID like house_1"),
    days: int = Query(7, description="Number of days to forecast"),
):
    """Predict future energy usage for given house"""
    house_map = load_house_files(DATA_DIR)
    if house_id not in house_map:
        return {"error": "Unknown house"}
    return predict_future_usage(house_id, house_map[house_id], days)

@app.post("/train/anomaly")
def train_anomaly(house_id: str = Query(...), appliances: List[str] = Query(...)):
    """Train autoencoder anomaly models for appliances of a house"""
    house_map = load_house_files(DATA_DIR)
    if house_id not in house_map:
        return {"error": "Unknown house"}
    return train_autoencoder(house_id, house_map[house_id], appliances)

@app.get("/predict/anomaly")
async def predict_anomaly(
    house_id: str = Query(..., description="House ID like house_1"),
    appliances: List[str] = Query(..., description="List of appliances (e.g., Fridge,AC)"),
    threshold: ThresholdOptions = Query(
        ThresholdOptions.medium,
        description="Percentile threshold for anomaly detection (choose from dropdown)"
    ),
):
    """
    Predict anomalies for appliances of a house.

    - Uses trained thresholds if available.
    - Otherwise applies a selected percentile threshold.
    - Triggers WebSocket alerts (`/ws/alerts`) when anomalies are found.
    """
    house_map = load_house_files(DATA_DIR)
    if house_id not in house_map:
        return {"error": "Unknown house"}

    # convert enum value (string) to float for detect_anomalies
    thr_value = float(threshold.value)

    # Run anomaly detection (detect_anomalies expects a float threshold)
    results = detect_anomalies(house_id, house_map[house_id], appliances, thr_value)

    #  Send alerts if anomalies found
    for app, timestamps in results.items():
        if timestamps:
            alert_msg = {
                "house_id": house_id,
                "appliance": app,
                "anomalies": timestamps,
                "message": f" Anomaly detected in {app} of {house_id} (Threshold: {threshold.name})"
            }
            await alert_manager.broadcast(alert_msg)

    return results

@app.get("/recommendations")
def get_recommendations(house_id: str = Query(...)):
    """Get energy-saving recommendations for a house"""
    house_map = load_house_files(DATA_DIR)
    if house_id not in house_map:
        return {"error": "Unknown house"}
    return recommendations(house_id, house_map[house_id])

# -------------------------------
# Plotting endpoints
# -------------------------------
@app.get("/plot/house")
def plot_house(house_id: str = Query(...), appliances: List[str] = Query([]), forecast_days: int = Query(7)):
    """
    Returns:
    - Daily usage
    - Hourly usage
    - Peak/min hours
    - Per-appliance hourly usage
    - Anomalies per appliance
    - Forecasted usage for next N days
    """
    house_map = load_house_files(DATA_DIR)
    if house_id not in house_map:
        return {"error": "Unknown house"}

    df_house = house_map[house_id]

    daily = daily_totals(df_house)
    hourly = hourly_totals(df_house)
    peak, min_hour = peak_min_hours(df_house)

    if not appliances:
        appliances = df_house['appliance'].unique().tolist()

    per_app = per_appliance_data(df_house)
    anomalies = detect_anomalies(house_id, df_house, appliances)

    # Forecast
    forecast = predict_future_usage(house_id, df_house, days=forecast_days)

    return {
        "house_id": house_id,
        "daily_usage": daily,
        "hourly_usage": hourly,
        "peak_hour": peak,
        "min_hour": min_hour,
        "per_appliance": per_app,
        "anomalies": anomalies,
        "forecast": forecast["predictions"]
    }

@app.get("/plot/compare")
def compare_houses():
    """Compare daily total usage of all houses"""
    house_map = load_house_files(DATA_DIR)
    result = {}
    for house_id, df_house in house_map.items():
        result[house_id] = daily_totals(df_house)
    return {"houses": list(house_map.keys()), "daily_totals": result}

@app.get("/plot/appliance")
def appliance_plot(house_id: str, appliance: str, forecast_days: int = Query(7)):
    """
    Returns:
    - Daily & hourly usage for a single appliance
    - Anomalies
    """
    house_map = load_house_files(DATA_DIR)
    if house_id not in house_map:
        return {"error": "Unknown house"}

    df_house = house_map[house_id]
    df_appl = df_house[df_house["appliance"].str.lower() == appliance.lower()]
    if df_appl.empty:
        return {"error": f"Appliance {appliance} not found in {house_id}"}

    daily = daily_totals(df_appl)
    hourly = hourly_totals(df_appl)
    anomalies = detect_anomalies(house_id, df_house, [appliance])

    return {
        "house_id": house_id,
        "appliance": appliance,
        "daily_usage": daily,
        "hourly_usage": hourly,
        "anomalies": anomalies.get(appliance, {}),
    }

@app.get("/plot/dashboard")
def dashboard_plot(house_id: str = Query(...), appliances: List[str] = Query([]), forecast_days: int = Query(7)):
    """
    Returns a single JSON with:
    - Daily usage
    - Hourly usage
    - Peak/min hours
    - Per-appliance hourly usage with anomalies flagged
    - Forecast
    - Recommendations
    """
    house_map = load_house_files(DATA_DIR)
    if house_id not in house_map:
        return {"error": "Unknown house"}

    df_house = house_map[house_id]
    daily = daily_totals(df_house)
    hourly = hourly_totals(df_house)
    peak, min_hour = peak_min_hours(df_house)

    if not appliances:
        appliances = df_house['appliance'].unique().tolist()

    # Get per-appliance hourly data
    per_app = per_appliance_data(df_house)

    # Detect anomalies
    anomalies = detect_anomalies(house_id, df_house, appliances)

    # Overlay anomalies into per-appliance data
    for app in per_app:
        anomaly_set = set(anomalies.get(app, []))
        for entry in per_app[app]:
            entry["anomaly"] = entry["timestamp"] in anomaly_set

    # Forecast
    forecast = predict_future_usage(house_id, df_house, forecast_days)

    # Recommendations
    recs = recommendations(house_id, df_house)

    return {
        "house_id": house_id,
        "daily_usage": daily,
        "hourly_usage": hourly,
        "peak_hour": peak,
        "min_hour": min_hour,
        "per_appliance": per_app,
        "forecast": forecast["predictions"],
        "recommendations": recs
    }

@app.get("/location/usage")
def location_usage():
    """Aggregate energy usage by location and return as a plotly graph."""
    houses = load_house_files()
    if not houses:
        return {"error": "No house data found"}

    # Load location mapping file (CSV with: house_id,location)
    loc_map = pd.read_csv("./data/house_locations.csv")
    
    all_data = []
    for hid, df in houses.items():
        df_sum = df.groupby(df["timestamp"].dt.date)["usage_kWh"].sum().reset_index()
        df_sum = df_sum.rename(columns={"timestamp": "date"})  # rename properly
        df_sum["house_id"] = hid
        all_data.append(df_sum)
    all_data = pd.concat(all_data)
    merged = all_data.merge(loc_map, on="house_id")

    fig = px.line(
        merged,
        x="date",
        y="usage_kWh",
        color="location",
        labels={"date": "Date", "usage_kWh": "Energy Usage (kWh)", "location": "Location"},
        title="Energy Usage by Location"
    )
    fig.update_traces(showlegend=True)
    fig.update_xaxes(rangeslider_visible=True)

    graph_json = pio.to_json(fig)
    return JSONResponse(content=json.loads(graph_json))


@app.get("/forecast/cost/{house_id}")
def forecast_cost(house_id: str, days: int = 7):
    """Forecast cost instead of kWh for a house."""
    houses = load_house_files()
    if house_id not in houses:
        return {"error": "Unknown house"}
    
    preds = predict_future_usage(house_id, houses[house_id], days)
    # Convert kWh to cost using avg unit cost from dataset
    avg_cost_per_kwh = houses[house_id]["cost"].sum() / houses[house_id]["usage_kWh"].sum()
    preds_cost = {d: round(v * avg_cost_per_kwh, 2) for d, v in preds["predictions"].items()}

    fig = px.line(
        x=list(preds_cost.keys()),
        y=list(preds_cost.values()),
        labels={"x": "Date", "y": "Cost (₹)"},
        title=f"Forecasted Cost for {house_id}"
    )
    fig.update_traces(name=f"{house_id} Forecasted Cost", showlegend=True)
    fig.update_xaxes(rangeslider_visible=True)

    graph_json = pio.to_json(fig)
    return JSONResponse(content=json.loads(graph_json))


@app.get("/compare/houses")
def compare_houses():
    """Compare energy usage & cost across houses in one graph."""
    houses = load_house_files()
    if not houses:
        return {"error": "No house data found"}

    all_data = []
    for hid, df in houses.items():
        df_sum = df.groupby(df["timestamp"].dt.date).agg(
            {"usage_kWh": "sum", "cost": "sum"}
        ).reset_index()
        df_sum["house_id"] = hid
        all_data.append(df_sum)
    all_data = pd.concat(all_data)

    # Energy Usage Graph
    fig_usage = px.line(
        all_data,
        x="timestamp",
        y="usage_kWh",
        color="house_id",
        labels={"timestamp": "Date", "usage_kWh": "Energy Usage (kWh)", "house_id": "House"},
        title="Comparison of Energy Usage across Houses"
    )
    fig_usage.update_traces(showlegend=True)
    fig_usage.update_xaxes(rangeslider_visible=True)

    # Cost Graph
    fig_cost = px.line(
        all_data,
        x="timestamp",
        y="cost",
        color="house_id",
        labels={"timestamp": "Date", "cost": "Cost (₹)", "house_id": "House"},
        title="Comparison of Energy Cost across Houses"
    )
    fig_cost.update_traces(showlegend=True)
    fig_cost.update_xaxes(rangeslider_visible=True)

    return {
        "usage_graph": json.loads(pio.to_json(fig_usage)),
        "cost_graph": json.loads(pio.to_json(fig_cost)),
    }

