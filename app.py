from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import joblib

app = Flask(__name__)

DATA_FILE = 'sensor_data.csv'
MODEL_FILE = 'weather_model.pkl'
SCALER_FILE = 'scaler.pkl'
REFRESH_INTERVAL = 60  # seconds

WEATHER_CONDITIONS = {
    0: "Clear/Sunny",
    1: "Partly Cloudy",
    2: "Cloudy",
    3: "Light Rain",
    4: "Heavy Rain"
}

def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    
    df = pd.read_csv(DATA_FILE)
    if df.empty:
        return None
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df

@app.route('/')
def index():
    return render_template('index.html', refresh_interval=REFRESH_INTERVAL * 1000)

@app.route('/api/current')
def current_readings():
    df = load_data()
    if df is None or df.empty:
        return jsonify({'error': 'No data available'})
    
    latest = df.iloc[-1]
    
    current_data = {
        'timestamp': latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
        'temperature': round(float(latest['temperature']), 1),
        'humidity': round(float(latest['humidity']), 1),
        'light': round(float(latest['light']), 1),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return jsonify(current_data)

@app.route('/api/history')
def history_data():
    df = load_data()
    if df is None or df.empty:
        return jsonify({'error': 'No data available'})
    
    cutoff = datetime.now() - timedelta(hours=48)
    recent_df = df[df['timestamp'] > pd.Timestamp(cutoff)]
    
    if len(recent_df) < 10:
        recent_df = df
    
    if len(recent_df) > 100:
        step = len(recent_df) // 100
        recent_df = recent_df.iloc[::step, :]
    
    history = []
    for _, row in recent_df.iterrows():
        history.append({
            'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': round(row['temperature'], 1),
            'humidity': round(row['humidity'], 1),
            'light': round(row['light'], 1)
        })
    
    return jsonify(history)

@app.route('/api/forecast')
def weather_forecast():
    df = load_data()
    if df is None or df.empty or len(df) < 24:
        return jsonify({'error': 'Insufficient data for predictions'})
    
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return jsonify({'error': 'Weather model not available'})
    
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        
        for lag in [1, 3, 6]:
            df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
            df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
            df[f'light_lag_{lag}'] = df['light'].shift(lag)
        
        for window in [3, 6]:
            df[f'temp_rolling_{window}'] = df['temperature'].rolling(window=window).mean()
            df[f'humidity_rolling_{window}'] = df['humidity'].rolling(window=window).mean()
            df[f'light_rolling_{window}'] = df['light'].rolling(window=window).mean()
        
        df['temp_change'] = df['temperature'].diff()
        df['humidity_change'] = df['humidity'].diff()
        df['light_change'] = df['light'].diff()
        
        df = df.dropna()
        
        last_data = df.iloc[-1:].copy()
        
        predictions = []
        current_time = df['timestamp'].max()
        
        for hour in range(1, 25):
            pred_time = current_time + timedelta(hours=hour)
            
            next_hour = last_data.copy()
            next_hour['hour'] = pred_time.hour
            next_hour['day'] = pred_time.day
            next_hour['month'] = pred_time.month
            next_hour['dayofweek'] = pred_time.dayofweek
            
            features = next_hour.drop(['timestamp'], axis=1, errors='ignore')
            
            model_columns = [col for col in features.columns if col not in ['temperature', 'humidity', 'light']]
            
            X_scaled = scaler.transform(features[model_columns])
            
            pred_value = float(model.predict(X_scaled)[0])
            
            weather_idx = max(0, min(4, int(round(pred_value))))
            
            predictions.append({
                'time': pred_time.strftime('%Y-%m-%d %H:%M'),
                'hour': pred_time.strftime('%H:%M'),
                'condition_index': weather_idx,
                'condition': WEATHER_CONDITIONS[weather_idx]
            })
            
            last_data = next_hour
        
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/status')
def system_status():
    data_exists = os.path.exists(DATA_FILE)
    model_exists = os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)
    
    df = None
    if data_exists:
        df = pd.read_csv(DATA_FILE)
    
    status = {
        'data_available': data_exists and df is not None and not df.empty,
        'model_available': model_exists,
        'data_points': len(df) if df is not None else 0,
        'first_reading': df['timestamp'].iloc[0] if df is not None and not df.empty else None,
        'last_reading': df['timestamp'].iloc[-1] if df is not None and not df.empty else None,
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return jsonify(status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)