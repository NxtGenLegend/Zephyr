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
    """Load data from CSV file with robust error handling"""
    if not os.path.exists(DATA_FILE):
        print(f"File '{DATA_FILE}' does not exist")
        pd.DataFrame(columns=['timestamp', 'temperature', 'humidity', 'light']).to_csv(DATA_FILE, index=False)
        return None
    
    try:
        if os.path.getsize(DATA_FILE) < 50:  # Likely only has headers, too small a file size, kind of a weird edge case
            print(f"File exists but appears to have only headers (size: {os.path.getsize(DATA_FILE)} bytes)")

        df = pd.read_csv(DATA_FILE)

        if df.empty:
            print("DataFrame is empty (has only headers)")
            return None
            
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            print(f"Error parsing timestamps: {e}")
            backup_file = f"{DATA_FILE}.bak.{int(time.time())}"
            os.rename(DATA_FILE, backup_file)
            print(f"Backed up corrupted file to {backup_file}")
            
            pd.DataFrame(columns=['timestamp', 'temperature', 'humidity', 'light']).to_csv(DATA_FILE, index=False)
            return None

        for col in ['temperature', 'humidity', 'light']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    print(f"Warning: NaN values found in {col} column, replacing with defaults")
                    if col == 'temperature':
                        df[col].fillna(20.0, inplace=True)
                    elif col == 'humidity':
                        df[col].fillna(50.0, inplace=True)
                    elif col == 'light':
                        df[col].fillna(100.0, inplace=True)
        
        df = df.sort_values('timestamp')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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
       
        if hasattr(model, 'feature_names_in_'):
            required_features = model.feature_names_in_
            print(f"Model requires these features: {required_features}")
        
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        
        for lag in [1, 2, 3, 6, 12]:
            df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
            df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
            df[f'light_lag_{lag}'] = df['light'].shift(lag)
        
        for window in [3, 6, 12]:
            df[f'temp_rolling_{window}'] = df['temperature'].rolling(window=window).mean()
            df[f'humidity_rolling_{window}'] = df['humidity'].rolling(window=window).mean()
            df[f'light_rolling_{window}'] = df['light'].rolling(window=window).mean()
        
        df['temp_change'] = df['temperature'].diff()
        df['humidity_change'] = df['humidity'].diff()
        df['light_change'] = df['light'].diff()
       
        df = df.dropna()
        
        if len(df) < 12:  
            return jsonify({'error': 'After creating lag features, not enough complete data rows remain'})
        
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
            
            if hasattr(model, 'feature_names_in_'):
                missing_cols = set(model.feature_names_in_) - set(features.columns)
                if missing_cols:
                    print(f"Missing columns for prediction: {missing_cols}")
                    for col in missing_cols:
                        features[col] = 0.0 

            if hasattr(model, 'feature_names_in_'):
                features = features[model.feature_names_in_]
            
            X_scaled = scaler.transform(features)

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
        import traceback
        print(traceback.format_exc())
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
    app.run(host='0.0.0.0', port=5555, debug=True)