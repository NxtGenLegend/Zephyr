import socket
import json
import threading
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='weather_server.log')

# Server configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 12345
DATA_FILE = 'sensor_data.csv'
MODEL_FILE = 'weather_model.pkl'
SCALER_FILE = 'scaler.pkl'

# Weather conditions mapping
WEATHER_CONDITIONS = {
    0: "Clear/Sunny",
    1: "Partly Cloudy",
    2: "Cloudy",
    3: "Light Rain",
    4: "Heavy Rain"
}

class WeatherServer:
    def __init__(self):
        self.data_lock = threading.Lock()
        self.model = None
        self.scaler = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model if it exists"""
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            try:
                self.model = joblib.load(MODEL_FILE)
                self.scaler = joblib.load(SCALER_FILE)
                logging.info("Loaded existing model and scaler")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                self.model = None
                self.scaler = None
    
    def save_data(self, data_points):
        """Save received data points to CSV file"""
        with self.data_lock:
            file_exists = os.path.exists(DATA_FILE)
            
            df = pd.DataFrame(data_points)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            mode = 'a' if file_exists else 'w'
            header = not file_exists
            df.to_csv(DATA_FILE, mode=mode, header=header, index=False)
            
            logging.info(f"Saved {len(data_points)} data points to {DATA_FILE}")
    
    def create_features(self, df):
        """Create time-based features for the weather model"""
        df = df.copy()
        
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
        
        return df
    
    def assign_weather_labels(self, df):
        """Assign weather condition labels based on sensor data"""
        df = df.copy()
        
        conditions = [
            (df['light'] > 200) & (df['humidity'] < 60),
            
            (df['light'].between(80, 200)) & (df['humidity'].between(40, 70)),
            
            (df['light'] < 80) & (df['humidity'].between(60, 85)),
            
            (df['light'] < 50) & (df['humidity'].between(75, 90)),
            
            (df['light'] < 30) & (df['humidity'] > 85)
        ]
        
        choices = list(range(len(conditions)))
        df['weather_condition'] = np.select(conditions, choices, default=2)  # Default to Cloudy
        
        return df
    
    def train_model(self):
        """Train a weather prediction model using collected data"""
        if not os.path.exists(DATA_FILE):
            logging.warning("No data file exists yet, cannot train model")
            return False
        
        try:
            df = pd.read_csv(DATA_FILE)
            
            if len(df) < 24:
                logging.info(f"Only {len(df)} data points available, need at least 24 for training")
                return False
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.sort_values('timestamp')
            
            df = self.assign_weather_labels(df)
            
            df_features = self.create_features(df)
            
            feature_columns = [col for col in df_features.columns if col not in 
                              ['timestamp', 'weather_condition']]
            
            X = df_features[feature_columns]
            y = df_features['weather_condition']
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = RandomForestRegressor(
                n_estimators=50,  # Lightweight - using only 50 trees
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            
            joblib.dump(self.model, MODEL_FILE)
            joblib.dump(self.scaler, SCALER_FILE)
            
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            logging.info("Model trained successfully")
            logging.info(f"Top 5 important features: {feature_importance.head(5)}")
            
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            
            return True
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            return False
    
    def predict_weather(self, hours_ahead=6):
        """Predict weather conditions for the next N hours"""
        if not os.path.exists(DATA_FILE) or self.model is None or self.scaler is None:
            logging.warning("Cannot predict: missing data, model, or scaler")
            return None
        
        try:
            # Read the data
            df = pd.read_csv(DATA_FILE)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.sort_values('timestamp')
            
            if len(df) < 24:
                logging.warning(f"Not enough historical data for prediction ({len(df)} points)")
                return None
            
            df = self.assign_weather_labels(df)
            df_features = self.create_features(df)
            
            last_data = df_features.iloc[-1:].copy()
            
            predictions = []
            current_time = df['timestamp'].max()
            
            for hour in range(1, hours_ahead + 1):
                pred_time = current_time + timedelta(hours=hour)
                
                next_hour = last_data.copy()
                
                next_hour['hour'] = pred_time.hour
                next_hour['day'] = pred_time.day
                next_hour['month'] = pred_time.month
                next_hour['dayofweek'] = pred_time.dayofweek
                
                pred_features = next_hour.drop(['timestamp', 'weather_condition'], axis=1)
                
                pred_scaled = self.scaler.transform(pred_features)
                
                pred_value = self.model.predict(pred_scaled)[0]
                
                weather_idx = max(0, min(4, int(round(pred_value))))
                weather_condition = WEATHER_CONDITIONS[weather_idx]
                
                predictions.append({
                    'time': pred_time.strftime('%Y-%m-%d %H:%M'),
                    'condition_index': weather_idx,
                    'condition': weather_condition
                })
                
                last_data = next_hour
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return None
    
    def handle_client(self, conn, addr):
        """Handle incoming client connections"""
        logging.info(f"Connection from {addr}")
        
        try:
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                
                try:
                    json.loads(data.decode('utf-8'))
                    break
                except json.JSONDecodeError:
                    continue
            
            if data:
                sensor_data = json.loads(data.decode('utf-8'))
                logging.info(f"Received {len(sensor_data)} readings from client")
                
                self.save_data(sensor_data)
                
                model_updated = False
                if len(sensor_data) >= 10:  # Only retrain if we got significant new data
                    model_updated = self.train_model()
                
                weather_forecast = self.predict_weather(hours_ahead=12)
                
                response = {
                    'status': 'SUCCESS',
                    'message': f'Received {len(sensor_data)} readings',
                    'model_updated': model_updated
                }
                
                if weather_forecast:
                    response['forecast'] = weather_forecast[:3]  # Send back just next 3 hours
                
                conn.sendall(json.dumps(response).encode('utf-8'))
                
        except Exception as e:
            logging.error(f"Error handling client: {e}")
            conn.sendall(json.dumps({'status': 'ERROR', 'message': str(e)}).encode('utf-8'))
        finally:
            conn.close()
    
    def run(self):
        """Start the server and listen for connections"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()
            
            logging.info(f"Server started, listening on {HOST}:{PORT}")
            
            while True:
                try:
                    conn, addr = s.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(conn, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except KeyboardInterrupt:
                    logging.info("Server shutting down")
                    break
                except Exception as e:
                    logging.error(f"Error accepting connection: {e}")

if __name__ == "__main__":
    server = WeatherServer()
    server.run()