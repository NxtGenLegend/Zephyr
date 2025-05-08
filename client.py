import time
import socket
import json
import numpy as np
from datetime import datetime
import logging
import sys

try:
    import grovepi
    from grovepi import dht
    SIMULATION = False
except ImportError:
    print("GrovePi library not found, running in simulation mode")
    SIMULATION = True

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='rpi_sensor_client.log')

SERVER_HOST = '192.168.1.100'  # Replace with your VM's IP address
SERVER_PORT = 12345
SEND_INTERVAL = 60  
BATCH_SIZE = 10  

DHT_SENSOR_PIN = 4      
LIGHT_SENSOR_PIN = 0    

class SensorClient:
    def __init__(self):
        self.data_buffer = []
        self.setup_sensors()
        
    def setup_sensors(self):
        """Initialize the GrovePi sensors"""
        if not SIMULATION:
            try:
                grovepi.pinMode(LIGHT_SENSOR_PIN, "INPUT")
                
                # For DHT sensor, no specific pinMode needed
                # Optional: Set I2C address if needed
                # grovepi.set_bus("RPI_1")
                
                logging.info("GrovePi sensors initialized")
            except Exception as e:
                logging.error(f"Error initializing GrovePi: {e}")
                if "Permission denied" in str(e):
                    logging.error("Permission error. Try running with sudo")
                    sys.exit(1)
        else:
            logging.info("Using simulated sensors")
    
    def read_sensors(self):
        """Read data from GrovePi sensors"""
        if not SIMULATION:
            try:
                # Read from DHT sensor (temperature & humidity)
                # DHT spec is [temp, hum] = dht(pin, dht_type)
                # dht_type: 0 for blue DHT, 1 for white DHT (DHT Pro)
                [temperature, humidity] = dht(DHT_SENSOR_PIN, 0)  # 0 for DHT11/DHT22
                
                attempts = 0
                while (temperature == 0 and humidity == 0) and attempts < 3:
                    time.sleep(1)
                    [temperature, humidity] = dht(DHT_SENSOR_PIN, 0)
                    attempts += 1
                
                light = grovepi.analogRead(LIGHT_SENSOR_PIN)
                
                # Convert raw light reading (0-1023) to lux (approximate)
                light_voltage = light * 5.0 / 1023.0
                light_lux = light_voltage * 800  # Approximate conversion for Grove Light Sensor
                
            except Exception as e:
                logging.error(f"Error reading GrovePi sensors: {e}")
                return None
        else:
            temperature = 20 + np.random.normal(0, 3)  # °C
            humidity = 60 + np.random.normal(0, 10)    # %
            light_lux = max(0, 200 + np.random.normal(0, 100))  # lux
            
            if light_lux > 300:  # Sunny
                temperature += 2
                humidity -= 5
            elif light_lux < 50:  # Dark/cloudy
                temperature -= 1
                humidity += 5
        
        temperature = max(-10, min(40, temperature))
        humidity = max(0, min(100, humidity))
        light_lux = max(0, light_lux)
        
        timestamp = datetime.now().isoformat()
        
        reading = {
            'timestamp': timestamp,
            'temperature': float(temperature),
            'humidity': float(humidity),
            'light': float(light_lux)
        }
        
        logging.info(f"Sensor reading: Temp={temperature:.1f}°C, Humidity={humidity:.1f}%, Light={light_lux:.1f} lux")
        return reading
    
    def send_data(self):
        """Send buffered data to the server"""
        if not self.data_buffer:
            return
            
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((SERVER_HOST, SERVER_PORT))
                
                data_json = json.dumps(self.data_buffer)
                s.sendall(data_json.encode('utf-8'))
                
                ack = s.recv(1024).decode('utf-8')
                
                if "SUCCESS" in ack:
                    logging.info(f"Successfully sent {len(self.data_buffer)} readings to server")
                    self.data_buffer = []  
                else:
                    logging.error(f"Server returned error: {ack}")
        
        except Exception as e:
            logging.error(f"Error sending data to server: {e}")
    
    def run(self):
        """Main loop to collect and send data"""
        logging.info("Starting GrovePi sensor data collection")
        
        while True:
            try:
                reading = self.read_sensors()
                if reading:
                    self.data_buffer.append(reading)
                
                if len(self.data_buffer) >= BATCH_SIZE:
                    self.send_data()
                    
                time.sleep(SEND_INTERVAL)
                
            except KeyboardInterrupt:
                logging.info("Stopping data collection")
                if self.data_buffer:
                    self.send_data()
                break
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                time.sleep(10)  

if __name__ == "__main__":
    client = SensorClient()
    client.run()