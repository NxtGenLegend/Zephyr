import time
import socket
import json
import numpy as np
from datetime import datetime
import logging

# For Raspberry Pi sensors
try:
    import board
    import adafruit_dht
    import adafruit_tsl2591  # Light sensor
    SIMULATION = False
except ImportError:
    print("Adafruit libraries not found, running in simulation mode")
    SIMULATION = True

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='rpi_sensor_client.log')

SERVER_HOST = '192.168.1.100'  # Replace with your VM's IP address
SERVER_PORT = 12345
SEND_INTERVAL = 60  
BATCH_SIZE = 10  

class SensorClient:
    def __init__(self):
        self.data_buffer = []
        self.setup_sensors()
        
    def setup_sensors(self):
        """Initialize the sensors"""
        if not SIMULATION:
            # Setup the DHT22 temp/humidity sensor
            self.dht_sensor = adafruit_dht.DHT22(board.D4)
            
            # Setup the TSL2591 light sensor
            i2c = board.I2C()
            self.light_sensor = adafruit_tsl2591.TSL2591(i2c)
            
            logging.info("Hardware sensors initialized")
        else:
            logging.info("Using simulated sensors")
    
    def read_sensors(self):
        """Read data from all sensors"""
        if not SIMULATION:
            try:
                temperature = self.dht_sensor.temperature
                humidity = self.dht_sensor.humidity
                light = self.light_sensor.lux
            except Exception as e:
                logging.error(f"Error reading sensors: {e}")
                return None
        else:
            temperature = 20 + np.random.normal(0, 3)  # °C
            humidity = 60 + np.random.normal(0, 10)    # %
            light = max(0, 200 + np.random.normal(0, 100))  # lux
            
            if light > 300:  # Sunny
                temperature += 2
                humidity -= 5
            elif light < 50:  # Dark/cloudy
                temperature -= 1
                humidity += 5
        
        temperature = max(-10, min(40, temperature))
        humidity = max(0, min(100, humidity))
        light = max(0, light)
        
        timestamp = datetime.now().isoformat()
        
        reading = {
            'timestamp': timestamp,
            'temperature': float(temperature),
            'humidity': float(humidity),
            'light': float(light)
        }
        
        logging.info(f"Sensor reading: Temp={temperature:.1f}°C, Humidity={humidity:.1f}%, Light={light:.1f} lux")
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
        logging.info("Starting sensor data collection")
        
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