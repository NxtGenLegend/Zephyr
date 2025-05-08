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

SERVER_HOST = '192.168.130.103'  # Mac IP Address
SERVER_PORT = 12345
SEND_INTERVAL = 2  # seconds between readings

DHT_SENSOR_PIN = 4      # DHT to D4
LIGHT_SENSOR_PIN = 0    # Light sensor to A0

def read_sensors():
    """Read data from GrovePi sensors"""
    if not SIMULATION:
        try:
            grovepi.pinMode(LIGHT_SENSOR_PIN, "INPUT")
            
            [temperature, humidity] = dht(DHT_SENSOR_PIN, 0)
            
            attempts = 0
            while (temperature == 0 and humidity == 0) and attempts < 3:
                time.sleep(1)
                [temperature, humidity] = dht(DHT_SENSOR_PIN, 0)
                attempts += 1
            
            light = grovepi.analogRead(LIGHT_SENSOR_PIN)
            light_lux = light * 5.0 / 1023.0 * 800
            
        except Exception as e:
            logging.error(f"Error reading GrovePi sensors: {e}")
            return None
    else:
        temperature = 20 + np.random.normal(0, 3)  # °C
        humidity = 60 + np.random.normal(0, 10)    # %
        light_lux = max(0, 200 + np.random.normal(0, 100))  # lux
    
    reading = {
        'timestamp': datetime.now().isoformat(),
        'temperature': float(temperature),
        'humidity': float(humidity),
        'light': float(light_lux)
    }
    
    logging.info(f"Reading: Temp={temperature:.1f}°C, Humidity={humidity:.1f}%, Light={light_lux:.1f} lux")
    return reading

def send_data(reading):
    """Send data to the server"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((SERVER_HOST, SERVER_PORT))
            data_json = json.dumps(reading)
            s.sendall(data_json.encode('utf-8'))
            
            s.settimeout(5)
            try:
                ack = s.recv(1024).decode('utf-8')
                logging.info(f"Server response: {ack}")
            except socket.timeout:
                logging.error("Timeout waiting for server acknowledgment")
            
    except Exception as e:
        logging.error(f"Error sending data: {e}")

def main():
    """Main loop to collect and send data"""
    logging.info("Starting GrovePi sensor data collection")
    
    while True:
        try:
            reading = read_sensors()
            
            if reading:
                send_data(reading)
                print("Sending data") 
            time.sleep(SEND_INTERVAL)
            
        except KeyboardInterrupt:
            logging.info("Stopping data collection")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            time.sleep(10)

if _name_ == "_main_":
    main()