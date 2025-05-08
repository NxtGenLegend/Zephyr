# Zephyr

Weather Station Using GrovePi Sensors
====================================

Team Members:
- Adhish Chakravorty
- Yash Gupta

Instructions:
------------

1. Hardware Setup:
   - Connect GrovePi Temperature & Humidity sensor (v1.2) to port D4
   - Connect GrovePi Light sensor (v1.2) to port A0
   - Mount GrovePi Shield onto RaspberryPi
   - Ensure Raspberry Pi is powered and connected to the same network as your computer

2. Installation:
   - Install required libraries on Raspberry Pi:
     $ sudo curl -kL dexterindustries.com/update_grovepi | bash
     $ sudo pip3 install numpy

   - Install required libraries on your computer:
     $ pip3 install flask pandas numpy scikit-learn joblib matplotlib

3. Configuration:
   - Update SERVER_HOST in client.py with your computer's IP address
   - Ensure ports 12345 (TCP server) and 5000 (Flask app) are accessible

4. Execution (in this order):
   - On your computer, start the TCP server:
     $ python3 server.py

   - On your computer, start the Flask web app:
     $ python3 app.py

   - On Raspberry Pi, start the client:
     $ sudo python3 client.py

5. Access the dashboard:
   - Open a web browser and go to: http://localhost:5000

External Libraries:
-----------------
- grovepi: Interface with GrovePi sensors
- numpy: Numerical computing for data processing
- pandas: Data manipulation and analysis
- scikit-learn: Machine learning (RandomForestRegressor) for weather prediction
- joblib: Model serialization
- flask: Web server framework
- matplotlib: Visualization for model analysis