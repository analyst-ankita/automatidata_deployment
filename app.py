from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = joblib.load('lr.pkl')

# Load the scaler object
scaler = joblib.load('scaler.pkl')

# Define route to render the form
@app.route('/')
def index():
    return render_template('index.html')

# Define route to handle form submission and predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    passenger_count = float(request.form['passenger_count'])
    mean_distance = float(request.form['mean_distance'])
    mean_duration = float(request.form['mean_duration'])
    rush_hour = int(request.form['rush_hour'])
    vendor_id = int(request.form['vendor_id'])

    # One-hot encode VendorID
    vendor_id_encoded = np.zeros(2)  # Assuming there are 2 categories for VendorID
    vendor_id_encoded[vendor_id - 1] = 1  # Adjust index to start from 0

    # Concatenate input features
    input_features = np.array([[passenger_count, mean_distance, mean_duration, rush_hour] + vendor_id_encoded.tolist()])

    # Scale numerical features
    input_features_scaled = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(input_features_scaled)

    # Extract and round the fare amount
    fare_amount = round(float(prediction[0]), 2)

    # Render the result
    return render_template('result.html', fare_amount=fare_amount)

if __name__ == '__main__':
    app.run(debug=False, host = '0.0.0.0', port= 8080)
