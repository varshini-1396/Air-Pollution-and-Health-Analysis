from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained LightGBM model (trained without HealthImpactClass)
model = joblib.load('models/health_impact_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        AQI = float(request.form['AQI'])
        PM10 = float(request.form['PM10'])
        PM25 = float(request.form['PM25'])
        NO2 = float(request.form['NO2'])
        SO2 = float(request.form['SO2'])
        O3 = float(request.form['O3'])
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        WindSpeed = float(request.form['WindSpeed'])
        RespiratoryCases = float(request.form['RespiratoryCases'])
        CardiovascularCases = float(request.form['CardiovascularCases'])
        HospitalAdmissions = float(request.form['HospitalAdmissions'])

        # Prepare the input feature array (12 features)
        features = np.array([[AQI, PM10, PM25, NO2, SO2, O3,
                              Temperature, Humidity, WindSpeed,
                              RespiratoryCases, CardiovascularCases, HospitalAdmissions]])

        # Make prediction
        prediction = model.predict(features)[0]
        prediction = np.clip(prediction, 0, 100)  # Optional: cap score to 0–100

        return render_template('index.html', prediction_text=f"{prediction:.2f}")

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
