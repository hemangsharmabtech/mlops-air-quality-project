from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model
def load_model():
    try:
        model = joblib.load('models/air_quality_model.pkl')
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

model = load_model()

@app.route('/')
def home():
    return """
    <h1>Air Quality Index Prediction API</h1>
    <p>Use /predict endpoint with POST request to get pollution level predictions</p>
    <p>Send JSON data with 'features' key containing all feature values in order</p>
    <p>Feature order: state_name, county_name, city_name, latitude, longitude, parameter_name, sample_duration, pollutant_standard, units_of_measure, first_max_value, ninety_eight_percentile, arithmetic_standard_dev, observation_count, observation_percent, valid_day_count, year, month</p>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.get_json()
        
        # Get features from request
        features = np.array(data['features']).reshape(1, -1)
        
        # Make prediction (0 = Low pollution, 1 = High pollution)
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)
        
        pollution_level = "High" if prediction[0] == 1 else "Low"
        confidence = float(prediction_proba[0][prediction[0]])
        
        return jsonify({
            'prediction': int(prediction[0]),
            'pollution_level': pollution_level,
            'confidence': confidence,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'endpoints': {
            'home': 'GET /',
            'health': 'GET /health',
            'predict': 'POST /predict'
        }
    })

@app.route('/features', methods=['GET'])
def features():
    """Return expected feature names and order"""
    expected_features = [
        'state_name', 'county_name', 'city_name', 'latitude', 'longitude',
        'parameter_name', 'sample_duration', 'pollutant_standard',
        'units_of_measure', 'first_max_value', 'ninety_eight_percentile',
        'arithmetic_standard_dev', 'observation_count', 'observation_percent',
        'valid_day_count', 'year', 'month'
    ]
    return jsonify({
        'feature_count': len(expected_features),
        'features': expected_features
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)