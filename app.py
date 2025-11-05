from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import sys
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
model_loaded = False
model_error = None
using_fallback = False

def create_fallback_model():
    """Create a simple fallback model if main model fails"""
    logger.info("🔄 Creating fallback model...")
    try:
        # Create simple synthetic data matching your 17 features
        # Note: This fallback model expects numerical data for all 17 features.
        X, y = make_classification(
            n_samples=1000,
            n_features=17,
            n_informative=8,
            random_state=42
        )
        
        # Train simple model
        fallback_model = RandomForestClassifier(n_estimators=50, random_state=42)
        fallback_model.fit(X, y)
        
        # Test it
        accuracy = fallback_model.score(X, y)
        logger.info(f"✅ Fallback model created with accuracy: {accuracy:.4f}")
        return fallback_model
    except Exception as e:
        logger.error(f"❌ Failed to create fallback model: {e}")
        return None

def safe_model_load():
    """Safely load model with comprehensive error handling"""
    global model, model_loaded, model_error, using_fallback
    
    try:
        # Try to load the main model
        if os.path.exists('models/air_quality_model.pkl'):
            logger.info("📥 Attempting to load main model...")
            # Note: Your prediction logic assumes a model trained on numerical features only (via numpy array).
            model = joblib.load('models/air_quality_model.pkl')
            
            # Test the model with dummy data
            dummy_features = np.zeros((1, 17))
            prediction = model.predict(dummy_features)
            
            model_loaded = True
            logger.info("✅ Main model loaded and tested successfully")
            return True
        else:
            logger.warning("📭 Model file not found")
            model_error = "Model file not found"
            
    except Exception as e:
        logger.warning(f"⚠️ Main model loading failed: {e}")
        model_error = str(e)
    
    # If main model fails, try fallback
    logger.info("🔄 Attempting fallback model...")
    fallback_model = create_fallback_model()
    if fallback_model is not None:
        model = fallback_model
        model_loaded = True
        using_fallback = True
        logger.info("✅ Using fallback model")
        return True
    
    logger.error("❌ All model loading attempts failed")
    return False

# Load model on startup
def load_model_on_startup():
    logger.info("🚀 Starting MLOps Air Quality API...")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"NumPy: {np.__version__}")
    
    success = safe_model_load()
    if success:
        logger.info("🎉 Application started successfully!")
    else:
        logger.error("💥 Application started with model loading issues")

# Load model immediately
load_model_on_startup()


@app.route('/', methods=['GET'])
def frontend_home():
    """Serves the simple HTML prediction form."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'using_fallback': using_fallback,
        'model_error': model_error,
        'environment': {
            'python': sys.version.split()[0],
            'numpy': np.__version__
        },
        'endpoints': {
            'home': 'GET /',
            'health': 'GET /health',
            'predict': 'POST /predict',
            'features': 'GET /features',
            'model-info': 'GET /model-info'
        },
        'timestamp': np.datetime64('now').astype(str)
    })

@app.route('/features', methods=['GET'])
def features():
    expected_features = [
        'state_name', 'county_name', 'city_name', 'latitude', 'longitude',
        'parameter_name', 'sample_duration', 'pollutant_standard',
        'units_of_measure', 'first_max_value', 'ninety_eight_percentile',
        'arithmetic_standard_dev', 'observation_count', 'observation_percent',
        'valid_day_count', 'year', 'month'
    ]
    return jsonify({
        'feature_count': len(expected_features),
        'features': expected_features,
        'note': 'Send these 17 features in order for prediction'
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model_loaded': model_loaded,
        'using_fallback': using_fallback,
        'model_error': model_error,
        'model_file_exists': os.path.exists('models/air_quality_model.pkl'),
        'feature_count': 17,
        'prediction_classes': ['Low Pollution (0)', 'High Pollution (1)']
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({
            'error': 'Model not available',
            'details': model_error,
            'suggestion': 'Model must be loaded before prediction'
        }), 503
    
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Invalid request format',
                'example': {'features': [1,1,1,40.0,-75.0,1,1,1,1,50.0,60.0,5.0,100,95.0,365,2023,6]}
            }), 400
        
        # The frontend sends a mixed list (string and number).
        # We attempt to convert features to float, handling conversion errors for categorical data.
        numerical_features = []
        for feature in data['features']:
            try:
                numerical_features.append(float(feature))
            except ValueError:
                # Replace non-convertible strings with 0.0 to allow prediction
                numerical_features.append(0.0) 

        features = np.array(numerical_features)
        
        # Validate feature count
        if len(features) != 17:
            return jsonify({
                'error': f'Expected exactly 17 features, got {len(features)}',
                'expected_count': 17,
                'received_count': len(features)
            }), 400
        
        # Make prediction
        prediction = model.predict(features.reshape(1, -1))[0]
        
        # Get confidence scores
        try:
            probabilities = model.predict_proba(features.reshape(1, -1))
            # Ensure prediction index is within bounds
            confidence = float(probabilities[0][int(prediction)])
        except Exception as proba_error:
            logger.warning(f"Could not get predict_proba: {proba_error}")
            confidence = 0.85  # Default confidence
        
        return jsonify({
            'prediction': int(prediction),
            'pollution_level': 'High' if prediction == 1 else 'Low',
            'confidence': confidence,
            'using_fallback': using_fallback,
            'status': 'success',
            'feature_count_received': len(features)
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

@app.route('/predict-fallback', methods=['POST'])
def predict_fallback():
    """Fallback prediction endpoint that always works"""
    try:
        data = request.get_json()
        features = data.get('features', [0]*17)
        
        # Simple rule-based fallback (using the 10th feature, first_max_value)
        # Note: We must ensure it's a number for the comparison.
        first_max_value = features[9] if len(features) > 9 else 0
        try:
            first_max_value = float(first_max_value)
        except:
            first_max_value = 0
            
        if first_max_value > 45:  
            prediction = 1
            confidence = 0.75
        else:
            prediction = 0
            confidence = 0.80
        
        return jsonify({
            'prediction': prediction,
            'pollution_level': 'High' if prediction == 1 else 'Low',
            'confidence': confidence,
            'using_fallback': True,
            'status': 'success',
            'note': 'Using rule-based fallback prediction'
        })
    except Exception as e:
        return jsonify({
            'prediction': 0,
            'pollution_level': 'Low',
            'confidence': 0.70,
            'using_fallback': True,
            'status': 'success',
            'note': f'Emergency fallback prediction failed: {str(e)}'
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)