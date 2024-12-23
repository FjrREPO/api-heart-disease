# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

# Load model
try:
    model = joblib.load("./hdp_model.pkl")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise

VALIDATION_RULES = {
    'age': {
        'type': int,
        'min': 18,
        'max': 100,
        'required': True
    },
    'sex': {
        'type': int,
        'allowed_values': [0, 1],
        'required': True
    },
    'cp': {
        'type': int,
        'allowed_values': [0, 1, 2, 3],
        'required': True
    },
    'trestbps': {
        'type': int,
        'min': 90,
        'max': 200,
        'required': True
    },
    'chol': {
        'type': int,
        'min': 120,
        'max': 570,
        'required': True
    },
    'fbs': {
        'type': int,
        'allowed_values': [0, 1],
        'required': True
    },
    'restecg': {
        'type': int,
        'allowed_values': [0, 1, 2],
        'required': True
    },
    'thalach': {
        'type': int,
        'min': 60,
        'max': 220,
        'required': True
    },
    'exang': {
        'type': int,
        'allowed_values': [0, 1],
        'required': True
    },
    'oldpeak': {
        'type': int,
        'allowed_values': [0, 1, 2, 3, 4, 5],
        'required': True
    },
    'slope': {
        'type': int,
        'allowed_values': [0, 1, 2],
        'required': True
    },
    'ca': {
        'type': int,
        'allowed_values': [0, 1, 2, 3],
        'required': True
    },
    'thal': {
        'type': int,
        'allowed_values': [0, 1, 2, 3],
        'required': True
    }
}

def validate_data(data):
    errors = []
    
    for field, rules in VALIDATION_RULES.items():
        if field not in data:
            if rules['required']:
                errors.append(f"Missing required field: {field}")
            continue
            
        value = data[field]
        
        # Type validation
        try:
            value = rules['type'](value)
        except (ValueError, TypeError):
            errors.append(f"{field} must be of type {rules['type'].__name__}")
            continue
            
        # Range validation
        if 'min' in rules and value < rules['min']:
            errors.append(f"{field} must be at least {rules['min']}")
        if 'max' in rules and value > rules['max']:
            errors.append(f"{field} must be no more than {rules['max']}")
            
        # Allowed values validation
        if 'allowed_values' in rules and value not in rules['allowed_values']:
            errors.append(f"{field} must be one of {rules['allowed_values']}")
    
    return errors

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        logging.info(f"Received prediction request: {data}")
        
        # Validate input
        errors = validate_data(data)
        if errors:
            return jsonify({
                "success": False,
                "errors": errors
            }), 400
        
        # Prepare input data
        values = np.array([[
            int(data['age']),
            int(data['sex']),
            int(data['cp']),
            int(data['trestbps']),
            int(data['chol']),
            int(data['fbs']),
            int(data['restecg']),
            int(data['thalach']),
            int(data['exang']),
            int(data['oldpeak']),
            int(data['slope']),
            int(data['ca']),
            int(data['thal'])
        ]])
        
        # Make prediction
        prediction = model.predict(values)
        prediction_proba = model.predict_proba(values)[0]
        
        return jsonify({
            "success": True,
            "prediction": int(prediction[0]),
            "probability": {
                "negative": float(prediction_proba[0]),
                "positive": float(prediction_proba[1])
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)