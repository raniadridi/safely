from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load("predictFraud.pkl")
    MODEL_LOADED = True
    
    try:
        scaler = joblib.load("scaler.pkl")
        USE_SCALER = True
    except:
        scaler = None
        USE_SCALER = False
        
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_LOADED = False
    model = None
    scaler = None
    USE_SCALER = False

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/form')
def show_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return render_template('result.html', 
                             prediction=['Model not available'],
                             confidence='N/A',
                             fraud_probability=0,
                             details={})
    
    try:
        # Get form data
        transaction_type = int(request.form['type'])
        amount = float(request.form['amount'])
        old_balance = float(request.form['oldbalanceOrg'])
        new_balance = float(request.form['newbalanceOrig'])
        
        # Prepare features
        input_features = [transaction_type, amount, old_balance, new_balance]
        
        # Apply scaling if needed
        if USE_SCALER and scaler is not None:
            input_features_scaled = scaler.transform([input_features])
            prediction = model.predict(input_features_scaled)[0]
            probabilities = model.predict_proba(input_features_scaled)[0]
        else:
            prediction = model.predict([input_features])[0]
            probabilities = model.predict_proba([input_features])[0]
        
        fraud_probability = probabilities[1]
        no_fraud_probability = probabilities[0]
        
        # Determine confidence
        max_prob = max(probabilities)
        if max_prob >= 0.8:
            confidence = "High"
        elif max_prob >= 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Format result
        result = ["Fraud"] if prediction == 1 else ["No Fraud"]
        
        transaction_types = {
            1: "CASH_OUT", 2: "PAYMENT", 3: "CASH_IN", 
            4: "TRANSFER", 5: "DEBIT"
        }
        
        details = {
            'transaction_type': transaction_types.get(transaction_type, 'Unknown'),
            'amount': f"${amount:,.2f}",
            'old_balance': f"${old_balance:,.2f}",
            'new_balance': f"${new_balance:,.2f}",
            'fraud_prob_percent': round(fraud_probability * 100, 1),
            'safe_prob_percent': round(no_fraud_probability * 100, 1)
        }
        
        return render_template('result.html', 
                             prediction=result,
                             confidence=confidence,
                             fraud_probability=round(fraud_probability * 100, 1),
                             details=details)
        
    except ValueError:
        return render_template('result.html', 
                             prediction=['Invalid input data'],
                             confidence='N/A',
                             fraud_probability=0,
                             details={})
    except Exception as e:
        return render_template('result.html', 
                             prediction=[f'Error: {str(e)}'],
                             confidence='N/A',
                             fraud_probability=0,
                             details={})

@app.route('/visualizations')
def visualizations():
    return render_template('visualization.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not available"}), 500
    
    try:
        data = request.get_json()
        
        # Handle input formats
        if isinstance(data, list) and len(data) == 4:
            features = [float(x) for x in data]
        else:
            features = [
                float(data.get('type', 1)),
                float(data.get('amount', 0)),
                float(data.get('oldbalanceOrg', 0)),
                float(data.get('newbalanceOrig', 0))
            ]
        
        # Apply scaling if needed
        if USE_SCALER and scaler is not None:
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
        else:
            prediction = model.predict([features])[0]
            probabilities = model.predict_proba([features])[0]
        
        # Calculate confidence
        max_prob = max(probabilities)
        confidence = "High" if max_prob >= 0.8 else "Medium" if max_prob >= 0.6 else "Low"
        
        result = "Fraud" if prediction == 1 else "No Fraud"
        
        return jsonify({
            "prediction": result,
            "numeric_prediction": int(prediction),
            "fraud_probability": round(probabilities[1], 4),
            "confidence": confidence,
            "features_used": features
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test')
def test():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"})
    
    # Test cases
    test_cases = [
        {"name": "Large Cash Out", "features": [1, 85000.0, 90000.0, 5000.0]},
        {"name": "Normal Payment", "features": [2, 1000.0, 5000.0, 4000.0]},
        {"name": "Account Drain", "features": [1, 45000.0, 45000.0, 0.0]},
        {"name": "Cash Deposit", "features": [3, 2000.0, 1000.0, 3000.0]}
    ]
    
    results = []
    for case in test_cases:
        try:
            features = case["features"]
            if USE_SCALER and scaler is not None:
                features = scaler.transform([features])
            
            pred = model.predict([features])[0] if not USE_SCALER else model.predict(features)[0]
            prob = model.predict_proba([features])[0] if not USE_SCALER else model.predict_proba(features)[0]
            
            result = "Fraud" if pred == 1 else "No Fraud"
            
            results.append({
                "name": case["name"],
                "prediction": result,
                "fraud_probability": round(prob[1], 3)
            })
        except Exception as e:
            results.append({
                "name": case["name"],
                "error": str(e)
            })
    
    return jsonify({
        "model_loaded": MODEL_LOADED,
        "test_results": results
    })

if __name__ == '__main__':
    import os
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host='0.0.0.0', port=port, debug=debug)