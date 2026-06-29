from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialize the Flask app
app = Flask(__name__)

# Define the path to the trained model and scaler files
MODEL_PATH = 'logistic_regression_model.joblib'
SCALER_PATH = 'scaler.joblib'

# Check if the model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure it's in the same directory as app.py.")

# Check if the scaler file exists before loading
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}. Please ensure it's in the same directory as app.py.")

# Load the trained model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Assuming you have the feature names from your training process.
# We'll use the column names from the original X DataFrame.
feature_names = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

@app.route('/')
def home():
    return '<h1>Diabetes Prediction API</h1><p>Send a POST request to /predict with JSON data for prediction.</p>'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Convert input data to a Pandas DataFrame
        input_df = pd.DataFrame([data], columns=feature_names)

        # Scale the input data using the loaded scaler
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

        # Make prediction
        prediction = model.predict(input_scaled_df)
        prediction_proba = model.predict_proba(input_scaled_df)

        # Return the prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'prediction_probability_class_0': prediction_proba[0][0],
            'prediction_probability_class_1': prediction_proba[0][1]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print(f"* Running Flask app on http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)
