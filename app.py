from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = "logistic_regression_model.joblib"
SCALER_PATH = "scaler.joblib"

# Check if files exist
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.isfile(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature names (must match training order)
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


@app.route("/")
def home():
    return jsonify({
        "message": "Diabetes Prediction API is running.",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "No JSON data received."
            }), 400

        # Check for missing features
        missing = [feature for feature in FEATURE_NAMES if feature not in data]

        if missing:
            return jsonify({
                "error": f"Missing fields: {missing}"
            }), 400

        # Create DataFrame
        input_df = pd.DataFrame([[data[feature] for feature in FEATURE_NAMES]],
                                columns=FEATURE_NAMES)

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        return jsonify({
            "prediction": int(prediction),
            "prediction_probability_class_0": float(probability[0]),
            "prediction_probability_class_1": float(probability[1])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
