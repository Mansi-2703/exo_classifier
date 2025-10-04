import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np

# --- 1. Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- 2. Feature Engineering Function (Must be identical to trainer) ---
def engineer_features(df):
    print("🔬 Engineering new features for prediction data...")
    df['luminosity_proxy'] = np.log1p((df.get('stellar_radius_solar_radii', 0)**2) * (df.get('stellar_effective_temperature_k', 0)**4))
    df['habitable_zone_proxy'] = df.get('insolation_flux_earth_flux', 0) / (df.get('stellar_effective_temperature_k', 0) + 1e-6)
    df['size_ratio'] = df.get('planetary_radius_earth_radii', 0) / (df.get('stellar_radius_solar_radii', 0) + 1e-6)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    print("✅ New features created for prediction.")
    return df

# --- 3. Load the Trained Model and Define Paths ---
MODEL_PATH = 'models/random_forest_model.pkl'
SAMPLE_DATA_PATH = os.path.join('data', 'final_dataset_exoplanet.csv')
model = None
REQUIRED_FEATURES_BEFORE_SELECTION = []

print("🔄 Loading the trained model pipeline...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    REQUIRED_FEATURES_BEFORE_SELECTION = model.named_steps['scaler'].get_feature_names_out().tolist()
    print("✅ Model pipeline loaded successfully!")
    print(f"🧠 Model expects these features initially: {REQUIRED_FEATURES_BEFORE_SELECTION}")
except FileNotFoundError:
    print(f"❌ ERROR: Model file not found at '{MODEL_PATH}'. Please run the new trainer script first.")
except Exception as e:
    print(f"❌ ERROR: An unexpected error occurred while loading the model: {e}")

# --- Utility function to process data and predict ---
def make_prediction(df):
    print("\n--- Prediction Pipeline Started ---")
    original_df = df.copy()
    df_engineered = engineer_features(df.copy())
    df_for_prediction = pd.DataFrame()

    for col in REQUIRED_FEATURES_BEFORE_SELECTION:
        if col in df_engineered.columns:
            df_for_prediction[col] = df_engineered[col]
        else:
            df_for_prediction[col] = 0
            
    df_for_prediction = df_for_prediction[REQUIRED_FEATURES_BEFORE_SELECTION]
    df_for_prediction = df_for_prediction.fillna(0)
    
    print(f"🤖 Making predictions on {len(df_for_prediction)} rows...")
    predictions = model.predict(df_for_prediction)
    
    original_df['disposition'] = predictions
    print("✅ Predictions complete.")
    return original_df.to_json(orient='records')

# --- Prediction Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None: return jsonify({"error": "Model is not loaded."}), 500
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    try:
        input_df = pd.read_csv(file)
        result_json = make_prediction(input_df)
        return jsonify(result_json)
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

@app.route('/predict_sample', methods=['GET'])
def predict_sample():
    if model is None: return jsonify({"error": "Model is not loaded."}), 500
    try:
        if not os.path.exists(SAMPLE_DATA_PATH): return jsonify({"error": "Sample data file not found."}), 404
        input_df = pd.read_csv(SAMPLE_DATA_PATH)
        result_json = make_prediction(input_df)
        return jsonify(result_json)
    except Exception as e:
        return jsonify({"error": f"An error occurred during sample prediction: {str(e)}"}), 500

@app.route('/sample_data', methods=['GET'])
def view_sample_data():
    try:
        if not os.path.exists(SAMPLE_DATA_PATH): return jsonify({"error": "Sample data file not found"}), 404
        df = pd.read_csv(SAMPLE_DATA_PATH)
        return df.head(10).to_json(orient='split')
    except Exception as e:
        return jsonify({"error": f"An error occurred during sample data viewing: {str(e)}"}), 500

# --- Run the App ---
if __name__ == '__main__':
    print("🚀 Flask server is starting...")
    app.run(debug=True, port=5000)

