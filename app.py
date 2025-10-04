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

# --- 3. Habitability Classification ---
def classify_stellar_type(stellar_temp):
    """Classify star based on temperature"""
    if 6000 <= stellar_temp <= 7200:
        return "F"
    elif 5300 <= stellar_temp < 6000:
        return "G"
    elif 3700 <= stellar_temp < 5300:
        return "K"
    elif 2600 <= stellar_temp < 3700:
        return "M"
    else:
        return None

def is_habitable(stellar_temp, stellar_radius, planet_radius, planet_temp, insolation_flux):
    """Check if a planet meets habitability criteria"""
    # Stellar classification ranges
    stellar_ranges = {
        "F": {"temp": (6000, 7200), "radius": (1.1, 1.7)},
        "G": {"temp": (5300, 6000), "radius": (0.8, 1.2)},
        "K": {"temp": (3700, 5300), "radius": (0.6, 0.9)},
        "M": {"temp": (2600, 3700), "radius": (0.1, 0.6)},
    }
    
    # Habitable zone conditions
    planet_radius_range = (0.5, 5.0)    # in Earth radii
    planet_temp_range   = (175, 270)    # in Kelvin
    flux_range          = (0.32, 1.77)  # in Earth flux units
    
    # Classify stellar type
    stellar_type = classify_stellar_type(stellar_temp)
    if stellar_type not in stellar_ranges:
        return False, "Invalid stellar type"
    
    s_range = stellar_ranges[stellar_type]
    
    # Check conditions
    stellar_ok = (s_range["temp"][0] <= stellar_temp <= s_range["temp"][1] and 
                  s_range["radius"][0] <= stellar_radius <= s_range["radius"][1])
    
    planet_ok = (planet_radius_range[0] <= planet_radius <= planet_radius_range[1] and 
                 planet_temp_range[0] <= planet_temp <= planet_temp_range[1] and 
                 flux_range[0] <= insolation_flux <= flux_range[1])
    
    if stellar_ok and planet_ok:
        return True, "Planet is potentially habitable."
    else:
        return False, "Conditions not satisfied."

def apply_habitability_classification(df):
    """Apply habitability classification to all candidate planets"""
    print("Classifying habitability for candidates...")
    
    habitability_results = []
    for _, row in df.iterrows():
        if row.get('disposition', 0) == 1:  # Only check candidates
            is_hab, reason = is_habitable(
                stellar_temp=row.get('stellar_effective_temperature_k', 0),
                stellar_radius=row.get('stellar_radius_solar_radii', 0),
                planet_radius=row.get('planetary_radius_earth_radii', 0),
                planet_temp=row.get('equilibrium_temperature_k', 0),
                insolation_flux=row.get('insolation_flux_earth_flux', 0)
            )
            habitability_results.append(1 if is_hab else 0)
        else:
            habitability_results.append(0)  # Non-candidates are not habitable
    
    df['habitable'] = habitability_results
    habitable_count = sum(habitability_results)
    print(f"Found {habitable_count} potentially habitable planets!")
    return df

# --- 4. Load the Trained Model and Define Paths ---
MODEL_PATH = 'models/random_forest_model.pkl'
SAMPLE_DATA_PATH = os.path.join('data', 'final_dataset.csv')
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

# --- 5. Utility function to process data and predict ---
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
    
    # Apply habitability classification
    original_df = apply_habitability_classification(original_df)
    
    print("✅ Predictions complete.")
    return original_df.to_json(orient='records')

# --- 6. Prediction Endpoints ---
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

# --- 7. Run the App ---
if __name__ == '__main__':
    print("🚀 Flask server is starting...")
    app.run(debug=True, port=5000)