import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
import json

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

@app.route('/heatmap_data', methods=['POST'])
def get_heatmap_data():
    """Generate proper KDE heatmap data server-side"""
    if model is None: 
        return jsonify({"error": "Model is not loaded."}), 500
    
    try:
        from scipy.stats import gaussian_kde
        
        data = request.get_json()
        predictions = data.get('predictions', [])
        df = pd.DataFrame(predictions)
        
        # Clean & prepare data
        df = df[(df['equilibrium_temperature_k'].notna()) & 
                (df['insolation_flux_earth_flux'].notna())]
        
        if len(df) == 0:
            return jsonify({"error": "No valid temperature/flux data found"}), 400
        
        x = df['equilibrium_temperature_k'].values
        y = df['insolation_flux_earth_flux'].values
        
        # Mock habitability probability (same formula as new_code.py)
        probability = np.exp(-((x - 288) ** 2)/(2*50**2)) * np.exp(-((y - 1)**2)/(2*0.5**2))
        
        # Normalize probability
        if probability.max() > 0:
            probability = probability / probability.max()
        
        # Define tight zoom grid (Earth-like range: 150–450K and 0–5 Earth Flux)
        x_min, x_max = 150, 450
        y_min, y_max = 0, 5
        
        x_lin = np.linspace(x_min, x_max, 400)
        y_lin = np.linspace(y_min, y_max, 400)
        X, Y = np.meshgrid(x_lin, y_lin)
        grid_coords = np.vstack([X.ravel(), Y.ravel()])
        
        # Smooth KDE
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, weights=probability, bw_method=0.1)
        Z = kde(grid_coords).reshape(X.shape)
        
        # Create Plotly heatmap figure
        fig = go.Figure(go.Heatmap(
            x=x_lin.tolist(),
            y=y_lin.tolist(),
            z=Z.tolist(),
            colorscale='Turbo',
            colorbar=dict(title="Habitability<br>Probability"),
            zsmooth="best"
        ))
        
        fig.update_layout(
            title="Temperature–Insolation Habitability Heatmap",
            xaxis_title="Equilibrium Temperature (K)",
            yaxis_title="Insolation Flux (Earth Flux)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0', family='Space Grotesk'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            autosize=True,
            height=600
        )
        
        return jsonify(json.loads(fig.to_json()))
    except Exception as e:
        return jsonify({"error": f"Heatmap generation error: {str(e)}"}), 500

@app.route('/generate_radar', methods=['POST'])
def generate_radar():
    """Generate planetary profile radar chart server-side"""
    try:
        data = request.get_json()
        predictions = data.get('predictions', [])
        selected_indices = data.get('selected', [])
        
        df = pd.DataFrame(predictions)
        candidates = df[df['disposition'] == 1].head(50)
        
        features = [
            "equilibrium_temperature_k",
            "planetary_radius_earth_radii",
            "insolation_flux_earth_flux",
            "stellar_effective_temperature_k",
            "stellar_radius_solar_radii",
            "orbital_period_days"
        ]
        
        feature_labels = [
            'Equilibrium Temp',
            'Planetary Radius',
            'Insolation Flux',
            'Stellar Temp',
            'Stellar Radius',
            'Orbital Period'
        ]
        
        earth_vals = {
            "equilibrium_temperature_k": 288,
            "planetary_radius_earth_radii": 1,
            "insolation_flux_earth_flux": 1,
            "stellar_effective_temperature_k": 5778,
            "stellar_radius_solar_radii": 1,
            "orbital_period_days": 365.25
        }
        
        fig = go.Figure()
        
        # Earth baseline
        fig.add_trace(go.Scatterpolar(
            r=[1]*len(features),
            theta=feature_labels,
            fill='toself',
            name='Earth',
            line_color='green',
            fillcolor='rgba(34,197,94,0.2)',
            opacity=0.8
        ))
        
        # Add selected planets
        colors = ['dodgerblue', 'orange', 'deeppink', 'gold', 'cyan']
        for i, idx in enumerate(selected_indices):
            if idx < len(candidates):
                row = candidates.iloc[idx]
                r_vals = [min(row[f] / earth_vals[f], 3) for f in features]
                planet_name = row.get('planet_name', f'Planet {idx}')
                
                fig.add_trace(go.Scatterpolar(
                    r=r_vals,
                    theta=feature_labels,
                    fill='toself',
                    name=str(planet_name),
                    line_color=colors[i % len(colors)],
                    opacity=0.6
                ))
        
        fig.update_layout(
            title="Planetary Profiles - Normalized to Earth",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 3]),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0', family='Space Grotesk'),
            autosize=True,
            height=480,
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        return jsonify(json.loads(fig.to_json()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
      
# --- 7. Run the App ---
if __name__ == '__main__':
    print("🚀 Flask server is starting...")
    app.run(debug=True, port=5000)