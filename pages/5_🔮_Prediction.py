"""
Prediction Page - Multi-Target Prediction Interface

Allows users to select a model and predict collision type AND severity
using trained machine learning models.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import os

# Model paths
MODELS_DIR = Path("models")

# Label mappings
LIGHTING_OPTIONS = {
    1: "Daylight", 2: "Dusk/Dawn", 3: "Night with street lights",
    4: "Night without street lights", 5: "Not specified"
}

WEATHER_OPTIONS = {
    1: "Normal", 2: "Light rain", 3: "Heavy rain", 4: "Snow/Hail",
    5: "Fog/Smoke", 6: "Strong wind", 7: "Glare", 8: "Overcast", 9: "Other"
}

LOCATION_OPTIONS = {1: "Urban area", 2: "Rural area"}

INTERSECTION_OPTIONS = {
    1: "None", 2: "X intersection", 3: "T intersection", 4: "Y intersection",
    5: "5+ branches", 6: "Roundabout", 7: "Square", 8: "Level crossing", 9: "Other"
}

COLLISION_LABELS = {
    0: "Frontale", 1: "Par arrière", 2: "Par le côté",
    3: "En chaîne", 4: "Multiples", 5: "Autre", 6: "Sans collision"
}

SEVERITY_LABELS = {
    0: "Indemne", 1: "Tué", 2: "Hospitalisé", 3: "Blessé léger"
}

DAY_OPTIONS = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
               4: "Friday", 5: "Saturday", 6: "Sunday"}

MONTH_OPTIONS = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
                 6: "June", 7: "July", 8: "August", 9: "September",
                 10: "October", 11: "November", 12: "December"}

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Multi-Target Prediction")
st.markdown("*Predict collision type AND severity based on accident conditions*")


@st.cache_resource
def discover_models():
    """Discover available trained models."""
    models = {}
    
    if not MODELS_DIR.exists():
        return models
    
    # Look for multitarget .pkl files
    for file in MODELS_DIR.glob("*_multitarget.pkl"):
        model_name = file.stem.replace('_multitarget', '')
        # Create friendly name
        parts = model_name.split('_')
        if len(parts) >= 2:
            model_type = parts[0].upper()
            pca_status = "with PCA" if parts[1] == 'pca' else "without PCA"
            friendly_name = f"{model_type} {pca_status}"
        else:
            friendly_name = model_name
        
        models[friendly_name] = file
    
    return models


@st.cache_resource
def load_model(model_path):
    """Load a trained model with metadata."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle both old and new format
        if isinstance(model_data, dict):
            return model_data
        else:
            # Old format - just the model
            return {'model': model_data, 'features': None, 'use_pca': False}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def predict_multitarget(model_data, features: dict) -> tuple:
    """Make multi-target prediction."""
    # Extract model and metadata
    if isinstance(model_data, dict):
        model = model_data.get('model')
        selected_features = model_data.get('features')
        use_pca = model_data.get('use_pca', False)
        pca = model_data.get('pca')
        scaler = model_data.get('scaler')
    else:
        model = model_data
        selected_features = None
        use_pca = False
        pca = None
        scaler = None
    
    # Determine feature names
    if selected_features:
        feature_names = selected_features
    else:
        # Default features
        feature_names = ['lum', 'atm', 'agg', 'int', 'hour', 'day_of_week', 'month']
    
    # Create feature array
    X = np.array([[features.get(f, 0) for f in feature_names]])
    
    # Apply PCA if needed
    if use_pca and scaler and pca:
        X = scaler.transform(X)
        X = pca.transform(X)
    
    try:
        # Predict
        predictions = model.predict(X)
        
        # Handle different model types
        if len(predictions.shape) == 2 and predictions.shape[1] == 2:
            # Multi-output model
            col_pred = int(predictions[0, 0])
            sev_pred = int(predictions[0, 1])
        else:
            # Single output model
            col_pred = int(predictions[0])
            sev_pred = None
        
        # Get probabilities if available
        col_proba = {}
        sev_proba = {}
        
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict_proba(X)
                if isinstance(probas, list) and len(probas) == 2:
                    # Multi-output
                    col_proba = {i: float(p) for i, p in enumerate(probas[0][0])}
                    sev_proba = {i: float(p) for i, p in enumerate(probas[1][0])}
                else:
                    col_proba = {i: float(p) for i, p in enumerate(probas[0])}
            except:
                pass
        
        return col_pred, sev_pred, col_proba, sev_proba
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, {}, {}


# Discover available models
available_models = discover_models()

if not available_models:
    st.error(
        "⚠️ **No models found!**\n\n"
        "Please train a model first by running:\n"
        "```bash\n"
        "python models/compare_multitarget_models.py\n"
        "```"
    )
    st.stop()

st.success(f"✅ Found {len(available_models)} model(s)")

# Model selection
st.header("🤖 Select Model")
model_name = st.selectbox(
    "Choose a trained model:",
    options=list(available_models.keys()),
    help="Select which model to use for prediction"
)

model_data = load_model(available_models[model_name])

if model_data is None:
    st.error(f"Failed to load model: {model_name}")
    st.stop()

# Show model info
if isinstance(model_data, dict):
    st.info(f"**Model:** {model_name}")
    if 'metrics' in model_data:
        metrics = model_data['metrics']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Collision Accuracy", f"{metrics.get('collision_accuracy', 0):.1%}")
        with col2:
            st.metric("Severity Accuracy", f"{metrics.get('severity_accuracy', 0):.1%}")
        with col3:
            st.metric("Overall F1", f"{metrics.get('avg_f1', 0):.3f}")
else:
    st.info(f"**Loaded:** {model_name}")

st.divider()

# Input Section
st.header("📝 Enter Accident Conditions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Environmental Conditions")
    
    lighting_label = st.selectbox("💡 Lighting", list(LIGHTING_OPTIONS.values()), index=0)
    lighting_code = [k for k, v in LIGHTING_OPTIONS.items() if v == lighting_label][0]
    
    weather_label = st.selectbox("🌧️ Weather", list(WEATHER_OPTIONS.values()), index=0)
    weather_code = [k for k, v in WEATHER_OPTIONS.items() if v == weather_label][0]
    
    location_label = st.selectbox("🏙️ Location", list(LOCATION_OPTIONS.values()), index=0)
    location_code = [k for k, v in LOCATION_OPTIONS.items() if v == location_label][0]
    
    intersection_label = st.selectbox("🔀 Intersection", list(INTERSECTION_OPTIONS.values()), index=0)
    intersection_code = [k for k, v in INTERSECTION_OPTIONS.items() if v == intersection_label][0]

with col2:
    st.subheader("Time & Context")
    
    hour = st.slider("🕐 Hour", 0, 23, 12)
    
    day_label = st.selectbox("📅 Day", list(DAY_OPTIONS.values()), index=0)
    day_code = [k for k, v in DAY_OPTIONS.items() if v == day_label][0]
    
    month_label = st.selectbox("📆 Month", list(MONTH_OPTIONS.values()), index=0)
    month_code = [k for k, v in MONTH_OPTIONS.items() if v == month_label][0]
    
    # Additional features for multi-target models
    num_users = st.number_input("👥 Number of people involved", min_value=1, max_value=10, value=2)
    num_light_injury = st.number_input("🤕 Light injuries", min_value=0, max_value=10, value=0)

st.divider()

# Collect features
features = {
    'lum': lighting_code,
    'atm': weather_code,
    'agg': location_code,
    'int': intersection_code,
    'hour': hour,
    'day_of_week': day_code,
    'month': month_code,
    'num_users': num_users,
    'num_light_injury': num_light_injury
}

# Prediction
st.header("🎯 Prediction Results")

if st.button("🔮 Predict", type="primary", use_container_width=True):
    with st.spinner("Making prediction..."):
        col_pred, sev_pred, col_proba, sev_proba = predict_multitarget(model_data, features)
        
        if col_pred is not None:
            # Display results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.success("### Collision Type")
                col_label = COLLISION_LABELS.get(col_pred, f"Unknown ({col_pred})")
                st.markdown(f"## 💥 {col_label}")
                if col_proba:
                    confidence = col_proba.get(col_pred, 0) * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
            
            with result_col2:
                if sev_pred is not None:
                    st.success("### Severity")
                    sev_label = SEVERITY_LABELS.get(sev_pred, f"Unknown ({sev_pred})")
                    st.markdown(f"## 🚑 {sev_label}")
                    if sev_proba:
                        confidence = sev_proba.get(sev_pred, 0) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show probabilities
            if col_proba or sev_proba:
                st.divider()
                st.subheader("📊 Probability Distributions")
                
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    if col_proba:
                        st.markdown("**Collision Type Probabilities:**")
                        prob_data = []
                        for code, prob in sorted(col_proba.items()):
                            label = COLLISION_LABELS.get(code, f"Code {code}")
                            prob_data.append({"Type": label, "Probability": prob * 100})
                        prob_df = pd.DataFrame(prob_data).sort_values("Probability", ascending=False)
                        st.dataframe(prob_df, hide_index=True, use_container_width=True)
                
                with prob_col2:
                    if sev_proba:
                        st.markdown("**Severity Probabilities:**")
                        prob_data = []
                        for code, prob in sorted(sev_proba.items()):
                            label = SEVERITY_LABELS.get(code, f"Code {code}")
                            prob_data.append({"Severity": label, "Probability": prob * 100})
                        prob_df = pd.DataFrame(prob_data).sort_values("Probability", ascending=False)
                        st.dataframe(prob_df, hide_index=True, use_container_width=True)

st.divider()

# Information
st.header("ℹ️ About")
st.markdown("""
This tool uses machine learning models trained on French road accident data (BAAC).

**Multi-Target Prediction:**
- **Collision Type**: 7 classes (frontal, rear-end, side, chain, multiple, other, no collision)
- **Severity**: 4 classes (unharmed, killed, hospitalized, light injury)

**Features Used:**
- Environmental: Lighting, weather, location, intersection type
- Temporal: Hour, day of week, month
- Context: Number of people involved, light injuries

**Note:** Predictions are based on historical patterns and should be used for informational purposes only.
""")
