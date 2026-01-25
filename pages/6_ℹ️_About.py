"""
About Page - Project Documentation

Explains project purpose, data sources, and model methodology.
Requirements: 6.4
"""
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="About - Road Accident Analysis",
    page_icon="ℹ",
    layout="wide"
)

st.title("ℹ About This Project")
st.markdown("*French Road Accident Analysis and Visualization System*")

st.divider()

# Project Purpose
st.header("🎯 Project Purpose")
st.markdown("""
This project provides a comprehensive solution for analyzing and visualizing French road traffic 
accidents. The system enables users to:

- **Explore accident data** through interactive visualizations
- **Identify patterns** in temporal, geographic, and environmental factors
- **Predict collision types** based on accident conditions using machine learning

The goal is to provide insights that can help understand road safety patterns and potentially 
inform prevention strategies.
""")

st.divider()

# Data Sources
st.header("📊 Data Sources")
st.markdown("""
This project uses official French road accident data from the **BAAC** 
(Bulletin d'Analyse des Accidents Corporels de la circulation routière).

### Source Files

| File | Description |
|------|-------------|
| `caracteristiques.csv` | Accident characteristics including location, time, weather, and road conditions |
| `usagers.csv` | Information about users/victims involved in each accident |

### Key Variables

**From caracteristiques.csv:**
- `Num_Acc` - Unique accident identifier
- `an`, `mois`, `jour`, `hrmn` - Date and time information
- `lum` - Lighting conditions (1-5)
- `atm` - Weather conditions (1-9)
- `col` - Collision type (1-7)
- `agg` - Urban/Rural location (1-2)
- `int` - Intersection type (1-9)
- `dep` - French department code
- `lat`, `long` - GPS coordinates

**From usagers.csv:**
- `Num_Acc` - Links to accident characteristics
- `grav` - Severity (1=Uninjured, 2=Killed, 3=Hospitalized, 4=Light injury)
- `catu` - User category (Driver, Passenger, Pedestrian)
- `sexe` - Gender
- `trajet` - Trip purpose

### Data Processing

The raw data undergoes several preprocessing steps:
1. **Merging** - Combining accident characteristics with user information
2. **Cleaning** - Handling missing values and removing duplicates
3. **Feature Engineering** - Creating derived features (hour, day of week, etc.)
4. **Encoding** - Converting categorical variables for analysis and modeling
""")

st.divider()

# Model Methodology
st.header("🤖 Model Methodology")
st.markdown("""
### Collision Type Prediction Model

The prediction model classifies accidents into one of seven collision types based on 
environmental and temporal conditions.

### Target Variable

**Collision Type (`col`):**
| Code | Description |
|------|-------------|
| 1 | Two vehicles - frontal collision |
| 2 | Two vehicles - rear-end collision |
| 3 | Two vehicles - side collision |
| 4 | Three+ vehicles - chain collision |
| 5 | Three+ vehicles - multiple collisions |
| 6 | Other collision type |
| 7 | No collision (single vehicle) |

### Features Used

| Feature | Description | Values |
|---------|-------------|--------|
| `lum` | Lighting conditions | 1-5 (Day, Dusk, Night with/without lights) |
| `atm` | Weather conditions | 1-9 (Normal, Rain, Snow, Fog, etc.) |
| `agg` | Location type | 1-2 (Urban, Rural) |
| `int` | Intersection type | 1-9 (None, X, T, Roundabout, etc.) |
| `hour` | Hour of day | 0-23 |
| `day_of_week` | Day of week | 0-6 (Monday-Sunday) |
| `month` | Month | 1-12 |

### Algorithm

**Random Forest Classifier** was chosen for this task because:
- Handles both categorical and numerical features well
- Robust to outliers and missing values
- Provides feature importance rankings
- Can output prediction probabilities

### Training Process

1. **Data Split**: 80% training, 20% testing (stratified)
2. **Class Balancing**: Using `class_weight='balanced'` to handle imbalanced classes
3. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
4. **Model Persistence**: Saved as pickle file for deployment

### Limitations

- The model predicts based on environmental conditions only
- Human factors (driver behavior, vehicle condition) are not captured
- Predictions should be used for informational purposes only
- Model performance varies across different collision types
""")

st.divider()

# Dashboard Pages
st.header("📑 Dashboard Pages")
st.markdown("""
| Page | Description |
|------|-------------|
| **📊 Overview** | Key statistics and summary metrics |
| **📈 Temporal** | Time-based analysis (by hour, day, month, year) |
| **🗺️ Geographic** | Department-based accident distribution |
| **🌤️ Conditions** | Analysis by weather and lighting conditions |
| **🔮 Prediction** | Interactive collision type prediction |
| **ℹ️ About** | Project documentation (this page) |
""")

st.divider()

# Technical Stack
st.header("🛠️ Technical Stack")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Processing")
    st.markdown("""
    - **Python** - Core programming language
    - **Pandas** - Data manipulation and analysis
    - **NumPy** - Numerical computing
    - **Jupyter Notebook** - Interactive preprocessing
    """)

with col2:
    st.subheader("Visualization & ML")
    st.markdown("""
    - **Streamlit** - Web dashboard framework
    - **Plotly** - Interactive visualizations
    - **Scikit-learn** - Machine learning
    - **Hypothesis** - Property-based testing
    """)

st.divider()

# Project Structure
st.header("📁 Project Structure")
st.code("""
road-accident-analysis/
├── app.py                    # Main Streamlit application
├── preprocess.ipynb          # Data preprocessing notebook
├── data/
│   ├── cleaned_accidents.csv # Cleaned dataset
│   └── model_ready.csv       # Feature-engineered dataset
├── models/
│   ├── train_model.py        # Model training script
│   ├── collision_model.pkl   # Trained model
│   └── collision_labels.pkl  # Label mappings
├── pages/
│   ├── 1_📊_Overview.py      # Overview page
│   ├── 2_📈_Temporal.py      # Temporal analysis
│   ├── 3_🗺️_Geographic.py    # Geographic analysis
│   ├── 4_🌤️_Conditions.py    # Conditions analysis
│   ├── 5_🔮_Prediction.py    # Prediction interface
│   └── 6_ℹ️_About.py         # About page (this file)
├── utils/
│   ├── data_loader.py        # Data loading utilities
│   └── visualizations.py     # Chart functions
└── tests/
    └── ...                   # Test files
""", language="text")

st.divider()

# Footer
st.markdown("""
---
**Data Source:** French Ministry of Interior - BAAC (Bulletin d'Analyse des Accidents Corporels)

**Note:** This project is for educational and analytical purposes. The predictions and analyses 
should not be used as the sole basis for road safety decisions.
""")
