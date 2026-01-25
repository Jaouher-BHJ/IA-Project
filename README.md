# 🚗 French Road Accident Analysis & Prediction

A comprehensive data analysis and machine learning project for predicting road accident collision types and severity using French government accident data (2005-2024).

> **📢 Recent Updates**: Project renamed to `ia-project` and preprocessing improved. See [QUICK_START.md](QUICK_START.md) for setup instructions.

## 📋 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [File Documentation](#-file-documentation)
- [Usage Guide](#-usage-guide)
- [Data](#-data)
- [Models](#-models)
- [Dashboard Pages](#-dashboard-pages)

## ✨ Features

- **Multi-target ML prediction**: Predicts both collision type (7 classes) and severity (4 classes) simultaneously
- **Automatic feature selection**: Correlation-based feature selection from accident data
- **Interactive dashboard**: Streamlit-based visualization and prediction interface
- **Comprehensive analysis**: Temporal, geographic, and condition-based insights
- **Model comparison**: Compare Random Forest and XGBoost with/without PCA
- **Data preprocessing**: Automated cleaning, outlier detection, and feature engineering

## 📁 Project Structure

```
ia-project/
├── app.py                      # Main Streamlit dashboard
├── main.py                     # Entry point (basic)
├── check_readiness.py          # Data validation script
├── preprocess.ipynb            # Data preprocessing notebook
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Locked dependency versions
├── .python-version             # Python version specification
│
├── data/                       # Accident data (CSV files)
│   ├── caracteristiques-*.csv # Accident characteristics (2005-2024)
│   ├── usagers-*.csv          # User/victim data (2005-2024)
│   ├── cleaned_accidents.csv  # Processed data (generated)
│   └── model_ready.csv        # ML-ready dataset (generated)
│
├── models/                     # Trained ML models
│   ├── compare_multitarget_models.py  # Model training script
│   ├── rf_nopca_multitarget.pkl       # Trained model
│   ├── collision_labels.pkl           # Label encoders
│   └── multitarget_comparison.png     # Performance comparison
│
├── pages/                      # Streamlit dashboard pages
│   ├── 1_📊_Overview.py       # Data overview and statistics
│   ├── 2_📈_Temporal.py       # Time-based analysis
│   ├── 3_🗺️_Geographic.py    # Geographic visualization
│   ├── 4_🌤️_Conditions.py    # Weather/road conditions
│   ├── 5_🔮_Prediction.py     # ML prediction interface
│   └── 6_ℹ️_About.py          # Project information
│
├── utils/                      # Utility modules
│   ├── data_loader.py         # Data loading and filtering
│   └── visualizations.py      # Plotting functions
│
└── tests/                      # Unit tests
    └── test_data_loader.py    # Data loader tests
```

## 🔧 Installation

### Prerequisites
- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd IA_project
```

2. **Install dependencies**

Using uv (recommended):
```bash
uv sync
```

Using pip:
```bash
pip install -e .
```

3. **Verify installation**
```bash
python check_readiness.py
```

## 🚀 Quick Start

### Step 1: Preprocess Data (3-5 minutes)

Open `preprocess.ipynb` in Jupyter/VS Code and run all cells:
- Loads raw accident data from `data/` folder
- Performs correlation analysis
- Automatically selects best features
- Detects and removes outliers
- Generates `cleaned_accidents.csv` and `model_ready.csv`

**Success indicator**: See "PREPROCESSING COMPLETE!" message

### Step 2: Train Models (10-15 minutes)

```bash
python models/compare_multitarget_models.py
```

This will:
- Train 4 different models (RF/XGB × PCA/No PCA)
- Evaluate multi-target performance
- Save the best model
- Generate comparison visualization

**Success indicator**: See "COMPARISON COMPLETE!" with metrics

### Step 3: Launch Dashboard

```bash
python -m streamlit run app.py
# Or use the batch file: run_app.bat
```

Opens interactive dashboard at `http://localhost:8501`

> **💡 Tip**: If you get "streamlit command not found", use `python -m streamlit run app.py` instead.

## 📄 File Documentation

### Core Application Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `app.py` | Main Streamlit dashboard entry point | Run to start the web interface |
| `main.py` | Basic Python entry point | Not actively used (placeholder) |
| `check_readiness.py` | Validates data files and dependencies | Run after setup to verify installation |
| `preprocess.ipynb` | Data preprocessing and feature selection | Run first to prepare data for ML |

### Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata and dependencies |
| `uv.lock` | Locked dependency versions for reproducibility |
| `.python-version` | Specifies required Python version |
| `.gitignore` | Git ignore patterns |

### Data Files (`data/`)

| File Pattern | Description |
|--------------|-------------|
| `caracteristiques-*.csv` | Raw accident characteristics data (2005-2024) |
| `usagers-*.csv` | Raw user/victim data (2005-2024) |
| `cleaned_accidents.csv` | Preprocessed data (generated by notebook) |
| `model_ready.csv` | ML-ready dataset with selected features (generated) |

### Model Files (`models/`)

| File | Purpose |
|------|---------|
| `compare_multitarget_models.py` | Trains and compares multiple ML models |
| `rf_nopca_multitarget.pkl` | Trained Random Forest model (generated) |
| `collision_labels.pkl` | Label encoders for predictions (generated) |
| `multitarget_comparison.png` | Model performance visualization (generated) |

### Utility Modules (`utils/`)

| File | Purpose |
|------|---------|
| `data_loader.py` | Functions for loading and filtering accident data |
| `visualizations.py` | Plotting and visualization utilities |

### Dashboard Pages (`pages/`)

| File | Page Content |
|------|--------------|
| `1_📊_Overview.py` | Dataset statistics and overview |
| `2_📈_Temporal.py` | Time-based trends and patterns |
| `3_🗺️_Geographic.py` | Geographic distribution maps |
| `4_🌤️_Conditions.py` | Weather and road condition analysis |
| `5_🔮_Prediction.py` | ML prediction interface |
| `6_ℹ️_About.py` | Project information and documentation |

### Test Files (`tests/`)

| File | Purpose |
|------|---------|
| `test_data_loader.py` | Unit tests for data loading functions |

## 📊 Usage Guide

### Running the Dashboard

```bash
streamlit run app.py
```

Navigate through pages using the sidebar:
1. **Overview**: Explore dataset statistics
2. **Temporal**: Analyze trends over time
3. **Geographic**: View accident distribution maps
4. **Conditions**: Examine weather/road impacts
5. **Prediction**: Make predictions with trained models
6. **About**: Learn about the project

### Making Predictions

1. Go to the "Prediction" page
2. Select a trained model from the dropdown
3. Enter accident conditions (weather, road type, time, etc.)
4. Click "Predict" to get:
   - Collision type prediction (7 classes)
   - Severity prediction (4 classes)
   - Probability distributions

### Training New Models

```bash
python models/compare_multitarget_models.py
```

Models trained:
- Random Forest with PCA
- Random Forest without PCA
- XGBoost with PCA
- XGBoost without PCA

The script automatically saves the best-performing model.

## 📈 Data

### Source
French government road accident data (2005-2024) from [data.gouv.fr](https://www.data.gouv.fr/)

### Features
- Temporal: Date, time, day of week
- Geographic: Department, commune, GPS coordinates
- Conditions: Weather, lighting, road surface
- Road characteristics: Type, category, layout
- Collision details: Type, location, severity

### Targets
- **Collision Type**: 7 classes (frontal, rear, side, chain, multiple, other, none)
- **Severity**: 4 classes (unharmed, light injury, hospitalized, fatal)

## 🤖 Models

### Multi-Target Prediction
Uses `MultiOutputClassifier` to predict both collision type and severity simultaneously.

### Algorithms
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting with regularization

### Feature Engineering
- Automatic correlation-based feature selection
- Optional PCA dimensionality reduction
- Outlier detection and removal

### Performance Metrics
- Accuracy per target
- Classification reports
- Confusion matrices
- Cross-validation scores

## 🎯 Dashboard Pages

### 1. Overview 📊
- Dataset size and date range
- Missing value analysis
- Feature distributions
- Summary statistics

### 2. Temporal 📈
- Accidents by year/month/day
- Hourly patterns
- Seasonal trends
- Day of week analysis

### 3. Geographic 🗺️
- Department-level heatmaps
- Regional distribution
- Urban vs rural patterns
- Interactive maps

### 4. Conditions 🌤️
- Weather impact analysis
- Lighting conditions
- Road surface effects
- Visibility factors

### 5. Prediction 🔮
- Model selection interface
- Input form for conditions
- Real-time predictions
- Probability distributions

### 6. About ℹ️
- Project description
- Data sources
- Methodology
- Contact information

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=utils --cov=models
```

## 🛠️ Troubleshooting

**Dashboard won't start?**
- Use `python -m streamlit run app.py` instead of just `streamlit run app.py`
- Ensure Streamlit is installed: `pip install streamlit`
- Check that `data/cleaned_accidents.csv` exists

**"Missing columns" error?**
- Your data is from old preprocessing
- Run `preprocess.ipynb` again to regenerate data files

**No predictions available?**
- Run `preprocess.ipynb` first
- Train models with `python models/compare_multitarget_models.py`

**Import errors?**
- Verify all dependencies: `pip install -e .`
- Check Python version: `python --version` (should be 3.10+)

**Need help?**
- Check [QUICK_START.md](QUICK_START.md) for step-by-step guide
- Check [ANSWER_TO_YOUR_QUESTIONS.md](ANSWER_TO_YOUR_QUESTIONS.md) for common questions
- Run `python check_readiness.py` to diagnose issues

## 📝 License

This project uses public French government data. Please refer to [data.gouv.fr](https://www.data.gouv.fr/) for data licensing terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or feedback, please open an issue on the repository.

---

**Built with**: Python, Streamlit, Scikit-learn, XGBoost, Pandas, Plotly
