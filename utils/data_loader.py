# Data loading utilities for Road Accident Analysis
"""
Functions for loading and caching preprocessed accident data.
"""
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple


DATA_DIR = Path("data")

# French department code to name mapping
DEPARTMENT_NAMES = {
    "01": "Ain",
    "02": "Aisne",
    "03": "Allier",
    "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes",
    "06": "Alpes-Maritimes",
    "07": "Ardèche",
    "08": "Ardennes",
    "09": "Ariège",
    "10": "Aube",
    "11": "Aude",
    "12": "Aveyron",
    "13": "Bouches-du-Rhône",
    "14": "Calvados",
    "15": "Cantal",
    "16": "Charente",
    "17": "Charente-Maritime",
    "18": "Cher",
    "19": "Corrèze",
    "2A": "Corse-du-Sud",
    "2B": "Haute-Corse",
    "21": "Côte-d'Or",
    "22": "Côtes-d'Armor",
    "23": "Creuse",
    "24": "Dordogne",
    "25": "Doubs",
    "26": "Drôme",
    "27": "Eure",
    "28": "Eure-et-Loir",
    "29": "Finistère",
    "30": "Gard",
    "31": "Haute-Garonne",
    "32": "Gers",
    "33": "Gironde",
    "34": "Hérault",
    "35": "Ille-et-Vilaine",
    "36": "Indre",
    "37": "Indre-et-Loire",
    "38": "Isère",
    "39": "Jura",
    "40": "Landes",
    "41": "Loir-et-Cher",
    "42": "Loire",
    "43": "Haute-Loire",
    "44": "Loire-Atlantique",
    "45": "Loiret",
    "46": "Lot",
    "47": "Lot-et-Garonne",
    "48": "Lozère",
    "49": "Maine-et-Loire",
    "50": "Manche",
    "51": "Marne",
    "52": "Haute-Marne",
    "53": "Mayenne",
    "54": "Meurthe-et-Moselle",
    "55": "Meuse",
    "56": "Morbihan",
    "57": "Moselle",
    "58": "Nièvre",
    "59": "Nord",
    "60": "Oise",
    "61": "Orne",
    "62": "Pas-de-Calais",
    "63": "Puy-de-Dôme",
    "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées",
    "66": "Pyrénées-Orientales",
    "67": "Bas-Rhin",
    "68": "Haut-Rhin",
    "69": "Rhône",
    "70": "Haute-Saône",
    "71": "Saône-et-Loire",
    "72": "Sarthe",
    "73": "Savoie",
    "74": "Haute-Savoie",
    "75": "Paris",
    "76": "Seine-Maritime",
    "77": "Seine-et-Marne",
    "78": "Yvelines",
    "79": "Deux-Sèvres",
    "80": "Somme",
    "81": "Tarn",
    "82": "Tarn-et-Garonne",
    "83": "Var",
    "84": "Vaucluse",
    "85": "Vendée",
    "86": "Vienne",
    "87": "Haute-Vienne",
    "88": "Vosges",
    "89": "Yonne",
    "90": "Territoire de Belfort",
    "91": "Essonne",
    "92": "Hauts-de-Seine",
    "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne",
    "95": "Val-d'Oise",
    "971": "Guadeloupe",
    "972": "Martinique",
    "973": "Guyane",
    "974": "La Réunion",
    "976": "Mayotte",
}
CLEANED_DATA_FILE = DATA_DIR / "cleaned_accidents.csv"
MODEL_READY_FILE = DATA_DIR / "model_ready.csv"
MODEL_FILE = Path("models") / "collision_model.pkl"


def check_data_availability() -> Tuple[bool, bool, bool]:
    """
    Check if required data files exist.
    
    Returns:
        Tuple of (cleaned_data_exists, model_ready_exists, model_exists)
    """
    return (
        CLEANED_DATA_FILE.exists(),
        MODEL_READY_FILE.exists(),
        MODEL_FILE.exists()
    )


def get_data_status_message() -> Optional[str]:
    """
    Get a user-friendly message about data availability.
    
    Returns:
        Error message if data is missing, None if all data is available
    """
    cleaned_exists, model_ready_exists, _ = check_data_availability()
    
    if not cleaned_exists:
        return (
            "⚠️ **Data not found!**\n\n"
            f"The cleaned data file `{CLEANED_DATA_FILE}` does not exist.\n\n"
            "Please run the preprocessing notebook (`preprocess.ipynb`) first to generate the required data files."
        )
    if not model_ready_exists:
        return (
            "⚠️ **Model-ready data not found!**\n\n"
            f"The model-ready data file `{MODEL_READY_FILE}` does not exist.\n\n"
            "Please run the preprocessing notebook (`preprocess.ipynb`) first to generate the required data files."
        )
    return None


@st.cache_data
def load_cleaned_data() -> pd.DataFrame:
    """
    Load the cleaned accidents dataset with Streamlit caching.
    
    Returns:
        pd.DataFrame: The cleaned accidents data
        
    Raises:
        FileNotFoundError: If the cleaned data file doesn't exist
    """
    if not CLEANED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned data file not found at {CLEANED_DATA_FILE}. "
            "Please run the preprocessing notebook first."
        )
    
    # CRITICAL: Force 'dep' column to be loaded as string to prevent sorting errors
    # The department codes must be strings (e.g., '01', '2A', '971')
    df = pd.read_csv(CLEANED_DATA_FILE, dtype={'dep': str}, low_memory=False)
    
    # Add year column if it doesn't exist but 'an' does
    if "year" not in df.columns and "an" in df.columns:
        df["year"] = df["an"]
    
    return df


@st.cache_data
def load_model_ready_data() -> pd.DataFrame:
    """
    Load the model-ready dataset with encoded features.
    
    Returns:
        pd.DataFrame: The model-ready data with encoded features
        
    Raises:
        FileNotFoundError: If the model-ready data file doesn't exist
    """
    if not MODEL_READY_FILE.exists():
        raise FileNotFoundError(
            f"Model-ready data file not found at {MODEL_READY_FILE}. "
            "Please run the preprocessing notebook first."
        )
    return pd.read_csv(MODEL_READY_FILE)


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply filters to the dataframe.
    
    Args:
        df: The dataframe to filter
        filters: Dictionary with filter keys and values
            - year: List of years to include
            - department: List of departments to include
            - collision_type: List of collision types to include
    
    Returns:
        Filtered dataframe
    """
    filtered_df = df.copy()
    
    if filters.get("year") and len(filters["year"]) > 0:
        filtered_df = filtered_df[filtered_df["year"].isin(filters["year"])]
    
    if filters.get("department") and len(filters["department"]) > 0:
        filtered_df = filtered_df[filtered_df["dep"].isin(filters["department"])]
    
    if filters.get("collision_type") and len(filters["collision_type"]) > 0:
        filtered_df = filtered_df[filtered_df["col"].isin(filters["collision_type"])]
    
    return filtered_df


def get_department_name(code) -> str:
    """
    Map French department code to department name.
    
    Args:
        code: Department code (e.g., "59", "75", "2A", or encoded as 590, 750, 201)
    
    Returns:
        Department name (e.g., "Nord", "Paris", "Corse-du-Sud")
        Falls back to code if not found in mapping.
    """
    code_str = str(code).strip()
    
    # Handle encoded department codes from the dataset
    # Codes like 590 mean department 59, 750 means 75, etc.
    # Corsica: 201 = 2A, 202 = 2B
    # Overseas: 971, 972, 973, 974, 976 stay as-is
    if code_str.isdigit():
        code_int = int(code_str)
        if code_int == 201:
            code_str = "2A"
        elif code_int == 202:
            code_str = "2B"
        elif code_int >= 971:
            # Overseas territories - use as-is
            code_str = str(code_int)
        elif code_int >= 10 and code_int % 10 == 0:
            # Encoded format: 590 -> 59, 750 -> 75, etc.
            code_str = str(code_int // 10).zfill(2)
        elif code_int < 100:
            # Already in correct format, just zero-pad
            code_str = str(code_int).zfill(2)
    
    return DEPARTMENT_NAMES.get(code_str, code_str)


def get_department_display_name(code) -> str:
    """
    Get display name for department including both name and code.
    
    Args:
        code: Department code (e.g., "59", "75", or encoded as 590, 750)
    
    Returns:
        Display name in format "Name (Code)" (e.g., "Nord (59)", "Paris (75)")
    """
    code_str = str(code).strip()
    name = get_department_name(code_str)
    
    # Get the normalized code for display
    if code_str.isdigit():
        code_int = int(code_str)
        if code_int == 201:
            display_code = "2A"
        elif code_int == 202:
            display_code = "2B"
        elif code_int >= 971:
            display_code = str(code_int)
        elif code_int >= 10 and code_int % 10 == 0:
            display_code = str(code_int // 10).zfill(2)
        else:
            display_code = str(code_int).zfill(2)
    else:
        display_code = code_str
    
    if name != display_code:
        return f"{name} ({display_code})"
    return code_str
