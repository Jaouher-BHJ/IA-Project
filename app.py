"""
Road Accident Analysis Dashboard

Main Streamlit application for exploring French road accident data.
"""
import streamlit as st
from utils.data_loader import (
    load_cleaned_data,
    get_data_status_message,
    apply_filters
)

# Page configuration
st.set_page_config(
    page_title="Road Accident Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Label mappings for categorical variables
COLLISION_TYPE_LABELS = {
    1: "Two vehicles - frontal",
    2: "Two vehicles - rear",
    3: "Two vehicles - side",
    4: "Three+ vehicles - chain",
    5: "Three+ vehicles - multiple",
    6: "Other collision",
    7: "No collision"
}

LIGHTING_LABELS = {
    1: "Daylight",
    2: "Dusk/Dawn",
    3: "Night with lights",
    4: "Night without lights",
    5: "Not specified"
}

WEATHER_LABELS = {
    1: "Normal",
    2: "Light rain",
    3: "Heavy rain",
    4: "Snow/Hail",
    5: "Fog/Smoke",
    6: "Strong wind",
    7: "Glare",
    8: "Overcast",
    9: "Other"
}


def create_filter_sidebar(df):
    """
    Create sidebar filters for the dashboard.
    
    Args:
        df: DataFrame with accident data
        
    Returns:
        dict: Dictionary with filter selections
    """
    st.sidebar.header("🔍 Filters")
    
    filters = {}
    
    # Year filter
    if "year" in df.columns:
        years = sorted(df["year"].dropna().unique())
        filters["year"] = st.sidebar.multiselect(
            "Year",
            options=years,
            default=[],
            help="Select years to filter (leave empty for all)"
        )
    
    # Department filter
    if "dep" in df.columns:
        departments = sorted(df["dep"].dropna().astype(str).unique())
        filters["department"] = st.sidebar.multiselect(
            "Department",
            options=departments,
            default=[],
            help="Select departments to filter (leave empty for all)"
        )
    
    # Collision type filter
    if "col" in df.columns:
        collision_types = sorted(df["col"].dropna().unique())
        collision_options = {
            ct: COLLISION_TYPE_LABELS.get(ct, f"Type {ct}") 
            for ct in collision_types
        }
        selected_labels = st.sidebar.multiselect(
            "Collision Type",
            options=list(collision_options.values()),
            default=[],
            help="Select collision types to filter (leave empty for all)"
        )
        # Convert labels back to codes
        label_to_code = {v: k for k, v in collision_options.items()}
        filters["collision_type"] = [
            label_to_code[label] for label in selected_labels
        ]
    
    return filters


def display_overview(df, filtered_df):
    """Display overview statistics."""
    st.header("📊 Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Accidents",
            f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
        )
    
    with col2:
        if "num_killed" in filtered_df.columns:
            total_killed = filtered_df["num_killed"].sum()
            st.metric("Total Fatalities", f"{int(total_killed):,}")
        else:
            st.metric("Total Fatalities", "N/A")
    
    with col3:
        if "num_hospitalized" in filtered_df.columns:
            total_hospitalized = filtered_df["num_hospitalized"].sum()
            st.metric("Hospitalized", f"{int(total_hospitalized):,}")
        else:
            st.metric("Hospitalized", "N/A")
    
    with col4:
        if "num_light_injury" in filtered_df.columns:
            total_light = filtered_df["num_light_injury"].sum()
            st.metric("Light Injuries", f"{int(total_light):,}")
        else:
            st.metric("Light Injuries", "N/A")
    
    st.divider()
    
    # Data summary
    st.subheader("📋 Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "year" in filtered_df.columns:
            years = filtered_df["year"].dropna()
            if len(years) > 0:
                st.write(f"**Year Range:** {int(years.min())} - {int(years.max())}")
        
        if "dep" in filtered_df.columns:
            st.write(f"**Departments:** {filtered_df['dep'].nunique()}")
    
    with col2:
        if "col" in filtered_df.columns:
            st.write(f"**Collision Types:** {filtered_df['col'].nunique()}")
        
        st.write(f"**Records after filtering:** {len(filtered_df):,}")


def main():
    """Main application entry point."""
    st.title("🚗 French Road Accident Analysis")
    st.markdown("*Explore and analyze French road traffic accident data (BAAC)*")
    
    # Check data availability
    error_message = get_data_status_message()
    if error_message:
        st.error(error_message)
        st.stop()
    
    # Load data
    try:
        df = load_cleaned_data()
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
        st.stop()
    
    # Create filter sidebar
    filters = create_filter_sidebar(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Show filter status
    if any(len(v) > 0 for v in filters.values() if isinstance(v, list)):
        st.sidebar.success(f"Showing {len(filtered_df):,} of {len(df):,} records")
    
    # Display overview
    display_overview(df, filtered_df)
    
    # Navigation info
    st.divider()
    st.info(
        "👈 Use the sidebar to navigate to different analysis pages:\n"
        "- **Temporal**: Time-based analysis\n"
        "- **Geographic**: Department-based analysis\n"
        "- **Conditions**: Weather/lighting analysis\n"
        "- **Prediction**: Collision type prediction"
    )


if __name__ == "__main__":
    main()
