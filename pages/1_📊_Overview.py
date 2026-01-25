"""
Overview Page - Key Statistics and Summary

Displays key statistics including total accidents, fatalities, and injuries.
Shows summary cards with metrics.
Requirements: 3.1
"""
import streamlit as st
import pandas as pd
from utils.data_loader import load_cleaned_data, get_data_status_message, apply_filters

# Label mappings
COLLISION_TYPE_LABELS = {
    1: "Two vehicles - frontal",
    2: "Two vehicles - rear",
    3: "Two vehicles - side",
    4: "Three+ vehicles - chain",
    5: "Three+ vehicles - multiple",
    6: "Other collision",
    7: "No collision"
}

st.set_page_config(
    page_title="Overview - Road Accident Analysis",
    page_icon="",
    layout="wide"
)

st.title(" Overview")
st.markdown("*Key statistics and summary of French road accident data*")

# Check data availability
error_message = get_data_status_message()
if error_message:
    st.error(error_message)
    st.stop()

# Load data
try:
    df = load_cleaned_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


def create_filter_sidebar(df: pd.DataFrame) -> dict:
    """Create sidebar filters."""
    st.sidebar.header("🔍 Filters")
    filters = {}
    
    if "year" in df.columns:
        years = sorted(df["year"].dropna().unique())
        filters["year"] = st.sidebar.multiselect(
            "Year", options=years, default=[],
            help="Select years to filter (leave empty for all)"
        )
    
    if "dep" in df.columns:
        departments = sorted(df["dep"].dropna().astype(str).unique())
        filters["department"] = st.sidebar.multiselect(
            "Department", options=departments, default=[],
            help="Select departments to filter (leave empty for all)"
        )
    
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
        label_to_code = {v: k for k, v in collision_options.items()}
        filters["collision_type"] = [label_to_code[label] for label in selected_labels]
    
    return filters


# Create filters and apply
filters = create_filter_sidebar(df)
filtered_df = apply_filters(df, filters)

# Show filter status
if any(len(v) > 0 for v in filters.values() if isinstance(v, list)):
    st.sidebar.success(f"Showing {len(filtered_df):,} of {len(df):,} records")

# Key Metrics Section
st.header("🎯 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_accidents = len(filtered_df)
    st.metric(
        label="Total Accidents",
        value=f"{total_accidents:,}",
        help="Total number of recorded accidents"
    )

with col2:
    if "num_killed" in filtered_df.columns:
        total_killed = int(filtered_df["num_killed"].sum())
        st.metric(
            label="Total Fatalities",
            value=f"{total_killed:,}",
            help="Total number of people killed in accidents"
        )
    else:
        st.metric(label="Total Fatalities", value="N/A")

with col3:
    if "num_hospitalized" in filtered_df.columns:
        total_hospitalized = int(filtered_df["num_hospitalized"].sum())
        st.metric(
            label="Hospitalized",
            value=f"{total_hospitalized:,}",
            help="Total number of people hospitalized"
        )
    else:
        st.metric(label="Hospitalized", value="N/A")

with col4:
    if "num_light_injury" in filtered_df.columns:
        total_light = int(filtered_df["num_light_injury"].sum())
        st.metric(
            label="Light Injuries",
            value=f"{total_light:,}",
            help="Total number of people with light injuries"
        )
    else:
        st.metric(label="Light Injuries", value="N/A")

st.divider()

# Additional Statistics
st.header("📈 Additional Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Time Range")
    if "year" in filtered_df.columns:
        years = filtered_df["year"].dropna()
        if len(years) > 0:
            st.write(f"**From:** {int(years.min())}")
            st.write(f"**To:** {int(years.max())}")
            st.write(f"**Years covered:** {int(years.nunique())}")

with col2:
    st.subheader("🗺️ Geographic Coverage")
    if "dep" in filtered_df.columns:
        num_departments = filtered_df["dep"].nunique()
        st.write(f"**Departments:** {num_departments}")
    if "num_users" in filtered_df.columns:
        total_users = int(filtered_df["num_users"].sum())
        st.write(f"**Total users involved:** {total_users:,}")

with col3:
    st.subheader("💥 Collision Types")
    if "col" in filtered_df.columns:
        num_collision_types = filtered_df["col"].nunique()
        st.write(f"**Types recorded:** {num_collision_types}")
        most_common = filtered_df["col"].mode()
        if len(most_common) > 0:
            most_common_type = int(most_common.iloc[0])
            st.write(f"**Most common:** {COLLISION_TYPE_LABELS.get(most_common_type, f'Type {most_common_type}')}")

st.divider()

# Severity Summary
st.header("⚠️ Severity Summary")

if all(col in filtered_df.columns for col in ["num_killed", "num_hospitalized", "num_light_injury"]):
    severity_data = {
        "Category": ["Fatalities", "Hospitalized", "Light Injuries"],
        "Count": [
            int(filtered_df["num_killed"].sum()),
            int(filtered_df["num_hospitalized"].sum()),
            int(filtered_df["num_light_injury"].sum())
        ]
    }
    severity_df = pd.DataFrame(severity_data)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(severity_df, hide_index=True, use_container_width=True)
    
    with col2:
        import plotly.express as px
        fig = px.pie(
            severity_df,
            values="Count",
            names="Category",
            title="Severity Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Severity data not available in the dataset.")

st.divider()

# Data Quality Info
st.header("📋 Data Quality")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Total records:** {len(filtered_df):,}")
    st.write(f"**Columns:** {len(filtered_df.columns)}")
    
with col2:
    missing_pct = (filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns))) * 100
    st.write(f"**Missing values:** {missing_pct:.2f}%")
    st.write(f"**Memory usage:** {filtered_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
