"""
Temporal Analysis Page - Time-based Analysis

Displays accidents over time (yearly trend).
Includes interactive filters.

Requirements: 3.4
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    page_title="Temporal Analysis - Road Accident Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Temporal Analysis")
st.markdown("*Analyze accident trends over time*")

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
            "Year", 
            options=years, 
            default=[],
            help="Select years to filter (leave empty for all)"
        )
    
    if "dep" in df.columns:
        departments = sorted(df["dep"].dropna().unique())
        filters["department"] = st.sidebar.multiselect(
            "Department", 
            options=departments, 
            default=[],
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

# Yearly Trend - Main visualization
st.header("Number of Accidents by Year")

if "year" in filtered_df.columns:
    yearly_counts = filtered_df.groupby("year").size().reset_index(name="count")
    
    # Create the line chart with markers
    fig_yearly = go.Figure()
    
    fig_yearly.add_trace(go.Scatter(
        x=yearly_counts["year"],
        y=yearly_counts["count"],
        mode='lines+markers',
        line=dict(color='#636EFA', width=2),
        marker=dict(size=8, color='#636EFA'),
        hovertemplate='Year: %{x}<br>Accidents: %{y:,}<extra></extra>'
    ))
    
    fig_yearly.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Accidents",
        hovermode="x unified",
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        )
    )
    
    st.plotly_chart(fig_yearly, use_container_width=True)
    
    # Show summary statistics
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Accidents",
            f"{filtered_df.shape[0]:,}"
        )
    
    with col2:
        if len(yearly_counts) > 0:
            avg_per_year = yearly_counts["count"].mean()
            st.metric(
                "Average per Year",
                f"{int(avg_per_year):,}"
            )
    
    with col3:
        if len(yearly_counts) > 0:
            max_year = yearly_counts.loc[yearly_counts["count"].idxmax()]
            st.metric(
                "Peak Year",
                f"{int(max_year['year'])}",
                f"{int(max_year['count']):,} accidents"
            )
    
    with col4:
        if len(yearly_counts) > 0:
            min_year = yearly_counts.loc[yearly_counts["count"].idxmin()]
            st.metric(
                "Lowest Year",
                f"{int(min_year['year'])}",
                f"{int(min_year['count']):,} accidents"
            )
    
else:
    st.warning("Year column not available in the dataset.")
