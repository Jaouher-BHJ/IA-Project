"""
Conditions Analysis Page - Weather/Lighting Analysis

Displays accidents by lighting conditions, weather conditions, and collision type distribution.
Requirements: 3.6, 3.7
"""
import streamlit as st
import pandas as pd
import plotly.express as px
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

INTERSECTION_LABELS = {
    1: "None",
    2: "X intersection",
    3: "T intersection",
    4: "Y intersection",
    5: "5+ branches",
    6: "Roundabout",
    7: "Square",
    8: "Level crossing",
    9: "Other"
}

st.set_page_config(
    page_title="Conditions Analysis - Road Accident Analysis",
    page_icon="",
    layout="wide"
)

st.title(" Conditions Analysis")
st.markdown("*Analyze accidents by weather, lighting, and collision conditions*")

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

# Lighting Conditions Analysis
st.header("💡 Accidents by Lighting Conditions")

col1, col2 = st.columns(2)

with col1:
    if "lum" in filtered_df.columns:
        lighting_counts = filtered_df.groupby("lum").size().reset_index(name="count")
        lighting_counts["lighting"] = lighting_counts["lum"].map(LIGHTING_LABELS)
        
        fig_lighting = px.bar(
            lighting_counts,
            x="lighting",
            y="count",
            title="Number of Accidents by Lighting Condition",
            color="count",
            color_continuous_scale="YlOrBr",
            text="count"
        )
        fig_lighting.update_layout(
            xaxis_title="Lighting Condition",
            yaxis_title="Number of Accidents",
            showlegend=False
        )
        fig_lighting.update_traces(
            texttemplate='%{text:,}',
            textposition='outside',
            hovertemplate="Lighting: %{x}<br>Accidents: %{y:,}<extra></extra>"
        )
        st.plotly_chart(fig_lighting, use_container_width=True)
    else:
        st.warning("Lighting column not available in the dataset.")

with col2:
    if "lum" in filtered_df.columns:
        lighting_counts = filtered_df.groupby("lum").size().reset_index(name="count")
        lighting_counts["lighting"] = lighting_counts["lum"].map(LIGHTING_LABELS)
        
        fig_lighting_pie = px.pie(
            lighting_counts,
            values="count",
            names="lighting",
            title="Distribution by Lighting Condition",
            color_discrete_sequence=px.colors.sequential.YlOrBr
        )
        fig_lighting_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_lighting_pie, use_container_width=True)

st.divider()

# Weather Conditions Analysis
st.header("🌧️ Accidents by Weather Conditions")

col1, col2 = st.columns(2)

with col1:
    if "atm" in filtered_df.columns:
        weather_counts = filtered_df.groupby("atm").size().reset_index(name="count")
        weather_counts["weather"] = weather_counts["atm"].map(WEATHER_LABELS)
        
        fig_weather = px.bar(
            weather_counts,
            x="weather",
            y="count",
            title="Number of Accidents by Weather Condition",
            color="count",
            color_continuous_scale="Blues",
            text="count"
        )
        fig_weather.update_layout(
            xaxis_title="Weather Condition",
            yaxis_title="Number of Accidents",
            showlegend=False
        )
        fig_weather.update_traces(
            texttemplate='%{text:,}',
            textposition='outside',
            hovertemplate="Weather: %{x}<br>Accidents: %{y:,}<extra></extra>"
        )
        st.plotly_chart(fig_weather, use_container_width=True)
    else:
        st.warning("Weather column not available in the dataset.")

with col2:
    if "atm" in filtered_df.columns:
        weather_counts = filtered_df.groupby("atm").size().reset_index(name="count")
        weather_counts["weather"] = weather_counts["atm"].map(WEATHER_LABELS)
        
        fig_weather_pie = px.pie(
            weather_counts,
            values="count",
            names="weather",
            title="Distribution by Weather Condition",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_weather_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_weather_pie, use_container_width=True)

st.divider()

# Collision Type Distribution
st.header("💥 Collision Type Distribution")

col1, col2 = st.columns(2)

with col1:
    if "col" in filtered_df.columns:
        collision_counts = filtered_df.groupby("col").size().reset_index(name="count")
        collision_counts["collision_type"] = collision_counts["col"].map(COLLISION_TYPE_LABELS)
        
        fig_collision = px.bar(
            collision_counts,
            x="collision_type",
            y="count",
            title="Number of Accidents by Collision Type",
            color="count",
            color_continuous_scale="Reds",
            text="count"
        )
        fig_collision.update_layout(
            xaxis_title="Collision Type",
            yaxis_title="Number of Accidents",
            showlegend=False,
            xaxis_tickangle=-45
        )
        fig_collision.update_traces(
            texttemplate='%{text:,}',
            textposition='outside',
            hovertemplate="Type: %{x}<br>Accidents: %{y:,}<extra></extra>"
        )
        st.plotly_chart(fig_collision, use_container_width=True)
    else:
        st.warning("Collision type column not available in the dataset.")

with col2:
    if "col" in filtered_df.columns:
        collision_counts = filtered_df.groupby("col").size().reset_index(name="count")
        collision_counts["collision_type"] = collision_counts["col"].map(COLLISION_TYPE_LABELS)
        
        fig_collision_pie = px.pie(
            collision_counts,
            values="count",
            names="collision_type",
            title="Distribution by Collision Type",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_collision_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_collision_pie, use_container_width=True)

st.divider()

# Intersection Type Analysis
st.header("🔀 Accidents by Intersection Type")

if "int" in filtered_df.columns:
    intersection_counts = filtered_df.groupby("int").size().reset_index(name="count")
    intersection_counts["intersection"] = intersection_counts["int"].map(INTERSECTION_LABELS)
    
    fig_intersection = px.bar(
        intersection_counts,
        x="intersection",
        y="count",
        title="Number of Accidents by Intersection Type",
        color="count",
        color_continuous_scale="Purples",
        text="count"
    )
    fig_intersection.update_layout(
        xaxis_title="Intersection Type",
        yaxis_title="Number of Accidents",
        showlegend=False,
        xaxis_tickangle=-45
    )
    fig_intersection.update_traces(
        texttemplate='%{text:,}',
        textposition='outside',
        hovertemplate="Intersection: %{x}<br>Accidents: %{y:,}<extra></extra>"
    )
    st.plotly_chart(fig_intersection, use_container_width=True)
else:
    st.warning("Intersection column not available in the dataset.")

st.divider()

# Cross-Analysis: Lighting vs Weather
st.header("🔄 Cross-Analysis: Lighting vs Weather")

if "lum" in filtered_df.columns and "atm" in filtered_df.columns:
    cross_data = filtered_df.groupby(["lum", "atm"]).size().reset_index(name="count")
    cross_pivot = cross_data.pivot(index="lum", columns="atm", values="count").fillna(0)
    
    # Rename index and columns
    cross_pivot.index = [LIGHTING_LABELS.get(i, str(i)) for i in cross_pivot.index]
    cross_pivot.columns = [WEATHER_LABELS.get(c, str(c)) for c in cross_pivot.columns]
    
    fig_heatmap = px.imshow(
        cross_pivot,
        labels=dict(x="Weather Condition", y="Lighting Condition", color="Accidents"),
        title="Accident Frequency: Lighting vs Weather Conditions",
        color_continuous_scale="YlOrRd",
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.warning("Lighting and/or weather columns not available for cross-analysis.")

st.divider()

# Severity by Conditions
st.header("⚠️ Severity Analysis by Conditions")

tab1, tab2 = st.tabs(["By Lighting", "By Weather"])

with tab1:
    if "lum" in filtered_df.columns and "num_killed" in filtered_df.columns:
        severity_by_lighting = filtered_df.groupby("lum").agg({
            "num_killed": "sum",
            "num_hospitalized": "sum" if "num_hospitalized" in filtered_df.columns else lambda x: 0,
            "num_light_injury": "sum" if "num_light_injury" in filtered_df.columns else lambda x: 0
        }).reset_index()
        severity_by_lighting["lighting"] = severity_by_lighting["lum"].map(LIGHTING_LABELS)
        
        fig_sev_light = px.bar(
            severity_by_lighting,
            x="lighting",
            y=["num_killed", "num_hospitalized", "num_light_injury"],
            title="Casualties by Lighting Condition",
            barmode="group",
            color_discrete_map={
                "num_killed": "#d62728",
                "num_hospitalized": "#ff7f0e",
                "num_light_injury": "#2ca02c"
            }
        )
        fig_sev_light.update_layout(
            xaxis_title="Lighting Condition",
            yaxis_title="Number of Casualties",
            legend_title="Severity"
        )
        fig_sev_light.for_each_trace(lambda t: t.update(
            name={"num_killed": "Fatalities", "num_hospitalized": "Hospitalized", "num_light_injury": "Light Injuries"}.get(t.name, t.name)
        ))
        st.plotly_chart(fig_sev_light, use_container_width=True)
    else:
        st.info("Severity data not available for lighting analysis.")

with tab2:
    if "atm" in filtered_df.columns and "num_killed" in filtered_df.columns:
        severity_by_weather = filtered_df.groupby("atm").agg({
            "num_killed": "sum",
            "num_hospitalized": "sum" if "num_hospitalized" in filtered_df.columns else lambda x: 0,
            "num_light_injury": "sum" if "num_light_injury" in filtered_df.columns else lambda x: 0
        }).reset_index()
        severity_by_weather["weather"] = severity_by_weather["atm"].map(WEATHER_LABELS)
        
        fig_sev_weather = px.bar(
            severity_by_weather,
            x="weather",
            y=["num_killed", "num_hospitalized", "num_light_injury"],
            title="Casualties by Weather Condition",
            barmode="group",
            color_discrete_map={
                "num_killed": "#d62728",
                "num_hospitalized": "#ff7f0e",
                "num_light_injury": "#2ca02c"
            }
        )
        fig_sev_weather.update_layout(
            xaxis_title="Weather Condition",
            yaxis_title="Number of Casualties",
            legend_title="Severity",
            xaxis_tickangle=-45
        )
        fig_sev_weather.for_each_trace(lambda t: t.update(
            name={"num_killed": "Fatalities", "num_hospitalized": "Hospitalized", "num_light_injury": "Light Injuries"}.get(t.name, t.name)
        ))
        st.plotly_chart(fig_sev_weather, use_container_width=True)
    else:
        st.info("Severity data not available for weather analysis.")
