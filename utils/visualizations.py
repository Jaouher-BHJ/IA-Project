# Visualization utilities for Road Accident Analysis
"""
Functions for generating charts and visualizations for the dashboard.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


def create_temporal_chart(
    df: pd.DataFrame,
    time_column: str,
    title: str,
    x_label: str = None
) -> go.Figure:
    """
    Create a bar chart showing accident counts over time.
    
    Args:
        df: DataFrame with accident data
        time_column: Column name for time grouping (e.g., 'hour', 'day_of_week')
        title: Chart title
        x_label: Optional x-axis label
        
    Returns:
        Plotly figure object
    """
    counts = df[time_column].value_counts().sort_index()
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        title=title,
        labels={'x': x_label or time_column, 'y': 'Number of Accidents'}
    )
    return fig


def create_categorical_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    labels_map: Optional[dict] = None
) -> go.Figure:
    """
    Create a bar chart for categorical variable distribution.
    
    Args:
        df: DataFrame with accident data
        column: Column name for the categorical variable
        title: Chart title
        labels_map: Optional mapping from codes to labels
        
    Returns:
        Plotly figure object
    """
    counts = df[column].value_counts().sort_index()
    
    if labels_map:
        x_labels = [labels_map.get(idx, str(idx)) for idx in counts.index]
    else:
        x_labels = counts.index
    
    fig = px.bar(
        x=x_labels,
        y=counts.values,
        title=title,
        labels={'x': column, 'y': 'Number of Accidents'}
    )
    return fig


def create_pie_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    labels_map: Optional[dict] = None
) -> go.Figure:
    """
    Create a pie chart for categorical variable distribution.
    
    Args:
        df: DataFrame with accident data
        column: Column name for the categorical variable
        title: Chart title
        labels_map: Optional mapping from codes to labels
        
    Returns:
        Plotly figure object
    """
    counts = df[column].value_counts()
    
    if labels_map:
        names = [labels_map.get(idx, str(idx)) for idx in counts.index]
    else:
        names = [str(idx) for idx in counts.index]
    
    fig = px.pie(
        values=counts.values,
        names=names,
        title=title
    )
    return fig
