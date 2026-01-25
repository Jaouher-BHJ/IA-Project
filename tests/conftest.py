# Shared fixtures and generators for tests
"""
Pytest fixtures and Hypothesis strategies for property-based testing.
"""
import pytest
import pandas as pd
from hypothesis import strategies as st


# Hypothesis strategies for generating test data
hrmn_strategy = st.integers(min_value=0, max_value=2359).filter(
    lambda x: (x % 100) < 60  # Valid minutes (0-59)
)

collision_type_strategy = st.integers(min_value=1, max_value=7)

lighting_strategy = st.integers(min_value=1, max_value=5)

weather_strategy = st.integers(min_value=1, max_value=9)

location_strategy = st.integers(min_value=1, max_value=2)

intersection_strategy = st.integers(min_value=1, max_value=9)


@pytest.fixture
def sample_caracteristiques_df():
    """Create a sample caracteristiques DataFrame for testing."""
    return pd.DataFrame({
        'Num_Acc': [1, 2, 3],
        'an': [21, 21, 22],
        'mois': [1, 6, 12],
        'jour': [15, 20, 25],
        'hrmn': [830, 1430, 2200],
        'lum': [1, 2, 3],
        'agg': [1, 2, 1],
        'int': [1, 2, 3],
        'atm': [1, 2, 3],
        'col': [1, 2, 3],
        'dep': ['75', '13', '69'],
    })


@pytest.fixture
def sample_usagers_df():
    """Create a sample usagers DataFrame for testing."""
    return pd.DataFrame({
        'Num_Acc': [1, 1, 2, 3, 3, 3],
        'catu': [1, 2, 1, 1, 2, 3],
        'grav': [1, 2, 3, 4, 1, 2],
        'sexe': [1, 2, 1, 2, 1, 2],
    })
