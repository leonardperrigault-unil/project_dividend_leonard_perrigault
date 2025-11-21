import pytest
import pandas as pd
import numpy as np
from src.models.vif_analysis import calculate_vif


@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    n = 100
    # Create some correlated features
    x1 = np.random.uniform(0, 100, n)
    x2 = x1 * 2 + np.random.normal(0, 5, n)  # Highly correlated with x1
    x3 = np.random.uniform(0, 100, n)  # Independent

    return pd.DataFrame({
        'Life expectancy': np.random.uniform(60, 85, n),
        'Feature1': x1,
        'Feature2': x2,
        'Feature3': x3
    })


@pytest.fixture
def independent_dataframe():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'Life expectancy': np.random.uniform(60, 85, n),
        'A': np.random.uniform(0, 100, n),
        'B': np.random.uniform(0, 100, n),
        'C': np.random.uniform(0, 100, n)
    })


class TestCalculateVIF:
    def test_returns_dataframe(self, sample_dataframe):
        vif_data = calculate_vif(sample_dataframe)
        assert isinstance(vif_data, pd.DataFrame)

    def test_contains_feature_column(self, sample_dataframe):
        vif_data = calculate_vif(sample_dataframe)
        assert 'Feature' in vif_data.columns

    def test_contains_vif_column(self, sample_dataframe):
        vif_data = calculate_vif(sample_dataframe)
        assert 'VIF' in vif_data.columns

    def test_excludes_target_column(self, sample_dataframe):
        vif_data = calculate_vif(sample_dataframe)
        assert 'Life expectancy' not in vif_data['Feature'].values

    def test_vif_values_positive(self, sample_dataframe):
        vif_data = calculate_vif(sample_dataframe)
        assert all(vif_data['VIF'] > 0)

    def test_correlated_features_high_vif(self, sample_dataframe):
        vif_data = calculate_vif(sample_dataframe)
        # Feature1 and Feature2 are correlated, should have higher VIF
        feature1_vif = vif_data[vif_data['Feature'] == 'Feature1']['VIF'].values[0]
        feature2_vif = vif_data[vif_data['Feature'] == 'Feature2']['VIF'].values[0]
        feature3_vif = vif_data[vif_data['Feature'] == 'Feature3']['VIF'].values[0]

        # Correlated features should have higher VIF than independent
        assert feature1_vif > feature3_vif or feature2_vif > feature3_vif

    def test_independent_features_low_vif(self, independent_dataframe):
        vif_data = calculate_vif(independent_dataframe)
        # Independent features should have VIF close to 1
        for vif in vif_data['VIF']:
            assert vif < 5  # Low VIF threshold

    def test_sorted_by_vif_descending(self, sample_dataframe):
        vif_data = calculate_vif(sample_dataframe)
        vif_values = vif_data['VIF'].values
        assert all(vif_values[i] >= vif_values[i+1] for i in range(len(vif_values)-1))
