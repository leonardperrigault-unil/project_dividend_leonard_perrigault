import pytest
import pandas as pd
import numpy as np
from src.data_cleaning.clean import clean_data, explore_data


@pytest.fixture
def df_with_string_numbers():
    """DataFrame with string-formatted numbers ($, %, commas)"""
    return pd.DataFrame({
        'Country': ['A', 'B', 'C', 'D'],
        'Life expectancy': [75.0, 80.0, 70.0, 65.0],
        'GDP': ['$1,000', '$2,500', '$3,000', '$4,000'],
        'Tax rate': ['15%', '20%', '25%', '30%'],
        'Population': ['1,000,000', '2,000,000', '3,000,000', '4,000,000']
    })


@pytest.fixture
def df_with_missing_values():
    """DataFrame with missing values"""
    return pd.DataFrame({
        'Country': ['A', 'B', 'C', 'D', 'E'],
        'Life expectancy': [75.0, 80.0, np.nan, 65.0, 70.0],
        'GDP': [1000, np.nan, 3000, 4000, 5000],
        'Population': [100, 200, 300, np.nan, 500]
    })


@pytest.fixture
def df_with_duplicates():
    """DataFrame with duplicate rows"""
    return pd.DataFrame({
        'Country': ['A', 'A', 'B', 'C'],
        'Life expectancy': [75.0, 75.0, 80.0, 70.0],
        'GDP': [1000, 1000, 2000, 3000]
    })


@pytest.fixture
def df_with_high_missing():
    """DataFrame with column having >50% missing"""
    return pd.DataFrame({
        'Country': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'Life expectancy': [75, 80, 70, 65, 78, 82, 68, 72, 77, 79],
        'GDP': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        'Sparse_col': [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    })


class TestCleanData:
    def test_converts_string_numbers(self, df_with_string_numbers):
        """Test that $, %, commas are removed and converted to numeric"""
        result = clean_data(df_with_string_numbers)

        assert result['GDP'].dtype in ['float64', 'int64']
        assert result['Tax rate'].dtype in ['float64', 'int64']
        assert result['Population'].dtype in ['float64', 'int64']

    def test_drops_rows_with_missing_life_expectancy(self, df_with_missing_values):
        """Test that rows with missing Life expectancy are dropped"""
        result = clean_data(df_with_missing_values)

        assert result['Life expectancy'].isnull().sum() == 0
        assert len(result) == 4  # One row removed

    def test_fills_missing_numeric_with_median(self, df_with_missing_values):
        """Test that missing numeric values are filled with median"""
        result = clean_data(df_with_missing_values)

        # After dropping the row with missing life expectancy, check other columns
        assert result['GDP'].isnull().sum() == 0
        assert result['Population'].isnull().sum() == 0

    def test_removes_duplicates(self, df_with_duplicates):
        """Test that duplicate rows are removed"""
        result = clean_data(df_with_duplicates)

        assert result.duplicated().sum() == 0
        assert len(result) == 3  # One duplicate removed

    def test_drops_columns_with_high_missing(self, df_with_high_missing):
        """Test that columns with >50% missing are dropped"""
        result = clean_data(df_with_high_missing)

        assert 'Sparse_col' not in result.columns

    def test_preserves_country_column(self, df_with_string_numbers):
        """Test that Country column is preserved as string"""
        result = clean_data(df_with_string_numbers)

        assert 'Country' in result.columns
        assert result['Country'].dtype == 'object'

    def test_output_has_no_missing_values(self, df_with_missing_values):
        """Test that cleaned data has no missing values in numeric columns"""
        result = clean_data(df_with_missing_values)
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            assert result[col].isnull().sum() == 0

    def test_returns_dataframe(self, df_with_string_numbers):
        """Test that clean_data returns a DataFrame"""
        result = clean_data(df_with_string_numbers)

        assert isinstance(result, pd.DataFrame)


class TestExploreData:
    def test_returns_missing_df(self, df_with_missing_values):
        """Test that explore_data returns missing values summary"""
        result = explore_data(df_with_missing_values)

        assert isinstance(result, pd.DataFrame)
        assert 'Missing_Count' in result.columns
        assert 'Percentage' in result.columns

    def test_only_shows_columns_with_missing(self, df_with_missing_values):
        """Test that only columns with missing values are shown"""
        result = explore_data(df_with_missing_values)

        # All columns in result should have missing values
        for count in result['Missing_Count']:
            assert count > 0
