import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from src.models.linear_regression import prepare_data, train_model, evaluate_model


@pytest.fixture
def sample_dataframe():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'Life expectancy': np.random.uniform(60, 85, n),
        'GDP': np.random.uniform(1000, 50000, n),
        'Population': np.random.uniform(1e6, 1e9, n),
        'Birth Rate': np.random.uniform(8, 40, n),
        'Education': np.random.uniform(0.3, 1.0, n)
    })


class TestPrepareData:
    def test_returns_X_and_y(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        assert X is not None
        assert y is not None

    def test_target_not_in_X(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        assert 'Life expectancy' not in X.columns

    def test_y_is_target_column(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        assert len(y) == len(sample_dataframe)

    def test_X_has_correct_features(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        expected_features = ['GDP', 'Population', 'Birth Rate', 'Education']
        assert list(X.columns) == expected_features


class TestTrainModel:
    def test_returns_model_and_data(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        result = train_model(X, y, regularization="none")
        assert len(result) == 5  # model, X_train, X_test, y_train, y_test

    def test_linear_regression_model(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, _, _, _, _ = train_model(X, y, regularization="none")
        assert isinstance(model, LinearRegression)

    def test_lasso_model(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, _, _, _, _ = train_model(X, y, regularization="l1", optimize_hyperparams=False)
        assert isinstance(model, Lasso)

    def test_ridge_model(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, _, _, _, _ = train_model(X, y, regularization="l2", optimize_hyperparams=False)
        assert isinstance(model, Ridge)

    def test_train_test_split_size(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        _, X_train, X_test, y_train, y_test = train_model(X, y, regularization="none")
        total = len(X_train) + len(X_test)
        assert total == len(X)


class TestEvaluateModel:
    def test_returns_metrics_dict(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, X_train, X_test, y_train, y_test = train_model(X, y, regularization="none")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        assert isinstance(metrics, dict)

    def test_metrics_contains_required_keys(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, X_train, X_test, y_train, y_test = train_model(X, y, regularization="none")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        required_keys = ['train_r2', 'train_mae', 'train_rmse', 'test_r2', 'test_mae', 'test_rmse']
        for key in required_keys:
            assert key in metrics

    def test_metrics_are_numeric(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, X_train, X_test, y_train, y_test = train_model(X, y, regularization="none")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        for value in metrics.values():
            assert isinstance(value, (int, float))

    def test_r2_in_valid_range(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, X_train, X_test, y_train, y_test = train_model(X, y, regularization="none")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        # R2 can be negative for bad models, but should be <= 1
        assert metrics['train_r2'] <= 1
        assert metrics['test_r2'] <= 1

    def test_mae_rmse_positive(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, X_train, X_test, y_train, y_test = train_model(X, y, regularization="none")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        assert metrics['train_mae'] >= 0
        assert metrics['test_mae'] >= 0
        assert metrics['train_rmse'] >= 0
        assert metrics['test_rmse'] >= 0
