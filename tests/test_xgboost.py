import pytest
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from src.models.xgboost_model import prepare_data, train_model, evaluate_model


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

    def test_correct_number_of_samples(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        assert len(X) == len(y) == len(sample_dataframe)


class TestTrainModel:
    def test_returns_model_and_data(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        result = train_model(X, y, optimize_hyperparams=False)
        assert len(result) == 5

    def test_returns_xgboost_model(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, _, _, _, _ = train_model(X, y, optimize_hyperparams=False)
        assert isinstance(model, XGBRegressor)

    def test_model_has_feature_importances(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, _, _, _, _ = train_model(X, y, optimize_hyperparams=False)
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X.shape[1]


class TestEvaluateModel:
    def test_returns_metrics_dict(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, X_train, X_test, y_train, y_test = train_model(X, y, optimize_hyperparams=False)
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        assert isinstance(metrics, dict)

    def test_contains_all_metrics(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, X_train, X_test, y_train, y_test = train_model(X, y, optimize_hyperparams=False)
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        required_keys = ['train_r2', 'train_mae', 'train_rmse', 'test_r2', 'test_mae', 'test_rmse']
        for key in required_keys:
            assert key in metrics

    def test_metrics_are_numeric(self, sample_dataframe):
        X, y = prepare_data(sample_dataframe)
        model, X_train, X_test, y_train, y_test = train_model(X, y, optimize_hyperparams=False)
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        for value in metrics.values():
            assert isinstance(value, (int, float))
