import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.config import RANDOM_SEED, SEPARATOR_WIDTH, CLEANED_DATA_FILE, TARGET_COLUMN, TEST_SIZE

MODELS_DIR = "saved_models"
RESULTS_DIR = "results"

def load_model(model_type):
    filename = f"{MODELS_DIR}/{model_type}_model.pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model not found: {filename}")

    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Loaded model: {filename}")
    return model

def load_and_prepare_data(scale_data=True):
    df = pd.read_csv(CLEANED_DATA_FILE)
    numeric_df = df.select_dtypes(include=[np.number])

    X = numeric_df.drop(columns=[TARGET_COLUMN])
    y = numeric_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    if scale_data:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    else:
        return X_train.values, X_test.values, y_train, y_test, X.columns

def plot_actual_vs_predicted(y_test, y_pred, model_type):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Life Expectancy')
    plt.ylabel('Predicted Life Expectancy')
    plt.title(f'Actual vs Predicted - {model_type.capitalize()}')
    plt.grid(True, alpha=0.3)

    model_dir = f"{RESULTS_DIR}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{model_dir}/actual_vs_predicted.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

def plot_residuals(y_test, y_pred, model_type):
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Residuals scatter
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Life Expectancy')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    axes[0].grid(True, alpha=0.3)

    # Residuals distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    model_dir = f"{RESULTS_DIR}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{model_dir}/residuals.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

def plot_feature_importance(model, feature_names, model_type):
    # Check if model has coef_ (linear models) or feature_importances_ (tree models)
    if hasattr(model, 'coef_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.coef_,
            'Abs_Importance': np.abs(model.coef_)
        })
        xlabel = 'Coefficient Value'
        value_col = 'Importance'
    elif hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        })
        feature_importance['Abs_Importance'] = feature_importance['Importance']
        xlabel = 'Feature Importance'
        value_col = 'Importance'
    else:
        print("Model does not have feature importance information")
        return

    feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance[value_col])
    plt.xlabel(xlabel)
    plt.ylabel('Feature')
    plt.title(f'Top 10 Feature Importance - {model_type.capitalize()}')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    model_dir = f"{RESULTS_DIR}/{model_type}"
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{model_dir}/feature_importance.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

def get_available_models():
    if not os.path.exists(MODELS_DIR):
        return []

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    available_models = []

    for f in model_files:
        model_type = f.replace('_model.pkl', '')
        available_models.append(model_type)

    return available_models

def test_model_interactive(model_type):
    print("=" * SEPARATOR_WIDTH)
    print(f"TESTING MODEL: {model_type.upper()}")
    print("=" * SEPARATOR_WIDTH)

    model = load_model(model_type)

    # Tree-based models don't need scaling
    needs_scaling = model_type not in ['random_forest', 'xgboost']
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(scale_data=needs_scaling)

    print(f"\nTest set: {X_test.shape[0]} samples")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("TEST METRICS")
    print("=" * SEPARATOR_WIDTH)
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    # Generate plots
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("GENERATING PLOTS")
    print("=" * SEPARATOR_WIDTH)

    plot_actual_vs_predicted(y_test, y_pred, model_type)
    plot_residuals(y_test, y_pred, model_type)
    plot_feature_importance(model, feature_names, model_type)

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("TESTING COMPLETED")
    print("=" * SEPARATOR_WIDTH)

def main():
    available_models = get_available_models()

    if not available_models:
        print("No saved models found. Please train a model first.")
        return

    print("=" * SEPARATOR_WIDTH)
    print("AVAILABLE SAVED MODELS")
    print("=" * SEPARATOR_WIDTH)

    for idx, model in enumerate(available_models, 1):
        print(f"{idx}. {model.capitalize()}")

    choice = input(f"\nSelect model to test (1-{len(available_models)}): ").strip()

    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(available_models):
            model_type = available_models[choice_idx]
            test_model_interactive(model_type)
        else:
            print(f"Invalid choice. Please select 1-{len(available_models)}.")
    except ValueError:
        print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()
