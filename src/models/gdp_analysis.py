import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.config import RANDOM_SEED, SEPARATOR_WIDTH, CLEANED_DATA_FILE, TARGET_COLUMN, TEST_SIZE

RESULTS_DIR = "results"

def load_cleaned_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def prepare_data(df, mode="with_gdp"):
    """
    mode: "with_gdp", "without_gdp", "gdp_per_capita"
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if TARGET_COLUMN not in numeric_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")

    X = numeric_df.drop(columns=[TARGET_COLUMN])

    if mode == "without_gdp":
        gdp_cols = [col for col in X.columns if 'gdp' in col.lower()]
        if gdp_cols:
            X = X.drop(columns=gdp_cols)
            print(f"Excluded GDP columns: {gdp_cols}")

    elif mode == "gdp_per_capita":
        # Find GDP and Population columns
        gdp_col = None
        pop_col = None
        for col in X.columns:
            if 'gdp' in col.lower() and 'per' not in col.lower():
                gdp_col = col
            if col.lower() == 'population':
                pop_col = col

        if gdp_col and pop_col:
            # Create GDP per capita
            X['GDP_per_capita'] = X[gdp_col] / X[pop_col]
            # Remove original GDP
            X = X.drop(columns=[gdp_col])
            print(f"Created GDP_per_capita from {gdp_col} / {pop_col}")
        else:
            print(f"Could not create GDP per capita (GDP: {gdp_col}, Population: {pop_col})")

    y = numeric_df[TARGET_COLUMN]

    return X, y

def train_and_evaluate(X, y, model_name, model, needs_scaling=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    if needs_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    return {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

def get_models():
    return {
        'Linear': (LinearRegression(), True),
        'Lasso': (Lasso(alpha=0.1, random_state=RANDOM_SEED), True),
        'Ridge': (Ridge(alpha=1.0, random_state=RANDOM_SEED), True),
        'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1), False),
        'XGBoost': (XGBRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1), False)
    }

def plot_comparison(results_with_gdp, results_without_gdp, results_gdp_per_capita):
    models = list(results_with_gdp.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(models))
    width = 0.25

    # R² comparison
    r2_with = [results_with_gdp[m]['r2'] for m in models]
    r2_without = [results_without_gdp[m]['r2'] for m in models]
    r2_percap = [results_gdp_per_capita[m]['r2'] for m in models]
    axes[0].bar(x - width, r2_with, width, label='With GDP', color='steelblue')
    axes[0].bar(x, r2_without, width, label='Without GDP', color='coral')
    axes[0].bar(x + width, r2_percap, width, label='GDP per capita', color='green')
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² Score')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45)
    axes[0].legend()

    # MAE comparison
    mae_with = [results_with_gdp[m]['mae'] for m in models]
    mae_without = [results_without_gdp[m]['mae'] for m in models]
    mae_percap = [results_gdp_per_capita[m]['mae'] for m in models]
    axes[1].bar(x - width, mae_with, width, label='With GDP', color='steelblue')
    axes[1].bar(x, mae_without, width, label='Without GDP', color='coral')
    axes[1].bar(x + width, mae_percap, width, label='GDP per capita', color='green')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45)
    axes[1].legend()

    # RMSE comparison
    rmse_with = [results_with_gdp[m]['rmse'] for m in models]
    rmse_without = [results_without_gdp[m]['rmse'] for m in models]
    rmse_percap = [results_gdp_per_capita[m]['rmse'] for m in models]
    axes[2].bar(x - width, rmse_with, width, label='With GDP', color='steelblue')
    axes[2].bar(x, rmse_without, width, label='Without GDP', color='coral')
    axes[2].bar(x + width, rmse_percap, width, label='GDP per capita', color='green')
    axes[2].set_ylabel('RMSE')
    axes[2].set_title('RMSE')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45)
    axes[2].legend()

    plt.tight_layout()

    gdp_dir = f"{RESULTS_DIR}/gdp_analysis"
    os.makedirs(gdp_dir, exist_ok=True)
    filename = f"{gdp_dir}/gdp_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")
    plt.close()


def main():
    print("=" * SEPARATOR_WIDTH)
    print("GDP DEPENDENCY ANALYSIS")
    print("=" * SEPARATOR_WIDTH)

    df = load_cleaned_data(CLEANED_DATA_FILE)

    # Prepare data for 3 scenarios
    print("\n--- With GDP ---")
    X_with, y = prepare_data(df, mode="with_gdp")
    print(f"Features: {X_with.shape[1]}")

    print("\n--- Without GDP ---")
    X_without, y = prepare_data(df, mode="without_gdp")
    print(f"Features: {X_without.shape[1]}")

    print("\n--- GDP per capita ---")
    X_percap, y = prepare_data(df, mode="gdp_per_capita")
    print(f"Features: {X_percap.shape[1]}")

    models = get_models()

    results_with_gdp = {}
    results_without_gdp = {}
    results_gdp_per_capita = {}

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("TRAINING MODELS")
    print("=" * SEPARATOR_WIDTH)

    for model_name, (model, needs_scaling) in models.items():
        print(f"\n{model_name}...")

        # With GDP
        model_with = type(model)(**model.get_params())
        results_with_gdp[model_name] = train_and_evaluate(X_with, y, model_name, model_with, needs_scaling)

        # Without GDP
        model_without = type(model)(**model.get_params())
        results_without_gdp[model_name] = train_and_evaluate(X_without, y, model_name, model_without, needs_scaling)

        # GDP per capita
        model_percap = type(model)(**model.get_params())
        results_gdp_per_capita[model_name] = train_and_evaluate(X_percap, y, model_name, model_percap, needs_scaling)

    # Display results
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("RESULTS")
    print("=" * SEPARATOR_WIDTH)

    print(f"\n{'Model':<15} {'R² (GDP)':<12} {'R² (no GDP)':<12} {'R² (per cap)':<12}")
    print("-" * 55)

    for model_name in models.keys():
        r2_with = results_with_gdp[model_name]['r2']
        r2_without = results_without_gdp[model_name]['r2']
        r2_percap = results_gdp_per_capita[model_name]['r2']
        print(f"{model_name:<15} {r2_with:<12.4f} {r2_without:<12.4f} {r2_percap:<12.4f}")

    # Find best scenario per model
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("BEST SCENARIO PER MODEL")
    print("=" * SEPARATOR_WIDTH)

    for model_name in models.keys():
        scenarios = {
            'With GDP': results_with_gdp[model_name]['r2'],
            'Without GDP': results_without_gdp[model_name]['r2'],
            'GDP per capita': results_gdp_per_capita[model_name]['r2']
        }
        best = max(scenarios, key=scenarios.get)
        print(f"{model_name}: {best} (R² = {scenarios[best]:.4f})")

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("GENERATING PLOTS")
    print("=" * SEPARATOR_WIDTH)

    plot_comparison(results_with_gdp, results_without_gdp, results_gdp_per_capita)

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("COMPLETED")
    print("=" * SEPARATOR_WIDTH)

if __name__ == "__main__":
    main()
