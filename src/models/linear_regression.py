import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.config import RANDOM_SEED, SEPARATOR_WIDTH, CLEANED_DATA_FILE, TARGET_COLUMN, TEST_SIZE

def load_cleaned_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def prepare_data(df):
    # Drop non-numeric columns and target
    numeric_df = df.select_dtypes(include=[np.number])

    if TARGET_COLUMN not in numeric_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")

    X = numeric_df.drop(columns=[TARGET_COLUMN])
    y = numeric_df[TARGET_COLUMN]

    print(f"\nFeatures: {X.shape[1]} columns")
    print(f"Target: {TARGET_COLUMN}")

    return X, y

def train_model(X, y, use_l1=False, alpha=1.0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    if use_l1:
        print(f"\nTraining Lasso (L1) with alpha={alpha}...")
        model = Lasso(alpha=alpha, random_state=RANDOM_SEED)
    else:
        print("\nTraining Linear Regression...")
        model = LinearRegression()

    model.fit(X_train_scaled, y_train)

    return model, X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Linear Regression"):
    # Train predictions
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # Test predictions
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("\n" + "=" * SEPARATOR_WIDTH)
    print(f"MODEL EVALUATION - {model_name}")
    print("=" * SEPARATOR_WIDTH)

    print("\nTrain Metrics:")
    print(f"  R²    : {train_r2:.4f}")
    print(f"  MAE   : {train_mae:.4f}")
    print(f"  RMSE  : {train_rmse:.4f}")

    print("\nTest Metrics:")
    print(f"  R²    : {test_r2:.4f}")
    print(f"  MAE   : {test_mae:.4f}")
    print(f"  RMSE  : {test_rmse:.4f}")

    # Feature importance based on coefficients
    print("\nFeature Importance (Top 10):")
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    })
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['Feature']:40s} : {row['Coefficient']:8.4f}")

    return {
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse
    }

def main(use_l1=False, alpha=1.0):
    df = load_cleaned_data(CLEANED_DATA_FILE)

    X, y = prepare_data(df)

    model, X_train, X_test, y_train, y_test = train_model(X, y, use_l1=use_l1, alpha=alpha)

    model_name = f"Lasso (L1, alpha={alpha})" if use_l1 else "Linear Regression"
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, model_name=model_name)

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("TRAINING COMPLETED")
    print("=" * SEPARATOR_WIDTH)

if __name__ == "__main__":
    main()
