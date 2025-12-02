# Life Expectancy Prediction - ML Project

Predicting and explaining life expectancy using interpretable machine learning models on global country data.

## Overview

This project analyzes the **Global Country Information Dataset 2023** (187 countries, 35 features) to predict life expectancy using five different machine learning algorithms. The focus is on model interpretability through SHAP analysis, feature importance comparison, and multicollinearity diagnosis.

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the interactive menu:

```bash
python main.py
```

## Menu Options

### 1. Clean Data
**What it does:**
- Loads raw data from `data/world-data-2023.csv`
- Converts string-formatted numbers to numeric types
- Handles missing values (median for numeric, mode for categorical)
- Removes duplicates and detects outliers
- Saves cleaned data to `data/cleaned_world_data.csv`

**Run first:** Yes, this is mandatory before training any models.

---

### 2. Train Linear Regression
**What it does:**
- Trains standard Linear Regression (no regularization)
- Uses StandardScaler for feature normalization
- Saves model to `saved_models/linear_model.pkl`
- Prints R², MAE, RMSE on test set

**Prerequisites:** Option 1 (Clean Data)

---

### 3. Train Lasso (L1)
**What it does:**
- Trains Lasso regression with L1 regularization
- Hyperparameter optimization via GridSearchCV (5-fold CV)
- Tests 50 alpha values from 10^-4 to 10^2
- Saves best model to `saved_models/lasso_model.pkl`

**Prerequisites:** Option 1 (Clean Data)

**Note:** L1 regularization performs feature selection (some coefficients → 0)

---

### 4. Train Ridge (L2)
**What it does:**
- Trains Ridge regression with L2 regularization
- Hyperparameter optimization via GridSearchCV (5-fold CV)
- Tests 50 alpha values from 10^-4 to 10^2
- Saves best model to `saved_models/ridge_model.pkl`

**Prerequisites:** Option 1 (Clean Data)

**Note:** L2 regularization shrinks coefficients but keeps all features

---

### 5. Train Random Forest
**What it does:**
- Trains Random Forest Regressor
- Hyperparameter tuning: n_estimators, max_depth, min_samples_split, min_samples_leaf
- Uses GridSearchCV with 5-fold cross-validation
- Saves best model to `saved_models/random_forest_model.pkl`

**Prerequisites:** Option 1 (Clean Data)

**Note:** No feature scaling needed (tree-based model)

---

### 6. Train XGBoost
**What it does:**
- Trains XGBoost Regressor
- Hyperparameter tuning: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- Uses GridSearchCV with 5-fold cross-validation
- Saves best model to `saved_models/xgboost_model.pkl`

**Prerequisites:** Option 1 (Clean Data)

**Note:** Gradient boosting with sequential tree building

---

### 7. Test Model
**What it does:**
- Prompts you to select a trained model
- Generates three visualizations:
  - **Actual vs Predicted:** Scatter plot with ideal fit line
  - **Residuals:** Residual plot + distribution histogram
  - **Feature Importance:** Top 10 most important features
- Saves plots to `results/{model_name}/`

**Prerequisites:** At least one model trained (options 2-6)

**Example output:**
- `results/random_forest/actual_vs_predicted.png`
- `results/random_forest/residuals.png`
- `results/random_forest/feature_importance.png`

---

### 8. Compare All Models
**What it does:**
- Evaluates all trained models on the test set
- Creates comparison visualizations:
  - **Metrics Comparison:** R², MAE, RMSE side-by-side (best highlighted in green)
  - **Feature Importance Comparison:** Top 10 features for each model
- Prints summary table sorted by R²
- Saves to `results/comparison/`

**Prerequisites:** Multiple models trained (options 2-6)

**Output:**
- `results/comparison/metrics_comparison.png`
- `results/comparison/feature_importance_comparison.png`

---

### 9. SHAP Analysis
**What it does:**
- Prompts you to select a trained model
- Calculates SHAP values for model interpretability
- Generates visualizations:
  - **SHAP Summary:** Beeswarm plot showing feature impact distribution
  - **SHAP Importance:** Bar chart of mean absolute SHAP values
  - **Dependence Plots:** Scatter plots for top 3 features
- Saves to `results/{model_name}/`

**Prerequisites:** At least one model trained (options 2-6)

**Best for:** Random Forest or XGBoost (TreeExplainer is efficient)

**Example output:**
- `results/random_forest/shap_summary.png`
- `results/random_forest/shap_importance.png`
- `results/random_forest/shap_dependence_Infant mortality.png`

---

### 10. VIF Analysis
**What it does:**
- Calculates Variance Inflation Factor for all numeric features
- Detects multicollinearity (VIF > 10 = high correlation)
- Creates color-coded visualization:
  - Green: VIF < 5 (low multicollinearity)
  - Orange: VIF 5-10 (moderate)
  - Red: VIF > 10 (high)
- Saves to `results/vif_analysis/vif_scores.png`

**Prerequisites:** Option 1 (Clean Data)

**Note:** Explains why Linear Regression performs poorly (multicollinearity)

---

### 11. GDP Dependency Analysis
**What it does:**
- Trains all 5 models under 3 conditions:
  1. **With GDP:** All features including raw GDP
  2. **Without GDP:** GDP columns removed
  3. **GDP per capita:** GDP replaced with GDP/Population
- Creates comparison plots showing performance across scenarios
- Saves to `results/gdp_analysis/gdp_comparison.png`

**Prerequisites:** Option 1 (Clean Data)

**Purpose:** Investigate whether GDP is essential for prediction

---

### 0. Exit
Exits the program.

---

## Recommended Workflow

### Quick Start (Essential Analysis)
```
1. Clean data                    (option 1)
2. Train Random Forest           (option 5)  - Best model
3. Train Ridge                   (option 4)  - Best linear model
4. Compare models                (option 8)
5. Test Random Forest            (option 7)
6. SHAP Analysis on RF           (option 9)
7. VIF Analysis                  (option 10)
```

### Complete Analysis
```
1. Clean data                    (option 1)
2. Train Linear Regression       (option 2)
3. Train Lasso                   (option 3)
4. Train Ridge                   (option 4)
5. Train Random Forest           (option 5)
6. Train XGBoost                 (option 6)
7. Compare All Models            (option 8)
8. VIF Analysis                  (option 10) - Explains multicollinearity
9. Test Random Forest            (option 7)  - Visualize best model
10. SHAP Analysis on RF          (option 9)  - Interpret predictions
11. GDP Dependency Analysis      (option 11) - GDP investigation
```

## Project Structure

```
project_predicting_leonard_perrigault/
├── data/
│   ├── world-data-2023.csv              # Raw data (Kaggle)
│   └── cleaned_world_data.csv           # Cleaned data
├── src/
│   ├── config.py                        # Configuration (random seed, paths)
│   ├── data_cleaning/
│   │   └── clean.py                     # Data preprocessing
│   └── models/
│       ├── linear_regression.py         # Linear/Lasso/Ridge
│       ├── random_forest.py             # Random Forest
│       ├── xgboost_model.py             # XGBoost
│       ├── test_model.py                # Model visualization
│       ├── compare_models.py            # Cross-model comparison
│       ├── shap_analysis.py             # SHAP interpretability
│       ├── vif_analysis.py              # Multicollinearity analysis
│       └── gdp_analysis.py              # GDP dependency study
├── saved_models/                        # Trained models (.pkl)
├── results/                             # Visualizations (.png)
│   ├── comparison/
│   ├── gdp_analysis/
│   ├── vif_analysis/
│   ├── random_forest/
│   └── ridge/
├── tests/                               # Pytest test suite
├── main.py                              # Interactive CLI menu
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── AI_USAGE.md                          # AI tools usage documentation
```

## Key Results

| Model          | R²   | MAE (years) | RMSE (years) |
|----------------|------|-------------|--------------|
| Random Forest  | 0.92 | 1.80        | 2.30         |
| XGBoost        | 0.90 | 2.05        | 2.45         |
| Lasso (L1)     | 0.87 | 1.95        | 2.55         |
| Ridge (L2)     | 0.86 | 2.00        | 2.70         |
| Linear         | 0.66 | 2.50        | 4.05         |

**Key Findings:**
- **Infant mortality** is the most important predictor across all models
- **GDP per capita** outperforms raw GDP
- Tree-based models are robust to multicollinearity
- Regularization (Lasso/Ridge) dramatically improves linear models

## Testing

Run tests with coverage:

```bash
pytest --cov=src tests/
```

## Configuration

Edit `src/config.py` to modify:
- `RANDOM_SEED`: Random seed for reproducibility (default: 2904)
- `TEST_SIZE`: Train/test split ratio (default: 0.2)
- `TARGET_COLUMN`: Prediction target (default: "Life expectancy")

## Requirements

- Python 3.10+
- numpy 2.3.4
- pandas 2.3.3
- scikit-learn 1.7.2
- xgboost 3.1.1
- shap 0.50.0
- matplotlib 3.10.7
- seaborn 0.13.2
- statsmodels 0.14.5

## Author

Leonard Perrigault

## License

This project is for educational purposes (Advanced Programming 2025 - HEC Lausanne / UNIL).
