import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.config import SEPARATOR_WIDTH, CLEANED_DATA_FILE, TARGET_COLUMN

RESULTS_DIR = "results"

def load_cleaned_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def calculate_vif(df):
    numeric_df = df.select_dtypes(include=[np.number])

    if TARGET_COLUMN in numeric_df.columns:
        X = numeric_df.drop(columns=[TARGET_COLUMN])
    else:
        X = numeric_df

    print(f"\nCalculating VIF for {X.shape[1]} features...")

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Sort by VIF descending
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

    return vif_data

def plot_vif(vif_data):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color code by VIF level
    colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in vif_data['VIF']]

    ax.barh(vif_data['Feature'], vif_data['VIF'], color=colors)
    ax.set_xlabel('VIF Score')
    ax.set_ylabel('Feature')
    ax.set_title('Variance Inflation Factor Analysis')
    ax.invert_yaxis()

    # Add reference lines
    ax.axvline(x=5, color='orange', linestyle='--', alpha=0.5, label='VIF = 5')
    ax.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='VIF = 10')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    vif_dir = f"{RESULTS_DIR}/vif_analysis"
    os.makedirs(vif_dir, exist_ok=True)
    filename = f"{vif_dir}/vif_scores.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")
    plt.close()

def main():
    print("=" * SEPARATOR_WIDTH)
    print("VIF ANALYSIS")
    print("=" * SEPARATOR_WIDTH)

    df = load_cleaned_data(CLEANED_DATA_FILE)

    vif_data = calculate_vif(df)

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("RESULTS")
    print("=" * SEPARATOR_WIDTH)

    print(f"\n{'Feature':<40} {'VIF':<10} {'Status':<20}")
    print("-" * 70)

    for idx, row in vif_data.iterrows():
        feature = row['Feature']
        vif = row['VIF']

        if vif > 10:
            status = "High"
        elif vif > 5:
            status = "Moderate"
        else:
            status = "Low"

        print(f"{feature:<40} {vif:<10.2f} {status:<20}")

    # Summary
    high_vif = vif_data[vif_data['VIF'] > 10]
    moderate_vif = vif_data[(vif_data['VIF'] > 5) & (vif_data['VIF'] <= 10)]

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("SUMMARY")
    print("=" * SEPARATOR_WIDTH)
    print(f"High multicollinearity (VIF > 10): {len(high_vif)} features")
    if len(high_vif) > 0:
        print(f"  {', '.join(high_vif['Feature'].tolist())}")

    print(f"\nModerate multicollinearity (VIF 5-10): {len(moderate_vif)} features")
    if len(moderate_vif) > 0:
        print(f"  {', '.join(moderate_vif['Feature'].tolist())}")

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("GENERATING PLOT")
    print("=" * SEPARATOR_WIDTH)

    plot_vif(vif_data)

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("COMPLETED")
    print("=" * SEPARATOR_WIDTH)

if __name__ == "__main__":
    main()
