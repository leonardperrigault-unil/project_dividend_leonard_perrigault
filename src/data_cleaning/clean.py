import pandas as pd
import numpy as np

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_data(filepath):
    """Load the dataset from CSV file"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df

def explore_data(df):
    """Initial exploration of the dataset"""
    print("="*80)
    print("INITIAL DATA EXPLORATION")
    print("="*80)

    print("\nFirst few rows:")
    print(df.head())

    print("\n\nDataset Info:")
    print(df.info())

    print("\n\nBasic Statistics:")
    print(df.describe())

    print("\n\nMissing Values Summary:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Percentage', ascending=False)
    print(missing_df)

    return missing_df

def clean_data(df):
    """Main data cleaning function"""
    print("\n" + "="*80)
    print("STARTING DATA CLEANING")
    print("="*80)

    df_clean = df.copy()

    # 1. Handle missing values in Life expectancy
    print("\n1. Handling missing values in Life expectancy...")
    if 'Life expectancy' in df_clean.columns:
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['Life expectancy'])
        removed = initial_rows - len(df_clean)
        print(f"   Removed {removed} rows with missing life expectancy values")

    # 2. Identify numeric and categorical columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

    print(f"\n2. Identified {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")

    # 3. Handle missing values in numeric columns
    print("\n3. Handling missing values in numeric columns...")
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100

            # If more than 50% missing, consider dropping the column
            if missing_pct > 50:
                print(f"   Dropping column '{col}' ({missing_pct:.1f}% missing)")
                df_clean = df_clean.drop(columns=[col])
            else:
                # Use median imputation for numeric columns
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"   Filled '{col}' with median value ({median_val:.2f})")

    # 4. Handle missing values in categorical columns
    print("\n4. Handling missing values in categorical columns...")
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100

            if missing_pct > 50:
                print(f"   Dropping column '{col}' ({missing_pct:.1f}% missing)")
                df_clean = df_clean.drop(columns=[col])
            else:
                # Use mode (most frequent) for categorical columns
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_val)
                print(f"   Filled '{col}' with mode value")
    
    # TODO: Handle outliers

    print(f"\n6. Final cleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

    return df_clean

def save_cleaned_data(df, output_path):
    """Save the cleaned dataset"""
    print("\n" + "="*80)
    print("SAVING CLEANED DATA")
    print("="*80)

    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")

def main(input_file=None, output_file=None):
    """Main execution function"""
    # File paths with defaults
    if input_file is None:
        input_file = 'data/world-data-2023.csv'
    if output_file is None:
        output_file = 'data/cleaned_world_data.csv'

    # Load data
    df = load_data(input_file)

    # Explore data
    missing_df = explore_data(df)

    # Clean data
    df_clean = clean_data(df)

    # Save cleaned data
    save_cleaned_data(df_clean, output_file)

    print("\n" + "="*80)
    print("DATA CLEANING COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()
