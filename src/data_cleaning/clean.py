import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df

def explore_data(df):
    print("INITIAL DATA EXPLORATION")
    print("========================================")

    print("\nFirst few rows:")
    print(df.head())

    print("\n\nDataset Info:")
    print(df.info())

    print("\n\nBasic Statistics:")
    print(df.describe())

    # TODO: Add missing values analysis
    # TODO: Add visualization

def clean_data(df):
    print("\nSTARTING DATA CLEANING")
    print("===============================")

    df_clean = df.copy()

    # TODO: Handle missing values
    # TODO: Remove duplicates
    # TODO: Handle outliers

    print(f"\nCurrent dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

    return df_clean

def main():
    input_file = 'data/world-data-2023.csv'
    output_file = 'data/cleaned_world_data.csv'

    df = load_data(input_file)

    explore_data(df)

    df_clean = clean_data(df)

    
    print("\nDATA EXPLORATION COMPLETED")
    print("===========================")

if __name__ == "__main__":
    main()
