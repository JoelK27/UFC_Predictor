"""
Data Exploration & Cleaning
---------------------------
This module loads the raw UFC fight data, explores its structure, checks for missing values,
removes unnecessary columns, and saves a cleaned version for further processing.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(input_path, output_path):
    # Load the raw data
    df = pd.read_csv(input_path)

    # Show the first few rows to get an overview
    print("First rows of the dataset:")
    print(df.head())

    # Display info about columns and data types
    print("\nDataFrame info:")
    print(df.info())

    # Show descriptive statistics for all columns
    print("\nDescriptive statistics:")
    print(df.describe(include='all'))

    # Check for missing values in each column
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Analyze the target variable distribution if present
    if 'Winner?' in df.columns:
        print("\nDistribution of the target variable (Winner?):")
        print(df['Winner?'].value_counts())
        sns.countplot(x='Winner?', data=df)
        plt.title('Distribution of Target Variable (Winner?)')
        plt.savefig('data/target_distribution.png')
        plt.close()


    # Remove columns that are completely empty
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    df = df.drop(columns=empty_cols)
    print(f"\nRemoved completely empty columns: {empty_cols}")

    # Attempt to convert object columns to numeric where possible
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass  # If conversion fails, keep as string

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Example: Detect outliers in 'Sig. Str.' column if it exists
    if 'Sig. Str.' in df.columns:
        sns.boxplot(x=df['Sig. Str.'])
        plt.title('Boxplot Significant Strikes')
        plt.savefig('data/sig_str_boxplot.png')
        plt.close()

    # Save the cleaned data for the next step
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved as {output_path}")