"""
Data Preprocessing
------------------
This module takes the cleaned UFC fight data and prepares it for modeling.
It handles missing values, scales numeric features, encodes categorical variables,
and extracts additional features such as weight class and gender.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path, output_path):
    # Load the cleaned data
    df = pd.read_csv(input_path)

    # Fill missing numeric values with 0
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Scale numeric features for better model performance
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode the target variable if present
    if 'Winner?' in df.columns:
        encoder = LabelEncoder()
        df['Winner?'] = encoder.fit_transform(df['Winner?'])

    # Extract weight class and gender from the 'Bout' column
    df['Weightclass'] = df['Bout'].str.extract(r'(\w+weight)')
    df['Gender'] = df['Bout'].apply(lambda x: 'Women' if 'Women' in x else 'Men')

    # Save the preprocessed data for model training
    df.to_csv(output_path, index=False)
    print(f"\nPreprocessed data saved as {output_path}")