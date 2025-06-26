"""
Data Preprocessing
------------------
This module takes the cleaned UFC fight data and prepares it for modeling.
It handles missing values, scales numeric features, encodes categorical variables,
and extracts additional features such as weight class and gender.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Plot distribution of weight classes (overall)
    if 'Weightclass' in df.columns:
        plt.figure(figsize=(10, 6))
        df['Weightclass'].value_counts().plot(kind='bar')
        plt.xlabel('Weight Class')
        plt.ylabel('Number of Fights')
        plt.title('Distribution of Weight Classes (Overall)')
        plt.tight_layout()
        plt.savefig('data/weightclass_distribution_overall.png')
        plt.close()

    # Plot distribution of weight classes by gender
    if 'Weightclass' in df.columns and 'Gender' in df.columns:
        plt.figure(figsize=(12, 7))
        sns.countplot(data=df, x='Weightclass', hue='Gender', order=df['Weightclass'].value_counts().index)
        plt.xlabel('Weight Class')
        plt.ylabel('Number of Fights')
        plt.title('Distribution of Weight Classes by Gender')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/weightclass_distribution_by_gender.png')
        plt.close()

    # Save the preprocessed data for model training
    df.to_csv(output_path, index=False)
    print(f"\nPreprocessed data saved as {output_path}")