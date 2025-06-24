"""
Model Training & Feature Analysis
---------------------------------
This module trains a Random Forest model for each weight class and gender combination.
It evaluates model performance, analyzes feature importance, and saves the trained models for later use.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle
import os

def train_and_analyze(input_path):
    # Load the preprocessed data
    df = pd.read_csv(input_path)
    models = {}

    # For each combination of weight class and gender, train a separate model
    for (weight, gender), group in df.groupby(['Weightclass', 'Gender']):
        # Skip groups with too few samples
        if len(group) < 20:
            continue

        # Define target and features
        y = group['Winner?']
        X = group.drop(columns=[
            'Fighter1', 'Fighter2', 'Winner?', 'Winner?.1', 'Fight Method', 'Time', 'Time Format', 'Referee',
            'Finish Details or Judges Scorecard', 'Bout', 'Event Name', 'Location', 'Date',
            'Weightclass', 'Gender'
        ])

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        models[(weight, gender)] = model

        # Evaluate model performance
        y_pred = model.predict(X_test)
        print(f"\nModel for {weight}, {gender}:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Analyze feature importance
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
        importance_df = importance_df[~importance_df['feature'].str.contains('Winner\?', regex=True)]
        top10 = importance_df.sort_values('importance', ascending=False).head(10)
        print("Top 10 features:")
        print(top10)

        # Visualize the top 10 features
        plt.figure(figsize=(8, 5))
        plt.barh(top10['feature'][::-1], top10['importance'][::-1])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 10 Features: {weight}, {gender}')
        plt.tight_layout()
        # Save the plot as a PNG file
        if not os.path.exists("data"):
            os.makedirs("data")
        plt.savefig(f'data/feature_importance_{weight}_{gender}.png')
        plt.close()

    # Save all trained models as a pickle file for later predictions
    if not os.path.exists("models"):
        os.makedirs("models")
    with open("models/random_forest_models.pkl", "wb") as f:
        pickle.dump(models, f)