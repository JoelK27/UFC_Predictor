"""
UFC Fight Outcome Prediction - Workflow
--------------------------------------
Dieses Skript führt alle Schritte von Datenbeschaffung, Preprocessing, Modelltraining bis zur Analyse aus.
"""

import os
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_models
from src.analysis import analyze_feature_importance, plot_feature_importance

# 1. Daten laden und vorverarbeiten
data_path = os.path.join("data", "UFC_Fight_Statistics.csv")
df = load_and_preprocess_data(data_path)

# 2. Modelle trainieren (nach Gewichtsklasse und Geschlecht)
models, feature_cols = train_models(df)

# 3. Analyse der wichtigsten Features
for key, model in models.items():
    importance_df = analyze_feature_importance(model, df, key, feature_cols)
    plot_feature_importance(importance_df, key)
    print(f"\nTop-Features für {key}:")
    print(importance_df.head(10))

# 4. Kurze Interpretation
print("\nFazit: Die wichtigsten Features für den Kampfausgang sind meist ... (hier eigene Analyse ergänzen)")