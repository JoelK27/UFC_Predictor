import pandas as pd
import matplotlib.pyplot as plt

def analyze_feature_importance(model, df, key, feature_cols):
    """
    Gibt ein DataFrame mit Feature Importance für ein Modell zurück.
    """
    weight, gender = key
    group = df[(df['Weightclass'] == weight) & (df['Gender'] == gender)]
    X_group = group[feature_cols]
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'feature': X_group.columns, 'importance': importances})
    # Entferne Winner-Spalten, falls sie doch auftauchen
    importance_df = importance_df[~importance_df['feature'].str.contains('Winner\?', regex=True)]
    return importance_df.sort_values('importance', ascending=False)

def plot_feature_importance(importance_df, key):
    """
    Plottet die Top 10 wichtigsten Features.
    """
    top10 = importance_df.head(10)
    plt.figure(figsize=(8, 5))
    plt.barh(top10['feature'][::-1], top10['importance'][::-1])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 10 Features: {key[0]}, {key[1]}')
    plt.tight_layout()
    plt.show()