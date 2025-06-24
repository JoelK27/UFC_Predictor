"""
Fight Prediction Script
----------------------
This script allows you to interactively enter two fighter names, a weight class, and gender.
It then predicts the winner using the trained model and displays the top 10 most important features.
"""

import pandas as pd
import pickle
import os

def get_fighter_stats(df, fighter_name, feature_cols, fighter_col_prefix):
    """
    Retrieve the average feature values for a given fighter.
    Looks for the fighter as either Fighter1 or Fighter2 and averages their stats.
    """
    if fighter_col_prefix == "F1":
        rows = df[df['Fighter1'] == fighter_name]
        prefix = "F1"
    else:
        rows = df[df['Fighter2'] == fighter_name]
        prefix = "F2"
    stats = rows[[col for col in feature_cols if prefix in col]].mean()
    stats = stats.reindex([col for col in feature_cols if prefix in col], fill_value=0)
    return stats

def predict_fight(fighter1, fighter2, weightclass, gender, df, models, feature_cols):
    """
    Build a feature vector for a new fight using the stats of both fighters,
    predict the winner, and display the top 10 most important features.
    """
    stats_f1 = get_fighter_stats(df, fighter1, feature_cols, "F1")
    stats_f2 = get_fighter_stats(df, fighter2, feature_cols, "F2")
    new_fight = {}
    for col in feature_cols:
        if "F1" in col:
            new_fight[col] = stats_f1.get(col, 0)
        elif "F2" in col:
            new_fight[col] = stats_f2.get(col, 0)
        else:
            new_fight[col] = 0

    model = models.get((weightclass, gender))
    if model is None:
        print(f"No model found for {weightclass}, {gender}")
        return

    X_new = pd.DataFrame([new_fight])
    winner_pred = model.predict(X_new)[0]

    # Feature Importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    top10 = importance_df.sort_values('importance', ascending=False).head(10)

    print(f"\nPrediction for {fighter1} vs {fighter2} ({weightclass}, {gender}):")
    print(f"Predicted Winner: {'Fighter1 (' + fighter1 + ')' if winner_pred == 1 else 'Fighter2 (' + fighter2 + ')'}")
    print("\nTop 10 Features for this model:")
    print(top10)

if __name__ == "__main__":
    # Load preprocessed data and trained models
    df = pd.read_csv(os.path.join("data", "fight_totals_preprocessed.csv"))
    with open(os.path.join("models", "random_forest_models.pkl"), "rb") as f:
        models = pickle.load(f)
    feature_cols = [col for col in df.columns if col not in [
        'Fighter1', 'Fighter2', 'Winner?', 'Winner?.1', 'Fight Method', 'Time', 'Time Format', 'Referee',
        'Finish Details or Judges Scorecard', 'Bout', 'Event Name', 'Location', 'Date',
        'Weightclass', 'Gender'
    ]]

    # Interactive input for fight prediction
    print("Welcome to the UFC Fight Predictor!")
    fighter1 = input("Enter name of Fighter 1: ")
    fighter2 = input("Enter name of Fighter 2: ")
    weightclass = input("Enter weight class (e.g. 'Flyweight'): ")
    gender = input("Enter gender ('Men' or 'Women'): ")

    predict_fight(
        fighter1=fighter1,
        fighter2=fighter2,
        weightclass=weightclass,
        gender=gender,
        df=df,
        models=models,
        feature_cols=feature_cols
    )