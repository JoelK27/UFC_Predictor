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
    outcome_models = {}
    round_models = {}

    # For each combination of weight class and gender, train a separate model
    for (weight, gender), group in df.groupby(['Weightclass', 'Gender']):
        # Skip groups with too few samples
        if len(group) < 20:
            continue

        # Define feature columns (same for all models)
        feature_cols = [
            col for col in group.columns if col not in [
                'Fighter1', 'Fighter2', 'Winner?', 'Winner?.1', 'Fight Method', 'Time', 'Time Format', 'Referee',
                'Finish Details or Judges Scorecard', 'Bout', 'Event Name', 'Location', 'Date',
                'Weightclass', 'Gender', 'Rounds'
            ]
        ]
        X = group[feature_cols]

        # --- Winner Model ---
        y = group['Winner?']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        models[(weight, gender)] = model
        print(f"\nWinner Model for {weight}, {gender}:")
        print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))


        # --- Outcome Model ---
        if 'Fight Method' in group.columns:
            y_outcome = group['Fight Method'].fillna('Unknown')
            X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X, y_outcome, test_size=0.2, random_state=42)
            outcome_model = RandomForestClassifier(n_estimators=100, random_state=42)
            outcome_model.fit(X_train_o, y_train_o)
            outcome_models[(weight, gender)] = outcome_model
            # Print accuracy
            print(f"\nOutcome Model for {weight}, {gender}:")
            print("Accuracy:", accuracy_score(y_test_o, outcome_model.predict(X_test_o)))
            print(classification_report(y_test_o, outcome_model.predict(X_test_o)))

        # --- Round Model ---
        if 'Rounds' in group.columns and 'Fight Method' in group.columns:
            # Create a new target variable for rounds
            # If the fight ended in a decision, label it as "Decision", otherwise use the round number
            def round_label(row):
                if "Decision" in str(row["Fight Method"]):
                    return "Decision"
                else:
                    return str(int(row["Rounds"]))
            
            y_round = group.apply(round_label, axis=1)
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_round, test_size=0.2, random_state=42)
            round_model = RandomForestClassifier(n_estimators=100, random_state=42)
            round_model.fit(X_train_r, y_train_r)
            round_models[(weight, gender)] = round_model
            # Print accuracy
            print(f"\nRound Model for {weight}, {gender}:")
            print("Accuracy:", accuracy_score(y_test_r, round_model.predict(X_test_r)))
            print(classification_report(y_test_r, round_model.predict(X_test_r)))
                
        # --- Winner Feature Importance ---
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
        importance_df = importance_df[~importance_df['feature'].str.contains('Winner\?', regex=True)]
        top10 = importance_df.sort_values('importance', ascending=False).head(10)
        print(f"\nTop 10 features for Winner Model ({weight}, {gender}):")
        print(top10)

        # Plot feature importance
        plt.figure(figsize=(8, 5))
        plt.barh(top10['feature'][::-1], top10['importance'][::-1])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 10 Features: {weight}, {gender}')
        plt.tight_layout()
        if not os.path.exists("data"):
            os.makedirs("data")
        plt.savefig(f'data/feature_importance_{weight}_{gender}.png')
        plt.close()

    # Save all trained models as pickle files for later predictions
    if not os.path.exists("models"):
        os.makedirs("models")
    with open("models/random_forest_models.pkl", "wb") as f:
        pickle.dump(models, f)
    with open("models/random_forest_outcome_models.pkl", "wb") as f:
        pickle.dump(outcome_models, f)
    with open("models/random_forest_round_models.pkl", "wb") as f:
        pickle.dump(round_models, f)