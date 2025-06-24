from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_models(df):
    """
    Trainiert für jede Gewichtsklasse und jedes Geschlecht ein Modell.
    Gibt ein Dictionary der Modelle und die verwendeten Feature-Spalten zurück.
    """
    models = {}
    feature_cols = None
    for (weight, gender), group in df.groupby(['Weightclass', 'Gender']):
        if len(group) < 20:
            continue
        y = group['Winner?']
        X = group.drop(columns=[
            'Fighter1', 'Fighter2', 'Winner?', 'Winner?.1', 'Fight Method', 'Time', 'Time Format', 'Referee',
            'Finish Details or Judges Scorecard', 'Bout', 'Event Name', 'Location', 'Date',
            'Weightclass', 'Gender'
        ])
        feature_cols = X.columns  # Für spätere Verwendung
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        models[(weight, gender)] = model
    return models, feature_cols