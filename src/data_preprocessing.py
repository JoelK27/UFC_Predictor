import pandas as pd

def load_and_preprocess_data(filepath):
    """
    Lädt die Rohdaten, bereinigt sie und gibt ein DataFrame zurück.
    """
    df = pd.read_csv(filepath)
    # Beispiel: Gewichtsklasse und Geschlecht extrahieren
    df['Weightclass'] = df['Bout'].str.extract(r'(\w+weight)')
    df['Gender'] = df['Bout'].apply(lambda x: 'Women' if 'Women' in x else 'Men')
    # Fehlende Werte behandeln
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    # Weitere Preprocessing-Schritte nach Bedarf
    return df