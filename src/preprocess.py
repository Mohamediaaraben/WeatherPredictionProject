import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_preprocess_data(file_path):
    # Charger les données
    df = pd.read_csv(file_path)

    # Sélectionner les colonnes pertinentes
    features = ['MinTemp', 'MaxTemp', '9amTemp', '3pmTemp']
    target = ['Rainfall']
    
    X = df[features].fillna(method='ffill')
    y = df[target].fillna(0)

    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values

