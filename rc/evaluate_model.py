from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from preprocess import load_and_preprocess_data

# Charger le modèle sauvegardé
model = load_model('../models/weather_forecast_model.h5')

# Charger les données de test
X, y = load_and_preprocess_data('../data/TemperatureRainFall.csv')

# Reshape des données
X = X.reshape((X.shape[0], X.shape[1], 1))

# Prédictions
y_pred = model.predict(X)

# Calculer l'erreur
mse = mean_squared_error(y, y_pred)
print(f'Erreur quadratique moyenne (MSE): {mse}')
