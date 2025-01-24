import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess_data

# Charger et pré-traiter les données
X, y = load_and_preprocess_data('../data/TemperatureRainFall.csv')

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape des données pour LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Définir le modèle LSTM
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

# Compiler le modèle
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Sauvegarder le modèle
model.save('../models/weather_forecast_model.h5')
print("Modèle entraîné et sauvegardé avec succès.")

