# WeatherPredictionProject
Introduction
This project is focused on building a time-series prediction model using LSTM (Long Short-Term Memory) networks to predict future values of temperature and rainfall based on past data. Time-series data is a sequence of data points indexed in time order, and in this case, the goal is to predict future weather patterns using historical data. The LSTM model is particularly suited for time-series forecasting due to its ability to capture long-range dependencies in sequential data.

In this code, we load a weather dataset containing temperature and rainfall values, preprocess the data, split it into training and test sets, build an LSTM model, train it on the data, evaluate its performance, and visualize the results. The project utilizes Python libraries such as Pandas, Matplotlib, Scikit-learn, and TensorFlow for deep learning.

1. Loading and Visualizing the Data

def load_and_visualize_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Display the first few rows and the information about the DataFrame
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Info:")
    df.info()

    # Handle missing values (using forward fill method)
    df.fillna(method='ffill', inplace=True)

    # Visualize the data
    plt.figure(figsize=(12, 6))
    for column in df.columns:
        plt.plot(df[column], label=column)
    plt.title('Temperature and Rainfall Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Scaled Values')
    plt.legend()
    plt.show()
Loading Data: The CSV file is loaded into a Pandas DataFrame.
Visualization: It displays the first few rows of the dataset (head()) and basic info (info()), then fills any missing values using the "forward fill" method (propagating the previous values).
Plotting: A plot is created to show the data (temperature, rainfall) over time.
2. Data Preprocessing

def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop the 'Date' column if it exists
    if 'Date' in df.columns:
        df.drop(columns=['Date'], inplace=True)

    # Handle missing values (forward fill)
    df.fillna(method='ffill', inplace=True)

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    return data_scaled, scaler
Dropping the 'Date' Column: If there is a 'Date' column, it is dropped because it's not used in training the model.
Handling Missing Values: Missing values are filled using forward fill.
Normalization: The data is normalized using MinMaxScaler from sklearn to scale all columns between 0 and 1, which is important for training neural networks.
3. Preparing Data for Time-Series Modeling

def prepare_time_series_data(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])  # Input sequence
        y.append(data[i + time_steps])   # Prediction target
    return np.array(X), np.array(y)
Time-Series Transformation: The goal is to prepare data for prediction. For each data point, the input sequence is made up of the last 10 time steps (default), and the prediction target is the next time step.
This transformation results in two arrays: X (inputs) and y (targets).
4. Splitting the Data into Training and Test Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Data Split: The data is divided into training (80%) and test (20%) sets.
5. Building and Training the LSTM Model
python
Copier
Modifier
def build_and_train_model(X_train, y_train, input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(5, activation='linear')  # Output of 5 predictions
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    return model, history
Model Architecture: An LSTM model is created with two LSTM layers and two dropout layers to reduce overfitting.
Model Compilation: The model uses the Adam optimizer and Mean Squared Error (MSE) as the loss function.
Model Training: The model is trained for 50 epochs with a batch size of 32, and 20% of the data is used for validation.
6. Plotting Training History

def plot_training_history(history):
    # Plot the loss and mean absolute error curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()
Training History Visualization: This function shows the loss (loss) and mean absolute error (mae) curves during training, for both the training set and the validation set.
7. Evaluating the Model

def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    return mse, mae
Predictions and Evaluation: After training, the model makes predictions on the test set. These predictions are then "denormalized" (scaled back to the original range), and the Mean Squared Error (MSE) and Mean Absolute Error (MAE) are calculated.
8. Denormalizing Predictions

predictions_denorm = scaler.inverse_transform(predictions)
y_test_denorm = scaler.inverse_transform(y_test)
Denormalization: The predictions made by the model are inverted back to the original scale to make them comparable to the actual values in the test set.
Conclusion
This project demonstrates the process of forecasting future weather patterns, specifically temperature and rainfall, using LSTM (Long Short-Term Memory) networks for time-series prediction. By preparing the data, building an LSTM model, training it, and evaluating its performance, this code serves as a foundation for time-series forecasting problems. The techniques and model used can be applied to other forecasting tasks, such as stock prices, sales data, or any type of sequential data.








   ```bash
   git clone https://github.com/mohamediaaraben/WeatherPredictionProject.git

