import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
import matplotlib.pyplot as plt
import math

# 1. Data Acquisition
try:
    data = yf.download('AAPL', start='2010-01-01', end='2025-05-23')
    if data.empty:
        raise ValueError("No data downloaded from yfinance. Check internet connection or ticker symbol.")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# 2. Data Preprocessing
data = data.dropna()
data['Close_lag1'] = data['Close'].shift(1)
data = data.dropna()

# 3. Feature Engineering
data['MA50'] = data['Close'].rolling(window=50).mean()
data = data.dropna()  # Remove rows with NaN values from moving average

# 4. Split Data
train_size = int(len(data) * 0.9)
train, test = data['Close'][:train_size], data['Close'][train_size:]
train_index, test_index = data.index[:train_size], data.index[train_size:]

# 5. ARIMA Model
try:
    model_arima = ARIMA(train, order=(1, 1, 1))
    model_fit = model_arima.fit()
    forecast_arima = model_fit.forecast(steps=len(test))

    # Evaluate ARIMA
    mae_arima = mean_absolute_error(test, forecast_arima)
    rmse_arima = math.sqrt(mean_squared_error(test, forecast_arima))
    print(f'ARIMA - MAE: {mae_arima:.2f}, RMSE: {rmse_arima:.2f}')
except Exception as e:
    print(f"ARIMA model failed: {e}")
    forecast_arima = None

# 6. LSTM Model
try:
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'MA50']].dropna())

    # Create sequences
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, :])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 2))

    # Split into train and test
    train_size_lstm = int(len(X) * 0.9)
    X_train, X_test = X[:train_size_lstm], X[train_size_lstm:]
    y_train, y_test = y[:train_size_lstm], y[train_size_lstm:]
    test_index_lstm = test_index[-len(y_test):]  # Align indices for plotting

    # Build LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 2)))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(units=50, return_sequences=False))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(units=1))
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Predict with LSTM
    predictions_lstm_scaled = model_lstm.predict(X_test)
    # Inverse transform predictions
    predictions_lstm = scaler.inverse_transform(
        np.concatenate((predictions_lstm_scaled, X_test[:, -1, 1:]), axis=1)
    )[:, 0]
    y_test_unscaled = scaler.inverse_transform(
        np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1)
    )[:, 0]

    # Evaluate LSTM
    mae_lstm = mean_absolute_error(y_test_unscaled, predictions_lstm)
    rmse_lstm = math.sqrt(mean_squared_error(y_test_unscaled, predictions_lstm))
    print(f'LSTM - MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}')
except Exception as e:
    print(f"LSTM model failed: {e}")
    predictions_lstm = None
    test_index_lstm = None

# 7. Prophet Model
predictions_prophet = None  # Initialize to None for plotting safety
try:
    # Ensure 'ds' is a datetime column and 'y' is 1-dimensional
    df_prophet = pd.DataFrame({
        'ds': pd.to_datetime(data.index),  # Ensure datetime format
        'y': data['Close'].to_numpy().flatten()  # Convert to 1D array
    })
    # Debug: Check DataFrame format
    print("Prophet DataFrame head:\n", df_prophet.head())
    print("Shape of df_prophet['y']:", df_prophet['y'].shape)
    print("Type of df_prophet['y']:", df_prophet['y'].dtype)
    print("Type of df_prophet['ds']:", df_prophet['ds'].dtype)

    # Initialize and fit Prophet model
    model_prophet = Prophet(daily_seasonality=True)
    model_prophet.fit(df_prophet)
    future = model_prophet.make_future_dataframe(periods=len(test))
    forecast_prophet = model_prophet.predict(future)
    predictions_prophet = forecast_prophet['yhat'][-len(test):].to_numpy().flatten()

    # Evaluate Prophet
    mae_prophet = mean_absolute_error(test, predictions_prophet)
    rmse_prophet = math.sqrt(mean_squared_error(test, predictions_prophet))
    print(f'Prophet - MAE: {mae_prophet:.2f}, RMSE: {rmse_prophet:.2f}')
except Exception as e:
    print(f"Prophet model failed: {e}")
    predictions_prophet = None

# 8. Plot Results
plt.figure(figsize=(12, 6))
plt.plot(test_index, test, label='Actual', color='blue')
if forecast_arima is not None:
    plt.plot(test_index, forecast_arima, label='ARIMA Forecast', color='green')
if predictions_lstm is not None and test_index_lstm is not None:
    plt.plot(test_index_lstm, predictions_lstm, label='LSTM Forecast', color='red')
if predictions_prophet is not None:
    plt.plot(test_index, predictions_prophet, label='Prophet Forecast', color='orange')
plt.legend()
plt.title('Stock Price Predictions for AAPL')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.grid(True)
plt.savefig('stock_predictions.png')
plt.show()