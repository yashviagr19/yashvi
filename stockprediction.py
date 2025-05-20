import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Function to download stock data
def download_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data['Close']

# Download stock prices for Apple and Amazon
start_date = '2015-01-01'
end_date = '2023-12-31'
apple_data = download_data('AAPL', start_date, end_date)
amazon_data = download_data('AMZN', start_date, end_date)

# Function to prepare data for LSTM
def prepare_data(data, time_step):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(data_scaled) - time_step):
        X.append(data_scaled[i:i + time_step, 0])
        y.append(data_scaled[i + time_step, 0])

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

# Prepare data for Apple and Amazon
time_step = 60
X_apple, y_apple, apple_scaler = prepare_data(apple_data, time_step)
X_amazon, y_amazon, amazon_scaler = prepare_data(amazon_data, time_step)

# Reshape data for LSTM
X_apple = X_apple.reshape(X_apple.shape[0], X_apple.shape[1], 1)
X_amazon = X_amazon.reshape(X_amazon.shape[0], X_amazon.shape[1], 1)

# Build the LSTM model
def create_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and predict for a stock
def train_and_predict(X, y, scaler, original_data):
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = create_model()
    model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)

    # Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Reverse scaling
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(original_data.index, original_data, label='Original Prices', color='blue')
    plt.plot(original_data.index[time_step:train_size + time_step], train_predict, label='Training Predictions', color='green')
    plt.plot(original_data.index[train_size + time_step:], test_predict, label='Testing Predictions', color='red')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return model

# Predict for Apple
print("Training for Apple...")
apple_model = train_and_predict(X_apple, y_apple, apple_scaler, apple_data)

# Predict for Amazon
print("Training for Amazon...")
amazon_model = train_and_predict(X_amazon, y_amazon, amazon_scaler,amazon_data)