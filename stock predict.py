import requests
import yfinance as yfin
import datetime as dt
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from kerastuner.tuners import RandomSearch

# Define a function to retrieve the ticker symbol for a company
def get_ticker(company_name):
    yfinance_url = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance_url, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code

# Enable yfinance for Pandas Datareader
yfin.pdr_override()

# Get user input for data retrieval and prediction
std = int(input("Enter start date: "))
sty = int(input("Enter start year: "))
stm = int(input("Enter start month: "))
comp = input("Enter company name: ")
n = int(input("Enter number of days to predict: "))

# Retrieve the company ticker symbol
ticker = get_ticker(comp)
print(ticker)

# Get data
start = dt.datetime(sty, stm, std)
end = dt.datetime.now()
data = pdr.get_data_yahoo(ticker, start, end)
if data.empty:
    print("No trading data available for the specified date range.")
else:
    sclr = MinMaxScaler(feature_range=(0, 1))
    scld_data = sclr.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare input data
    predict_days = 700
    xtrain = []
    ytrain = []

    # Ensure the loop stays within bounds
    for x in range(predict_days, len(scld_data) - 1):
        start_index = x - predict_days
        end_index = x

        # Check if both start and end indices are within bounds
        if start_index >= 0 and end_index < len(scld_data):
            xtrain.append(scld_data[start_index:end_index, 0])
            ytrain.append(scld_data[end_index, 0])

    # Convert to numpy arrays and reshape for LSTM input
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))



    # Define the hyperparameter tuner
    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32),
                              return_sequences=True,
                              input_shape=(xtrain.shape[1], 1)))
        model.add(layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(layers.LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32),
                              return_sequences=True))
        model.add(layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(layers.LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32)))
        model.add(layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
        model.add(layers.Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,
        directory='my_dir',
        project_name='stock_price_prediction')

    # Perform hyperparameter search
    tuner.search(xtrain, ytrain, epochs=25, validation_split=0.2)

    # Retrieve the best model and evaluate it
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()

    # Retrieve the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_units = best_hyperparameters.get('units')
    best_dropout = best_hyperparameters.get('dropout')

    # Train the model with the best hyperparameters
    best_model.fit(xtrain, ytrain, epochs=25, batch_size=64)

    # Predict future stock prices
    last_n_days = scld_data[-predict_days:]
    predictions = []
    for i in range(n):
        next_day = best_model.predict(last_n_days.reshape(1, predict_days, 1))
        prediction = sclr.inverse_transform(next_day)[0][0]
        predictions.append(prediction)
        print(f"Day {i+1}: {prediction:.2f}")
        last_n_days = np.append(last_n_days, next_day)[1:]

