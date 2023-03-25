import requests
import yfinance as yfin
import datetime as dt
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM

def getTicker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code
yfin.pdr_override()
std = int(input("Enter start date: "))
sty = int(input("Enter start year: "))
stm = int(input("Enter start month: "))
comp = input("Enter company name: ")
n = int(input("Enter number of days to predict: "))
a = str(getTicker(comp))
print(a)

# Get data
start = dt.datetime(sty, stm, std)
end = dt.datetime.now()
data = pdr.get_data_yahoo(a, start, end)
if data.empty:
    print("No trading data available for the specified date range.")
else:
    sclr = MinMaxScaler(feature_range=(0, 1))
    scld_data = sclr.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare input data
    predict_days = 360
    xtrain = []
    ytrain = []
    for x in range(predict_days, len(scld_data)):
        xtrain.append(scld_data[x-predict_days:x, 0])
        ytrain.append(scld_data[x, 0])

    # Convert to numpy arrays and reshape for LSTM input
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

    # Build model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(xtrain, ytrain, epochs=25, batch_size=34)

    # Predict future stock prices
    last_n_days = scld_data[-predict_days:]
    predictions = []
    for i in range(n):
        next_day = model.predict(last_n_days.reshape(1, predict_days, 1))
        prediction = sclr.inverse_transform(next_day)[0][0]
        predictions.append(prediction)
        if i > 0:
            prev_prediction = predictions[i-1]
            change = (prediction-prev_prediction)/prev_prediction * 100
            if change > 0:
                print(f"Day {i+1}: Price has increased by {change:.2f}% to {prediction:.2f}")
            else:
                print(f"Day {i+1}: Price has decreased by {-change:.2f}% to {prediction:.2f}")
        else:
            print(f"Day {i+1}: {prediction:.2f}")
        last_n_days = np.append(last_n_days, next_day)[1:]
    
    # Scale predictions back to original values
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = sclr.inverse_transform(predictions)
