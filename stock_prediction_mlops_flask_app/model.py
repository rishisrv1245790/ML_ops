import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

class StockModel:
    def __init__(self, ticker):
        self.ticker = ticker
        self.model = LinearRegression()

    def fetch_data(self, start_date, end_date):
        # Fetch OHLC data for the given date range
        df = yf.download(self.ticker, start=start_date, end=end_date)
        return df

    def preprocess_data(self, df):
        # Feature Engineering: Adding Moving Averages and Target
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['Target'] = df['Close'].shift(-1)  # The next day's closing price
        df.dropna(inplace=True)  # Drop rows with NaN values
        return df

    def train_model(self, df):
        X = df[['MA_5', 'MA_10']]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        print(f'Model trained with score: {self.model.score(X_test, y_test)}')
        
        # Save the model
        with open('stock_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        with open('stock_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, latest_data):
        return self.model.predict(latest_data)

def run_pipeline(ticker, start_date, end_date):
    stock_model = StockModel(ticker)
    df = stock_model.fetch_data(start_date, end_date)
    df = stock_model.preprocess_data(df)
    stock_model.train_model(df)
