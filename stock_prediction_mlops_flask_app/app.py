from flask import Flask, render_template, request
from model import StockModel
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        stock_model = StockModel(ticker)
        stock_model.load_model()

        # Fetch the data for the specified date range
        data = stock_model.fetch_data(start_date, end_date)
        
        if not data.empty:
            # Calculate moving averages
            latest_MA_5 = data['Close'].rolling(window=5).mean().iloc[-1]
            latest_MA_10 = data['Close'].rolling(window=10).mean().iloc[-1]

            # Prepare data for prediction
            latest_input = np.array([[latest_MA_5, latest_MA_10]])

            # Make prediction
            prediction = stock_model.predict(latest_input)[0]
        else:
            prediction = "No data available for the specified date range."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
