import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

# Set Matplotlib to use a non-GUI backend
plt.switch_backend('Agg')

app = Flask(__name__)

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to display stock data
@app.route('/dashboard', methods=['POST'])
def dashboard():
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Fetch stock data
    data = fetch_stock_data(ticker, start_date, end_date)

    # Plot stock price
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.title(f'{ticker} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.savefig('static/stock_plot.png')
    plt.close()

    # Prepare data for table
    table_data = data[['Close', 'Volume']].tail(10)

    return render_template('dashboard.html', ticker=ticker, table_data=table_data.to_html(classes='table'), plot_url='static/stock_plot.png')

if __name__ == '__main__':
    app.run(debug=True)
