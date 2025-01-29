# Stock_Dashboard
This repository contains a Flask web application that allows users to retrieve and visualize stock market data. The application integrates with Yahoo Finance to fetch historical stock data and presents the results as interactive plots and tables. Below is a detailed breakdown of the project:

Stock-Dashboard/
├── Stock_Dashboard.py       # Main application script
├── templates/               # Folder for HTML templates
│   ├── index.html           # Home page with a form to input stock details
│   ├── dashboard.html       # Page displaying stock data and visualization
├── static/                  # Folder for static files
│   └── stock_plot.png       # Dynamically generated stock price plot


Features

Home Page:

Accessible at http://127.0.0.1:5000/.
Contains a form where users can enter:
Stock ticker symbol (e.g., AAPL for Apple).
Start date for historical data.
End date for historical data.

Dashboard Page:
Accessible after submitting the form on the home page.

Displays:
A dynamically generated plot of the stock's closing price over the selected date range.
A table showing the latest 10 rows of data, including the Close price and Volume.
Data Source:

Uses Yahoo Finance API (via the yfinance library) to fetch historical stock data.
Visualization:

Stock price data is visualized using Matplotlib.
The plot is saved as a PNG file in the static/ directory and displayed on the dashboard page.

Dynamic Web Interface:
Flask handles routing and rendering of HTML templates.
HTML templates (index.html and dashboard.html) use Jinja2 for dynamic content rendering.

Requirements
To run this application, the following dependencies must be installed:
Python 3.7 or later
Required Python libraries:
pandas yfinance matplotlib flask



Summary of the Stock Analysis Dashboard Script
Your script is a Flask-based Stock Analysis Dashboard that retrieves stock market data via Yahoo Finance (yfinance) and provides key risk metrics, P&L tracking, sentiment analysis, and visualization. Here’s a breakdown of its key functionalities:

1. Stock Data Retrieval:
Fetches historical stock data for a given ticker from Yahoo Finance.
Computes key financial metrics:
Relative Volume (current volume vs. average volume)
Daily Return (percentage change in closing price)
Volatility (30-day rolling standard deviation, annualized)
Sharpe Ratio (risk-adjusted return)
Beta (systematic risk measure from yfinance)
Max Drawdown (largest peak-to-trough decline)
Rolling 30-Day Volatility
RSI (Relative Strength Index) (momentum indicator)
Skewness & Kurtosis (distribution shape analysis)
VaR (Value at Risk) 95% (potential loss under normal conditions)
CVAR (Conditional VaR) (expected loss beyond VaR)
Omega Ratio (ratio of positive to negative returns)
Correlation with S&P 500 (assessing market dependence)
2. Sentiment Analysis:
Uses an external module (sentiment_score.py) to analyze news sentiment for selected stocks.
Classifies sentiment as Bullish, Bearish, or Neutral.
3. Portfolio Performance Tracking:
Tracks Daily (DTD) and Monthly (MTD) P&L for selected stocks.
Calculates previous month's performance (MTD P&L T-1).
Provides comparative metrics on stock price, volume, volatility, and relative volume.
4. Data Visualization:
Stock Price Line Chart (Matplotlib/Seaborn)
Relative Volume Chart (Seaborn)
HTML-styled tables (using Pandas .style for formatting)
5. Flask Web Application:
Routes:
/ → Home Page
/dashboard → Displays the stock analysis dashboard based on user input.

