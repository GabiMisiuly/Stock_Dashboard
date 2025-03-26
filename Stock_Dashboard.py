import requests
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime



# Initialize Flask app
app = Flask(__name__)

# Set Matplotlib backend to non-GUI
plt.switch_backend('Agg')

# Function to fetch stock data using yfinance
# Function to fetch stock data using yfinance
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        data.reset_index(inplace=True)
        
        if data.empty:
            print(f"Data for {ticker} not available.")
            return pd.DataFrame()

        # Calculate additional metrics
        data['Relative Volume'] = data['Volume'] / data['Volume'].mean()
        data['Daily Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily Return'].rolling(window=30).std() * np.sqrt(252)
        data['2 Std Dev'] = 2 * data['Volatility']
        data['Sharpe Ratio'] = data['Daily Return'].mean() / data['Daily Return'].std() * np.sqrt(252)  # Sharpe Ratio
        data['Beta'] = stock.info.get('beta', 'N/A')  # Fetch beta value from yfinance
        data['Max Drawdown'] = (data['Close'] / data['Close'].cummax() - 1).min()
        data['Rolling 30-Day Volatility'] = data['Daily Return'].rolling(window=30).std() * np.sqrt(252)
        data['RSI'] = 100 - (100 / (1 + (data['Close'].pct_change().rolling(14).mean() / data['Close'].pct_change().rolling(14).std())))
        data['Skewness'] = data['Daily Return'].rolling(window=30).skew()
        data['Kurtosis'] = data['Daily Return'].rolling(window=30).kurt()
        data['VaR 95%'] = data['Daily Return'].quantile(0.05)
        data['CVAR 95%'] = data['Daily Return'][data['Daily Return'] <= data['VaR 95%']].mean()
        data['Omega Ratio'] = (data['Daily Return'][data['Daily Return'] > 0].sum()) / abs(data['Daily Return'][data['Daily Return'] < 0].sum())
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')

        # Fetch Forward P/E Ratio from Yahoo Finance
        forward_pe = stock.info.get('forwardPE', 'N/A')  # Get Forward P/E if available
        data['Forward P/E'] = forward_pe  # Store as a column (optional)

        # Fetch S&P 500 data for correlation calculation
        sp500 = yf.Ticker("^GSPC").history(start=start_date, end=end_date)['Close'].pct_change()
        if not sp500.empty:
            data['Correlation with S&P 500'] = data['Daily Return'].rolling(window=30).corr(sp500)
        else:
            data['Correlation with S&P 500'] = np.nan

        return data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

    



# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Dashboard route
@app.route('/dashboard', methods=['POST'])
def dashboard():
    tickers = [t.strip().upper() for t in request.form['ticker'].split(',')]
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    stock_data_summary = []
    stock_data = {ticker: fetch_stock_data(ticker, start_date, end_date) for ticker in tickers}
    print("Fetched Stock Data:", stock_data)  # Debugging step


    # Sentiment analysis   

    from sentiment_score import fetch_news_sentiment  # Import function 
    from news_fetcher import fetch_recent_news  # Import function to fetch recent news

    sentiment_scores = fetch_news_sentiment(tickers)  # Fetch sentiment

    if sentiment_scores:
        sentiment_summary_table = """
            <table class='table table-striped table-bordered text-center'>
            <thead class='table-dark'>
                <tr>
                    <th style="width: 20%;">Ticker</th>
                    <th style="width: 40%;">Sentiment Score</th>
                    <th style="width: 40%;">Trend</th>
                </tr>
            </thead>
            <tbody>
        """

        for ticker in tickers:
            sentiment_score = sentiment_scores.get(ticker, "N/A")

            # Determine sentiment classification
            if isinstance(sentiment_score, float):
                if sentiment_score > 0.1:
                    trend = "Bullish"
                    color = "green"
                elif sentiment_score < -0.1:
                    trend = "Bearish"
                    color = "red"
                else:
                    trend = "Neutral"
                    color = "gray"

                formatted_score = f"{sentiment_score:.4f}"
            else:
                trend = "Neutral"
                color = "gray"
                formatted_score = "N/A"

            sentiment_summary_table += f"""
                <tr>
                    <td style='font-weight: bold;'>{ticker}</td>
                    <td style='color: {color}; font-size: 10px;'>{formatted_score}</td>
                    <td style='color: {color}; font-weight: bold; font-size: 10px;'>{trend}</td>
                </tr>
            """

        sentiment_summary_table += "</tbody></table>"

        # Adding a section for recent news links
        news_links_table = """
            <h3>Recent News Articles</h3>
            <table class='table table-striped table-bordered text-center'>
            <thead class='table-dark'>
                <tr>
                    <th style="width: 20%;">Ticker</th>
                    <th style="width: 80%;">Recent News</th>
                </tr>
            </thead>
            <tbody>
        """

        for ticker in tickers:
            news_links = fetch_recent_news(ticker)  # Fetch the latest 5 news links
            if news_links:
                news_links_html = "".join(
                    [f"<li><a href='{link}' target='_blank'>{link}</a></li>" for link in news_links[:5]]
                )
            else:
                news_links_html = "<li>No recent news available</li>"

            news_links_table += f"""
                <tr>
                    <td style='font-weight: bold;'>{ticker}</td>
                    <td><ul>{news_links_html}</ul></td>
                </tr>
            """

        news_links_table += "</tbody></table>"

        # Final HTML output
        final_output = sentiment_summary_table + news_links_table

    else:
        print("⚠ Warning: No sentiment scores found.")
        final_output = "<p>No sentiment data available.</p>"



    # Generate stock risk and performance report data

    for ticker, data in stock_data.items():
        if data.empty:
            continue
        
        data = data.sort_values(by='Date', ascending=True)
        
        latest_data = data.iloc[-1] if not data.empty else None
        previous_data = data.iloc[-2] if len(data) >= 2 else latest_data
        
        month_start_data = data[data['Date'].str.startswith(start_date[:7])]
        mtd_start_price = month_start_data.iloc[0]['Close'] if not month_start_data.empty else latest_data['Close']
        
        if latest_data is not None:
            mtd_pnl = ((latest_data['Close'] - mtd_start_price) / mtd_start_price) * 100
            dtd_pnl = ((latest_data['Close'] - previous_data['Close']) / previous_data['Close']) * 100 if previous_data is not None else 0
        else:
            mtd_pnl, dtd_pnl = 0, 0
        
        mtd_pnl_t1 = ((previous_data['Close'] - mtd_start_price) / mtd_start_price) * 100 if previous_data is not None else 0
        
        stock_data_summary.append({
            "Ticker": ticker,
            "DTD P&L": f"{dtd_pnl:+.2f}%",
            "MTD P&L": f"{mtd_pnl:+.2f}%",
            "MTD P&L T-1": f"{mtd_pnl_t1:+.2f}%",
            "Stock Price": f"${latest_data['Close']:.2f}" if latest_data is not None else "N/A",
            "Stock Price T-1": f"${previous_data['Close']:.2f}" if previous_data is not None else "N/A",
            "Volume": f"{latest_data['Volume']:,}" if latest_data is not None else "N/A",
            "Volume T-1": f"{previous_data['Volume']:,}" if previous_data is not None else "N/A",
            "Relative Volume": f"{latest_data['Relative Volume']:.2f}" if latest_data is not None else "N/A",
            "Relative Volume T-1": f"{previous_data['Relative Volume']:.2f}" if previous_data is not None else "N/A",
            "Volatility": f"{latest_data['Volatility']:.4f}" if latest_data is not None else "N/A",
            "Volatility T-1": f"{previous_data['Volatility']:.4f}" if previous_data is not None else "N/A",
            "2 Std Dev": f"{latest_data['2 Std Dev']:.4f}" if latest_data is not None else "N/A",
            "Sharpe Ratio": f"{latest_data['Sharpe Ratio']:.4f}" if latest_data is not None else "N/A",
            "Beta": f"{float(latest_data['Beta']):.2f}" if latest_data is not None and isinstance(latest_data['Beta'], (int, float)) else "N/A",
            "Rolling 30-Day Volatility": f"{latest_data['Rolling 30-Day Volatility']:.4f}",
            "Correlation with S&P 500": "N/A",  # Placeholder - Needs market data
            "Omega Ratio": f"{latest_data['Omega Ratio']:.4f}",
            "CVAR 95%": f"{latest_data['CVAR 95%']:.4f}",
            "VaR 95%": f"{latest_data['VaR 95%']:.4f}",
            "Max Drawdown": f"{latest_data['Max Drawdown']:.4f}",
            "Skewness": f"{latest_data['Skewness']:.4f}",
            "Kurtosis": f"{latest_data['Kurtosis']:.4f}",
            "RSI": f"{latest_data['RSI']:.2f}",
            "P/E": f"{float(latest_data['Forward P/E']):.2f}" if isinstance(latest_data['Forward P/E'], (int, float)) else latest_data['Forward P/E']
             })
        

    
    # Generate and save stock price plot
    stock_price_fig, stock_price_ax = plt.subplots(figsize=(14, 7))
    for ticker, data in stock_data.items():
        if not data.empty:
            sns.lineplot(x=pd.to_datetime(data['Date']), y=data['Close'], label=f'{ticker} Close Price', linewidth=2, ax=stock_price_ax)
    stock_price_fig.tight_layout()
    stock_price_fig.savefig('static/stock_price_plot.png')
    plt.close(stock_price_fig)
    
    # Generate and save relative volume plot
    relative_volume_fig, relative_volume_ax = plt.subplots(figsize=(14, 7))
    for ticker, data in stock_data.items():
        if not data.empty:
            sns.lineplot(x=pd.to_datetime(data['Date']), y=data['Relative Volume'], label=f'{ticker} Relative Volume', linewidth=2, ax=relative_volume_ax)
    relative_volume_fig.tight_layout()
    relative_volume_fig.savefig('static/relative_volume_plot.png')
    plt.close(relative_volume_fig)

    # Ensure styled_stock_table has a default value before using it
    if stock_data_summary:
        stock_table_df = pd.DataFrame(stock_data_summary)

        styled_stock_table = (
            stock_table_df.style
            .set_table_attributes('class="table table-striped table-bordered text-center"')
            .set_properties(**{'text-align': 'center', 'font-size': '10px'})
            .set_table_styles([
                {'selector': 'thead th', 'props': [('background-color', 'black'), ('color', 'white'), 
                                                ('font-size', '10px'), ('text-align', 'center')]}
            ])
            .applymap(
                lambda val: 'color: green' if isinstance(val, str) and val.startswith('+') else 
                            'color: red' if isinstance(val, str) and val.startswith('-') else '',
                subset=["MTD P&L", "MTD P&L T-1", "DTD P&L"]
            )
            .to_html(index=False)
        )

    else:
        print("⚠ No stock data available. Creating empty table.")
        styled_stock_table = "<p>No stock data available.</p>"


    # Adding a section for recent news headlines (without links)
    news_links_table = """
        <h3>Recent News Articles</h3>
        <table class='table table-striped table-bordered text-center'>
        <thead class='table-dark'>
            <tr>
                <th style="width: 20%;">Ticker</th>
                <th style="width: 80%;">Recent News</th>
            </tr>
        </thead>
        <tbody>
    """

    for ticker in tickers:
        news_headlines = fetch_recent_news(ticker)  # Fetch headlines only (no links)
        
        if news_headlines:
            news_headlines_html = "".join([f"<li>{headline}</li>" for headline in news_headlines[:5]])
        else:
            news_headlines_html = "<li>No recent news available</li>"

        news_links_table += f"""
            <tr>
                <td style='font-weight: bold;'>{ticker}</td>
                <td><ul style="text-align:left; padding-left:15px;">{news_headlines_html}</ul></td>
            </tr>
        """

    news_links_table += "</tbody></table>"




    
    return render_template('dashboard.html', 
                           stock_table=styled_stock_table,
                           sentiment_summary_table=sentiment_summary_table,
                           stock_plot_url='static/stock_price_plot.png',
                           relative_volume_url='static/relative_volume_plot.png',
                           news_links_table=news_links_table)

if __name__ == '__main__':
    app.run(debug=True)
