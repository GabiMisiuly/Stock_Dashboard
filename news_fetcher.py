import yfinance as yf

def fetch_recent_news(ticker):
    """
    Fetch the latest news headlines for a given stock ticker using Yahoo Finance.
    Only displays the news title and publisher (No links).
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news  # Fetch news from yfinance

        if not news:
            print(f"No news available for {ticker}.")
            return ["No recent news available"]

        # Extract available news data (only title and publisher)
        news_list = []
        for article in news[:5]:  # Limit to 5 articles
            title = article.get('title', 'No title available')
            publisher = article.get('publisher', 'Unknown Source')
            news_list.append(f"<li>{title} - {publisher}</li>")

        return news_list

    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return ["Error retrieving news"]
