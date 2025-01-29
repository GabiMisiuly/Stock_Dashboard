import time
from yahoo_fin import news
from nltk.sentiment import SentimentIntensityAnalyzer

def fetch_news_sentiment(tickers):
    """
    Fetch news headlines for given tickers and compute sentiment scores.
    Returns a dictionary of ticker -> sentiment score.
    """
    sia = SentimentIntensityAnalyzer()  # Initialize sentiment analyzer once
    sentiment_results = {}

    for ticker in tickers:
        try:
            print(f"Fetching news for {ticker}...")
            articles = news.get_yf_rss(ticker)
            headlines = [article["title"] for article in articles[:5]]

            if not headlines:
                print(f"No news found for {ticker}. Skipping sentiment analysis.")
                sentiment_results[ticker] = "N/A"
                continue
            
            print(f"Found {len(headlines)} headlines for {ticker}")

            sentiment_scores = [sia.polarity_scores(headline)['compound'] for headline in headlines]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

            sentiment_results[ticker] = avg_sentiment  # Store average sentiment score
            print(f"{ticker} Sentiment Score: {avg_sentiment:.2f}")

            time.sleep(1)  # Add delay to avoid API rate limits

        except Exception as e:
            print(f"Error fetching news for {ticker}: {e} - Returning N/A")
            sentiment_results[ticker] = "N/A"
    
    return sentiment_results






def generate_sentiment_table(tickers):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = {}

    for ticker in tickers:
        articles = news.get_yf_rss(ticker)
        if not articles:
            sentiment_scores[ticker] = "N/A"
            continue

        headlines = [article["title"] for article in articles[:5]]
        sentiment_values = [sia.polarity_scores(headline)['compound'] for headline in headlines]
        
        sentiment_scores[ticker] = sum(sentiment_values) / len(sentiment_values) if sentiment_values else "N/A"

    return sentiment_scores

