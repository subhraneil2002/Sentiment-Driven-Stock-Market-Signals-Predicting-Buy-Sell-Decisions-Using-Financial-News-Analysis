import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import requests

class StockSentimentAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol.strip().upper()
        self.api_key = "cse9gdhr01qs1ihohca0cse9gdhr01qs1ihohcag"  # Using provided API key
        self.news_df = pd.DataFrame()
        self.stock_data = pd.DataFrame()
        self.merged_df = pd.DataFrame()
        self.sentiment_scores = []
        self.model = LogisticRegression()

    def get_news_data(self, from_date, to_date):
        url = f'https://finnhub.io/api/v1/company-news?symbol={self.symbol}&from={from_date}&to={to_date}&token={self.api_key}'
        response = requests.get(url)
        articles = response.json()

        news_data = []
        for article in articles:
            title = article.get('headline', '')
            description = article.get('summary', '')
            published_at = datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d %H:%M:%S')
            news_data.append((published_at, title + ' ' + description))

        self.news_df = pd.DataFrame(news_data, columns=['Date', 'News'])

    def get_stock_data(self, start_date, end_date):
        self.stock_data = yf.download(self.symbol, start=start_date, end=end_date)
        return not self.stock_data.empty

    def analyze_sentiment(self):
        analyzer = SentimentIntensityAnalyzer()
        self.sentiment_scores = []

        for news in self.news_df['News']:
            vs = analyzer.polarity_scores(news)
            self.sentiment_scores.append(vs['compound'])

        self.news_df['Sentiment'] = self.sentiment_scores

    def preprocess_and_merge_data(self):
        self.merged_df = pd.merge(self.stock_data.reset_index(), self.news_df, how='left', left_on='Date', right_on='Date')
        self.merged_df['Sentiment'].fillna(0, inplace=True)

    def feature_engineering(self):
        self.merged_df['Return'] = self.merged_df['Close'].pct_change()
        self.merged_df['Signal'] = np.where(self.merged_df['Sentiment'] > 0, 1, 0)  # 1 for buy, 0 for sell
        self.merged_df['Target'] = self.merged_df['Return'].shift(-1)  # Target is next day's return
        self.merged_df.dropna(inplace=True)

    def train_model(self):
        features = self.merged_df[['Sentiment', 'Close']]
        target = np.where(self.merged_df['Target'] > 0, 1, 0)  # Binary classification

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy * 100  # Return accuracy as a percentage

    def calculate_metrics(self):
        # Calculate CAGR, Max Drawdown
        if not self.merged_df.empty:
            total_return = (1 + self.merged_df['Return']).prod() - 1
            days = (self.merged_df['Date'].iloc[-1] - self.merged_df['Date'].iloc[0]).days
            cagr = ((1 + total_return) ** (365 / days)) - 1

            # Calculate maximum drawdown
            cumulative_return = (1 + self.merged_df['Return']).cumprod()
            peak = cumulative_return.cummax()
            drawdown = (cumulative_return - peak) / peak
            max_drawdown = drawdown.min()

            return {
                'CAGR': cagr,
                'Max Drawdown': max_drawdown,
                'Total Return': total_return
            }
        return {}

    def final_decision(self):
        latest_sentiment = self.merged_df['Sentiment'].iloc[-1]
        if latest_sentiment > 0:
            return "Buy"
        elif latest_sentiment < 0:
            return "Sell"
        else:
            return "Hold"

# Fetch and process data function
def fetch_and_process_data(analyzer, start_date, end_date):
    analyzer.get_news_data(from_date=start_date, to_date=end_date)
    if not analyzer.get_stock_data(start_date=start_date, end_date=end_date):
        return False

    # Preprocess and analyze sentiment
    analyzer.preprocess_and_merge_data()
    analyzer.analyze_sentiment()
    analyzer.feature_engineering()
    return True

# Streamlit App
def main():
    st.title("Stock Sentiment Analyzer")

    # Only input for stock symbol
    symbol = st.text_input("Enter the stock ticker symbol:", value='AAPL').strip().upper()

    if st.button("Analyze"):
        if not symbol:
            st.error("Please enter a stock ticker symbol.")
            return

        # Initialize analyzer with stock symbol only
        analyzer = StockSentimentAnalyzer(symbol)

        # Define date range for data collection
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)

        # Fetch and process data
        if fetch_and_process_data(analyzer, start_date, end_date):
            # Calculate performance metrics
            metrics = analyzer.calculate_metrics()

            # Train the model
            accuracy = analyzer.train_model()

            # Display results
            st.subheader("Performance Metrics")
            for key, value in metrics.items():
                st.write(f"{key}: {value:.2f}" if key in ['CAGR', 'Max Drawdown'] else f"{key}: {value:.2f}")

            st.subheader("Model Accuracy")
            st.write("Accuracy:", accuracy)

            # Final decision with summary
            decision = analyzer.final_decision()
            sentiment_summary = f"The overall sentiment for {symbol} is '{decision}', with recent trends indicating a likely {decision} recommendation."

            st.subheader("Final Decision")
            st.write(sentiment_summary)
        else:
            st.error("Invalid ticker symbol. Please try again.")

if __name__ == "__main__":
    main()
