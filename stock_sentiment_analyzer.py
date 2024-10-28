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
import re

class StockSentimentAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol.strip().upper()
        self.api_key = "cse9gdhr01qs1ihohca0cse9gdhr01qs1ihohcag"  # Using provided API key
        self.news_df = pd.DataFrame()
        self.stock_data = pd.DataFrame()
        self.merged_df = pd.DataFrame()

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
        if self.stock_data.empty:
            return False
        return True

    def calculate_metrics(self):
        def calculate_cagr(data):
            cumulative_return = data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0]
            years = len(data) / 252
            return (cumulative_return ** (1 / years) - 1) * 100

        def calculate_sharpe_ratio(data, risk_free_rate=0.02):
            returns = data['Adj Close'].pct_change()
            excess_return = returns.mean() - risk_free_rate / 252
            return excess_return / returns.std() * np.sqrt(252)

        def calculate_sortino_ratio(data, risk_free_rate=0.02):
            returns = data['Adj Close'].pct_change()
            negative_return = returns[returns < 0]
            excess_return = returns.mean() - risk_free_rate / 252
            return excess_return / negative_return.std() * np.sqrt(252) if negative_return.std() != 0 else np.nan

        def calculate_max_drawdown(data):
            cumulative_return = (1 + data['Adj Close'].pct_change()).cumprod()
            cumulative_max = cumulative_return.cummax()
            drawdown = cumulative_return / cumulative_max - 1
            return drawdown.min() * 100

        metrics = {
            'CAGR': calculate_cagr(self.stock_data),
            'Sharpe Ratio': calculate_sharpe_ratio(self.stock_data),
            'Sortino Ratio': calculate_sortino_ratio(self.stock_data),
            'Max Drawdown': calculate_max_drawdown(self.stock_data)
        }

        return metrics

    def preprocess_and_merge_data(self):
        self.news_df['News'] = self.news_df['News'].apply(self._preprocess_text)
        self.stock_data = self.stock_data.reset_index()
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date']).dt.date
        self.news_df['Date'] = pd.to_datetime(self.news_df['Date']).dt.date
        self.merged_df = pd.merge(self.news_df, self.stock_data, on='Date', how='inner')

    def _preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text.lower().strip()

    def analyze_sentiment(self):
        analyzer = SentimentIntensityAnalyzer()
        self.merged_df['sentiment_score'] = self.merged_df['News'].apply(
            lambda text: analyzer.polarity_scores(text)['compound'])

    def feature_engineering(self):
        self.merged_df['Price_MA'] = self.merged_df['Close'].rolling(window=5).mean()
        self.merged_df['Price_Change'] = self.merged_df['Price_MA'].pct_change() * 100
        self.merged_df['Sentiment_MA_5'] = self.merged_df['sentiment_score'].rolling(window=5).mean()
        self.merged_df['Sentiment_MA_10'] = self.merged_df['sentiment_score'].rolling(window=10).mean()
        self.merged_df.fillna(0, inplace=True)

    def classify_sentiments(self):
        def classify_sentiment_and_signal(score):
            if score > 0.35:
                return 'Positive', 1
            elif score < 0:
                return 'Negative', -1
            else:
                return 'Neutral', 0

        self.merged_df['Sentiment_Label'], self.merged_df['Signal'] = zip(
            *self.merged_df['sentiment_score'].apply(classify_sentiment_and_signal))

    def train_model(self):
        X = self.merged_df[['sentiment_score', 'Price_Change', 'Sentiment_MA_5', 'Sentiment_MA_10']]
        y = self.merged_df['Signal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        return f"{np.round(accuracy * 100, 2)}%"

    def final_decision(self):
        positive_count = (self.merged_df['Signal'] == 1).sum()
        neutral_count = (self.merged_df['Sentiment_Label'] == 'Neutral').sum()
        negative_count = (self.merged_df['Signal'] == -1).sum()

        if positive_count > max(neutral_count, negative_count):
            return "Buy"
        elif neutral_count > max(positive_count, negative_count):
            return "Hold"
        else:
            return "Sell"

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

        # Fetch news and stock data
        analyzer.get_news_data(from_date=start_date, to_date=end_date)
        if not analyzer.get_stock_data(start_date=start_date, end_date=end_date):
            st.error("Invalid ticker symbol. Please try again.")
            return

        # Calculate performance metrics
        metrics = analyzer.calculate_metrics()

        # Preprocess and analyze sentiment
        analyzer.preprocess_and_merge_data()
        analyzer.analyze_sentiment()
        analyzer.feature_engineering()
        analyzer.classify_sentiments()

        # Train the model
        accuracy = analyzer.train_model()

        # Display results
        st.subheader("Performance Metrics")
        for key, value in metrics.items():
            st.write(f"{key}: {value:.2f}%" if key in ['CAGR', 'Max Drawdown'] else f"{key}: {value:.2f}")

        st.subheader("Model Accuracy")
        st.write("Accuracy:", accuracy)

        # Final decision with summary
        decision = analyzer.final_decision()
        sentiment_summary = f"The overall sentiment for {symbol} is {decision.lower()}, with recent trends indicating a likely {decision} recommendation."

        st.subheader("Final Decision")
        st.write(sentiment_summary)

if __name__ == "__main__":
    main()
