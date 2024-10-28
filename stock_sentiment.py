import pandas as pd
import requests
import streamlit as st
from datetime import datetime
import numpy as np

class StockSentimentAnalyzer:
    def __init__(self, stock_ticker, api_key):
        self.stock_ticker = stock_ticker
        self.api_key = api_key
        self.stock_data = pd.DataFrame()
        self.news_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()

    def get_news_data(self, from_date, to_date):
        url = f'https://finnhub.io/api/v1/company-news?symbol={self.stock_ticker}&from={from_date}&to={to_date}&token={self.api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            self.news_df = pd.DataFrame(response.json())
            # Check if the 'Date' column is in correct format
            if 'date' in self.news_df.columns:
                self.news_df['Date'] = pd.to_datetime(self.news_df['date'], unit='s')
                self.news_df = self.news_df[['Date', 'headline', 'summary']]
                return True
        return False

    def get_stock_data(self, start_date, end_date):
        url = f'https://finnhub.io/api/v1/quote?symbol={self.stock_ticker}&token={self.api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            stock_data = response.json()
            stock_price_data = {
                'Date': [start_date, end_date],
                'Close': [stock_data['c'], stock_data['c']]  # Dummy data for example
            }
            self.stock_data = pd.DataFrame(stock_price_data)
            self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
            return True
        return False

    def preprocess_and_merge_data(self):
        if self.stock_data.empty or self.news_df.empty:
            raise ValueError("Stock data or news data is empty, cannot merge.")

        self.stock_data.reset_index(inplace=True)
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        self.news_df['Date'] = pd.to_datetime(self.news_df['Date'])

        self.merged_df = pd.merge(self.stock_data, self.news_df, how='left', on='Date')
        self.merged_df['Sentiment'] = self.merged_df['summary'].apply(self.analyze_sentiment)
        self.merged_df['Sentiment'].fillna(0, inplace=True)

    def analyze_sentiment(self, text):
        # Dummy sentiment analysis logic
        if 'good' in text.lower():
            return 1
        elif 'bad' in text.lower():
            return -1
        return 0

    def feature_engineering(self):
        # Add features based on the merged DataFrame
        self.merged_df['CAGR'] = self.calculate_cagr()
        self.merged_df['Sharpe Ratio'] = self.calculate_sharpe_ratio()

    def calculate_cagr(self):
        start_price = self.merged_df['Close'].iloc[0]
        end_price = self.merged_df['Close'].iloc[-1]
        num_years = (self.merged_df['Date'].iloc[-1] - self.merged_df['Date'].iloc[0]).days / 365.25
        return (end_price / start_price) ** (1 / num_years) - 1

    def calculate_sharpe_ratio(self, risk_free_rate=0.01):
        daily_returns = self.merged_df['Close'].pct_change()
        excess_returns = daily_returns - risk_free_rate / 252  # Assume 252 trading days
        return np.mean(excess_returns) / np.std(excess_returns)

def fetch_and_process_data(analyzer, start_date, end_date):
    analyzer.get_news_data(from_date=start_date, to_date=end_date)
    if not analyzer.get_stock_data(start_date, end_date):
        return False

    try:
        analyzer.preprocess_and_merge_data()
        analyzer.feature_engineering()
        return True
    except ValueError as e:
        st.error(str(e))
        return False

def main():
    st.title("Stock Sentiment Analyzer")
    stock_ticker = st.text_input("Enter stock ticker:", "AAPL")
    api_key = st.text_input("Enter API Key:", "YOUR_API_KEY")
    start_date = st.date_input("Start date", datetime(2023, 1, 1))
    end_date = st.date_input("End date", datetime(2023, 12, 31))

    if st.button("Analyze"):
        analyzer = StockSentimentAnalyzer(stock_ticker, api_key)
        if fetch_and_process_data(analyzer, start_date, end_date):
            st.success("Data fetched and processed successfully!")
            st.write(analyzer.merged_df)
            st.write("CAGR:", analyzer.merged_df['CAGR'].iloc[0])
            st.write("Sharpe Ratio:", analyzer.merged_df['Sharpe Ratio'].iloc[0])
        else:
            st.error("Failed to fetch data.")

if __name__ == "__main__":
    main()
