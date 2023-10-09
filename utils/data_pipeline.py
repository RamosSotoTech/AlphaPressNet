import pandas as pd
from utils.yfinance_pipeline import StockInfo
from utils.NewsManager import NewsManager


class CombinedFeatureExtractor:

    def __init__(self, ticker_symbol, period="1y", bert_model_name="yiyanghkust/finbert-tone"):
        self.stock_info = StockInfo(ticker_symbol, bert_model_name)
        self.news_manager = NewsManager(bert_model_name=bert_model_name)
        self.period = period

    def _convert_article_to_dataframe(self, articles):
        df = pd.DataFrame(articles)
        df['date'] = pd.to_datetime(df['publishedAt']).dt.date  # Convert datetime to date for merging
        return df

    def get_combined_features(self):
        # Extract stock features
        stock_data = self.stock_info.extract_features(period=self.period)
        stock_data['date'] = stock_data.index.date  # Convert datetime index to date column for merging

        # Extract text features
        text_features = self.stock_info.extract_text_features()

        # Fetch news articles for the period
        start_date = stock_data.index.min().strftime('%Y-%m-%d')
        end_date = stock_data.index.max().strftime('%Y-%m-%d')
        news_articles = self.news_manager.get_everything(from_param=start_date, to=end_date)
        news_df = self._convert_article_to_dataframe(news_articles['articles'])

        # Combine text features with stock data (same approach as before)
        for feature_name, feature_value in text_features.items():
            stock_data[feature_name] = feature_value

        # Merge stock data and news features based on date
        combined_df = pd.merge(stock_data, news_df[['date']], on='date', how='inner')

        return combined_df


# Sample usage
if __name__ == "__main__":
    combined_extractor = CombinedFeatureExtractor("AAPL")
    combined_features = combined_extractor.get_combined_features()
    print(combined_features)
