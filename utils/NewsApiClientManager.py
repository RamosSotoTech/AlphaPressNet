from newsapi.newsapi_client import NewsApiClient
from text_feature_extraction import BertFeatureExtractor
import datetime
import AlphaPressNet.settings as settings
import environ
import pandas as pd


def _filter_results(results):
    """
    Filters out articles with '[Removed]' attributes.

    :param results: The raw results from newsapi.
    :return: Filtered results.
    """
    filtered_articles = []
    for article in results.get('articles', []):
        if (article['title'] != '[Removed]' and article['description'] != '[Removed]'
                and article['content'] != '[Removed]'):
            filtered_articles.append(article)
    results['articles'] = filtered_articles
    return results


def extract_article_features(articles, bert_model_name="yiyanghkust/finbert-tone"):
    """
    Extracts features from the articles using BERT.
    :param bert_model_name:
    :param articles: raw articles from newsapi.
    :return: articles with features.
    """
    for article in articles:
        content = article['content']
        if not content:
            continue
        bert_feature_extractor = BertFeatureExtractor(bert_model_name)
        features = bert_feature_extractor.extract_features(content)
        article['features'] = features
        # todo: generalize this to any number of words
        article['top_10_words'] = bert_feature_extractor.get_top_n_important_words(content, 10)
    return articles


class NewsApiClientManager:
    def __init__(self, api_key=None):
        """
        Initializes the NewsManager with environment API key.
        """
        # Initialise environment variables
        environ.Env.read_env()
        if not api_key:
            api_key = settings.NEWSAPI_KEY
        self.client = NewsApiClient(api_key=api_key)

    def get_top_headlines(self, country='us', category=None, sources=None, q=None, page_size=20, include_removed=False):
        """
        Gathers top headlines from newsapi.

        :param country: The country code for the news source.
        :param category: Category of the news (e.g., 'business', 'technology').
        :param sources: A list of sources to be considered.
        :param q: Keywords or phrases to search.
        :param page_size: Number of results to return per request. Maximum is 100.
        :param include_removed: Whether to include articles with '[Removed]' attributes.
        :return: A dictionary containing the top headlines.
        """
        headlines = self.client.get_top_headlines(country=country, category=category, sources=sources, q=q,
                                                  page_size=page_size)
        return headlines if not include_removed else _filter_results(headlines)

    def get_everything(self, q=None, sources=None, domains=None, exclude_domains=None, from_param=None, to=None,
                       language='en', sort_by='relevancy', page_size=20, include_removed=False):
        """
        Gathers all news articles from newsapi based on the given parameters.

        :param q: Keywords or phrases to search.
        :param sources: A list of sources to be considered.
        :param domains: A list of domains where the news originates.
        :param exclude_domains: A list of domains to be excluded.
        :param from_param: A start date for the news articles.
        :param to: An end date for the news articles.
        :param language: The language for the news articles.
        :param sort_by: The criteria for sorting the results.
        :param page_size: Number of results to return per request. Maximum is 100.
        :param include_removed: Whether to include articles with '[Removed]' attributes.
        :return: A dictionary containing all the relevant news articles.
        """
        if not from_param:
            from_param = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        news = self.client.get_everything(q=q, sources=sources, domains=domains, exclude_domains=exclude_domains,
                                          from_param=from_param, to=to, language=language, sort_by=sort_by,
                                          page_size=page_size)
        return news if not include_removed else _filter_results(news)

    def get_sources(self, category=None, country='us', language='en'):
        """
        Retrieves the list of news sources from newsapi.

        :param category: Category of the news (e.g., 'business', 'technology').
        :param country: The country code for the news source.
        :param language: The language for the news articles.
        :return: A dictionary containing all the available sources.
        """
        return self.client.get_sources(category=category, country=country, language=language)

    def get_articles_and_features(self, *args, **kwargs):
        # Get articles using one of your existing methods, e.g., get_everything
        articles = self.get_everything(*args, **kwargs)
        # Extract features from the articles
        articles_with_features = extract_article_features(articles['articles'])
        # Convert the articles with features to a pandas DataFrame
        df = self._to_dataframe(articles_with_features)
        return df

    @staticmethod
    def _to_dataframe(articles):
        # Extracting the needed information from the articles
        data = []
        for article in articles:
            info = {
                'title': article['title'],
                'description': article['description'],
                'content': article['content'],
                'features': article.get('features', None)
            }
            data.append(info)

        # Creating a DataFrame
        df = pd.DataFrame(data)
        return df
