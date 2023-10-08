import yfinance as yf


class StockInfo:

    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)

    def get_history(self, period="1y"):
        # todo: Add error handling in case the ticker data retrieval fails
        return self.ticker.history(period=period)

    def get_dividends(self):
        return self.ticker.dividends

    def get_splits(self):
        return self.ticker.splits

    def get_info(self):
        return self.ticker.info

    def get_recommendations(self):
        # note: The recommendations can be large and may take time to download.
        return self.ticker.recommendations

    def get_website(self):
        return self.ticker.info.get('website', None)

    def get_industry(self):
        return self.ticker.info.get('industry', None)

    def get_company_officers(self):
        return self.ticker.info.get('companyOfficers', None)

    def get_short_name(self):
        return self.ticker.info.get('shortName', None)

    def get_long_name(self):
        return self.ticker.info.get('longName', None)


# Sample usage:
if __name__ == "__main__":
    stock = StockInfo("AAPL")
    print(stock.get_website())
    print(stock.get_industry())
    print(stock.get_company_officers())
    print(stock.get_short_name())
    print(stock.get_long_name())
    # todo: Implement additional functionalities or visualization features for stock data
