# This module uses the `yfinance` library, which scrapes data from Yahoo Finance.
# Users should be aware of Yahoo Finance's terms of use, as `yfinance` is
# intended for personal and educational use. Data accuracy and availability
# are subject to Yahoo Finance.
"""
This module provides the YFinanceDataFetcher class, responsible for interacting
with the yfinance library to retrieve various financial data for a given stock ticker.
It includes error handling and a validity check for ticker symbols.
"""

import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta # Add timedelta

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YFinanceDataFetcher:
    """
    A data fetcher class that uses the yfinance library to retrieve stock data.
    It encapsulates yfinance calls and provides a basic validation mechanism
    to check if a ticker symbol is likely valid upon initialization.
    """
    def __init__(self, ticker_symbol: str):
        """
        Initializes the YFinanceDataFetcher with a specific ticker symbol.

        Sets an internal flag `self.valid_ticker` based on successful Ticker object
        initialization and a basic data fetch attempt (e.g., trying to access ticker.info
        or ticker.history).

        Args:
            ticker_symbol (str): The stock ticker symbol (e.g., "AAPL").
        """
        self.ticker_symbol = ticker_symbol
        self.ticker = None # Holds the yf.Ticker object
        self.valid_ticker = False # Internal flag indicating if the ticker seems valid

        if not ticker_symbol:
            logging.error("Ticker symbol cannot be empty.")
            return # self.valid_ticker remains False

        try:
            self.ticker = yf.Ticker(ticker_symbol)
            # Perform a quick check to see if the ticker is likely valid
            # by trying to access basic information.
            # yfinance often returns an empty info dict for invalid tickers after the first call.
            # A more robust check involves seeing if 'regularMarketPrice' exists or if history is non-empty.
            if self.ticker.info and self.ticker.info.get('regularMarketPrice') is not None:
                self.valid_ticker = True
            elif self.ticker.info and not self.ticker.info: # .info call was made, but it's an empty dict
                 logging.warning(f"Ticker info is empty for {ticker_symbol} after initial fetch. Ticker may be invalid or delisted.")
                 self.valid_ticker = False
            else: # If .info is None or some other state, try history as a fallback.
                history_check = self.ticker.history(period="1d")
                if not history_check.empty:
                    self.valid_ticker = True
                    logging.info(f"Ticker {ticker_symbol} validated using history check.")
                else:
                    logging.warning(f"Failed to fetch initial data (info or 1d history) for {ticker_symbol}. Ticker may be invalid.")
                    self.valid_ticker = False
        except Exception as e:
            # This can catch various issues: network problems, yfinance internal errors for specific symbols, etc.
            logging.error(f"Exception during yf.Ticker initialization or validation for {ticker_symbol}: {e}")
            self.ticker = None # Ensure ticker object is not stored if invalid
            self.valid_ticker = False

    def get_info(self) -> dict:
        """
        Fetches general company information for the ticker using `ticker.info`.

        Returns:
            dict: A dictionary containing company information, or an empty dict if
                  the ticker is invalid, data cannot be fetched, or info is empty.
        """
        if not self.valid_ticker or not self.ticker:
            logging.info(f"get_info: Ticker {self.ticker_symbol} is invalid or not initialized. Returning empty data.")
            return {}
        try:
            # The self.ticker.info attribute caches the result. If it was empty during __init__
            # and the ticker is somehow still valid (e.g. validated by history), this might still be empty.
            # A fresh call might be desired if caching behavior is an issue, but yfinance handles this.
            info_data = self.ticker.info
            if not info_data: # Check if the info dict is empty
                logging.warning(f"get_info: Info data is empty for ticker {self.ticker_symbol}.")
                return {}
            return info_data
        except Exception as e:
            logging.error(f"Error fetching info for {self.ticker_symbol}: {e}")
            return {}

    def get_historical_prices(self, period="1y", interval="1d", start_date=None, end_date=None) -> pd.DataFrame:
        """
        Fetches historical market price data using `ticker.history()`.

        Args:
            period (str): The period for which to fetch data (e.g., "1y", "1mo", "max").
            interval (str): The interval of data points (e.g., "1d", "1wk", "1mo").
            start_date (str, optional): Start date string (YYYY-MM-DD). Defaults to None.
            end_date (str, optional): End date string (YYYY-MM-DD). Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing historical price data, or an empty
                          DataFrame if the ticker is invalid or data cannot be fetched.
        """
        if not self.valid_ticker or not self.ticker:
            logging.info(f"get_historical_prices: Ticker {self.ticker_symbol} is invalid or not initialized. Returning empty data.")
            return pd.DataFrame()
        
        effective_end_date = end_date
        if start_date and end_date and start_date == end_date:
            try:
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                effective_end_date = (end_date_obj + timedelta(days=4)).strftime("%Y-%m-%d") # Changed to days=4
                logging.info(f"Adjusted end_date to {effective_end_date} for single-day fetch query on {start_date} for ticker {self.ticker_symbol}.") # Updated log message
            except ValueError:
                logging.warning(f"Could not parse end_date '{end_date}' to adjust for single-day fetch query. Using original end_date.") # Updated log message
                # Keep effective_end_date as original end_date
        
        logging.info(f"YFinanceFetcher: Attempting to fetch history for {self.ticker_symbol} with period={period}, interval={interval}, start={start_date}, end={effective_end_date}")

        try:
            # Use effective_end_date in the history call
            history_data = self.ticker.history(period=period, interval=interval, start=start_date, end=effective_end_date)
            logging.info(f"YFinanceFetcher: For {self.ticker_symbol}, history_data is empty: {history_data.empty}. Shape: {history_data.shape}. Head: {history_data.head().to_string() if not history_data.empty else 'N/A'}")
            return history_data
        except Exception as e:
            logging.error(f"YFinanceFetcher: Exception fetching historical prices for {self.ticker_symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def get_financials(self, statement_type="financials", period="annual") -> pd.DataFrame:
        """
        Fetches financial statements (e.g., income statement, balance sheet, cash flow).
        Maps common statement type names to specific yfinance Ticker object attributes.

        Args:
            statement_type (str): Type of financial statement. Expected values:
                                  "financials" (maps to income statement),
                                  "income" (maps to income statement),
                                  "balance_sheet",
                                  "cashflow".
            period (str): "annual" or "quarterly".

        Returns:
            pd.DataFrame: A DataFrame containing the requested financial statement,
                          or an empty DataFrame if the ticker is invalid, data cannot
                          be fetched, or the statement type is invalid.
        """
        if not self.valid_ticker or not self.ticker:
            logging.info(f"get_financials: Ticker {self.ticker_symbol} is invalid or not initialized. Returning empty data.")
            return pd.DataFrame()
        try:
            data = pd.DataFrame() # Default to empty DataFrame
            if statement_type == "financials" or statement_type == "income": # Treat "financials" as "income"
                data = self.ticker.financials if period == "annual" else self.ticker.quarterly_financials
            elif statement_type == "balance_sheet":
                data = self.ticker.balance_sheet if period == "annual" else self.ticker.quarterly_balance_sheet
            elif statement_type == "cashflow":
                data = self.ticker.cashflow if period == "annual" else self.ticker.quarterly_cashflow
            else:
                logging.warning(f"Invalid statement_type '{statement_type}' requested for ticker {self.ticker_symbol}.")
                return pd.DataFrame() # Return empty for invalid type
            
            if data is None or data.empty: # yfinance might return None or empty DF for missing statements
                logging.info(f"No {period} {statement_type} data found for {self.ticker_symbol}.")
                return pd.DataFrame() # Ensure consistent empty DF return
            return data

        except Exception as e:
            logging.error(f"Error fetching {period} {statement_type} for {self.ticker_symbol}: {e}")
            return pd.DataFrame()

    def get_news(self) -> list:
        """
        Fetches news articles related to the ticker using `ticker.news`.

        Returns:
            list: A list of news articles (dictionaries), or an empty list if
                  the ticker is invalid or data cannot be fetched.
        """
        if not self.valid_ticker or not self.ticker:
            logging.info(f"get_news: Ticker {self.ticker_symbol} is invalid or not initialized. Returning empty data.")
            return []
        try:
            news_data = self.ticker.news
            if news_data is None: # Explicitly check for None, though yfinance usually returns list
                logging.info(f"News data for {self.ticker_symbol} was None.")
                return []
            return news_data
        except Exception as e:
            logging.error(f"Error fetching news for {self.ticker_symbol}: {e}")
            return []

    def get_insider_transactions(self) -> pd.DataFrame:
        """
        Fetches insider trading transaction data using `ticker.insider_transactions`.

        Note: The availability and structure of this data can change in the yfinance library.

        Returns:
            pd.DataFrame: A DataFrame containing insider transactions, or an empty
                          DataFrame if the ticker is invalid, data cannot be fetched,
                          or the underlying yfinance method is unavailable/returns no data.
        """
        if not self.valid_ticker or not self.ticker:
            logging.info(f"get_insider_transactions: Ticker {self.ticker_symbol} is invalid or not initialized. Returning empty data.")
            return pd.DataFrame()
        try:
            transactions = self.ticker.insider_transactions
            if transactions is None: # yfinance might return None if no data
                 logging.info(f"No insider transactions data found for {self.ticker_symbol} (method returned None).")
                 return pd.DataFrame()
            if transactions.empty:
                 logging.info(f"Insider transactions data is empty for {self.ticker_symbol}.")
            return transactions
        except AttributeError: # If the 'insider_transactions' attribute itself doesn't exist
             logging.warning(f"'insider_transactions' attribute not found for {self.ticker_symbol}. The yfinance API might have changed or this data is unavailable.")
             return pd.DataFrame()
        except Exception as e: # Catch any other exceptions during the fetch
            logging.error(f"Error fetching insider transactions for {self.ticker_symbol}: {e}")
            return pd.DataFrame()
