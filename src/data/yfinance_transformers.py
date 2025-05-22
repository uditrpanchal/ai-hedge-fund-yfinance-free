# This module transforms data fetched by `YFinanceDataFetcher` into Pydantic models.
# Due to reliance on `yfinance`, some data fields in the models will remain
# unpopulated (None) as yfinance does not provide them (e.g., CIK, SIC codes,
# news sentiment, specific insider trade details). Calculated financial metrics
# are best-effort based on available yfinance data.
"""
This module provides functions to transform financial data fetched by
`YFinanceDataFetcher` into structured Pydantic models. Each function
handles a specific type of financial data (e.g., prices, company facts,
financial statements, metrics), performing necessary data cleaning,
type conversions, and calculations.
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np # For np.nan handling
import logging
from src.data.yfinance_fetcher import YFinanceDataFetcher
from src.data.models import (
    Price, PriceResponse,
    CompanyFacts, CompanyFactsResponse,
    CompanyNews, CompanyNewsResponse,
    InsiderTrade, InsiderTradeResponse,
    LineItem, LineItemResponse,
    FinancialMetrics, FinancialMetricsResponse
)

# Configure basic logging (consistent with fetcher)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_price_response(ticker_symbol: str, period="1y", interval="1d", start_date=None, end_date=None) -> PriceResponse | None:
    """
    Fetches and transforms historical price data for a ticker into a PriceResponse object.

    Args:
        ticker_symbol (str): The stock ticker symbol.
        period (str): Data period to download (e.g., "1y", "1mo", "max").
        interval (str): Data interval (e.g., "1d", "1wk", "1mo").
        start_date (str, optional): Start date as "YYYY-MM-DD".
        end_date (str, optional): End date as "YYYY-MM-DD".

    Returns:
        PriceResponse | None: A PriceResponse object with price data,
                              or an empty PriceResponse if data is unavailable/invalid ticker,
                              or None if a major error occurs during fetching/transformation.
    """
    try:
        fetcher = YFinanceDataFetcher(ticker_symbol)
        if not fetcher.valid_ticker:
            logging.warning(f"get_price_response: Fetcher for {ticker_symbol} is invalid. Returning empty PriceResponse.")
            return PriceResponse(ticker=ticker_symbol, prices=[])
        
        prices_df = fetcher.get_historical_prices(period=period, interval=interval, start=start_date, end=end_date)
        if prices_df is None or prices_df.empty:
            logging.info(f"No historical price data found for {ticker_symbol} for the given parameters.")
            return PriceResponse(ticker=ticker_symbol, prices=[])
        
        prices_list = []
        for index, row in prices_df.iterrows():
            time_str = ""
            if isinstance(index, pd.Timestamp): # yfinance index is usually pd.Timestamp
                dt_object = index.to_pydatetime() if hasattr(index, 'to_pydatetime') else index
                time_str = dt_object.isoformat()
            elif isinstance(index, datetime): # Should already be datetime if not Timestamp
                time_str = index.isoformat()
            else: # Fallback for unexpected index types
                time_str = str(index)
            prices_list.append(Price(
                time=time_str,
                open=row.get('Open'), close=row.get('Close'), high=row.get('High'), low=row.get('Low'),
                volume=int(row.get('Volume')) if pd.notna(row.get('Volume')) else 0
            ))
        return PriceResponse(ticker=ticker_symbol, prices=prices_list)
    except Exception as e:
        logging.error(f"Error in get_price_response for {ticker_symbol}: {e}", exc_info=True)
        return None # Return None for major errors

def get_company_facts_response(ticker_symbol: str) -> CompanyFactsResponse | None:
    """
    Fetches and transforms general company information into a CompanyFactsResponse object.
    Many fields specific to SEC filings (CIK, SIC codes, etc.) are GAPs from yfinance
    and will be None.

    Args:
        ticker_symbol (str): The stock ticker symbol.

    Returns:
        CompanyFactsResponse | None: A CompanyFactsResponse with company data,
                                      or one with an empty CompanyFacts object if data is unavailable/invalid ticker,
                                      or None if a major error occurs.
    """
    try:
        fetcher = YFinanceDataFetcher(ticker_symbol)
        if not fetcher.valid_ticker:
            logging.warning(f"get_company_facts_response: Fetcher for {ticker_symbol} is invalid. Returning empty CompanyFacts.")
            return CompanyFactsResponse(company_facts=CompanyFacts(ticker=ticker_symbol, name=None))

        info_data = fetcher.get_info()
        if not info_data:
            logging.info(f"No company info data found for {ticker_symbol} via get_info.")
            return CompanyFactsResponse(company_facts=CompanyFacts(ticker=ticker_symbol, name=None))
            
        city, state, country = info_data.get('city', ''), info_data.get('state', ''), info_data.get('country', '')
        location = ", ".join(part for part in [city, state, country] if part if part) # Ensure parts are not empty
        
        company_facts_data = CompanyFacts(
            ticker=ticker_symbol, name=info_data.get('longName') or info_data.get('shortName'),
            industry=info_data.get('industry'), sector=info_data.get('sector'), exchange=info_data.get('exchange'),
            # is_active: Consider a company active if it has a recent market price.
            is_active=True if info_data.get('regularMarketPrice') is not None else False,
            location=location if location else None, # Ensure location is None if all parts were empty
            market_cap=info_data.get('marketCap'), number_of_employees=info_data.get('fullTimeEmployees'),
            website_url=info_data.get('website'), weighted_average_shares=info_data.get('sharesOutstanding'),
            # Explicitly None for GAPs (data not typically available from yfinance)
            cik=None, category=None, listing_date=None, sec_filings_url=None, 
            sic_code=None, sic_industry=None, sic_sector=None
        )
        return CompanyFactsResponse(company_facts=company_facts_data)
    except Exception as e:
        logging.error(f"Error in get_company_facts_response for {ticker_symbol}: {e}", exc_info=True)
        return None

def get_company_news_response(ticker_symbol: str) -> CompanyNewsResponse | None:
    """
    Fetches and transforms company news articles into a CompanyNewsResponse object.
    News sentiment is a GAP and will be None.

    Args:
        ticker_symbol (str): The stock ticker symbol.

    Returns:
        CompanyNewsResponse | None: A CompanyNewsResponse with news articles,
                                    or an empty one if data is unavailable/invalid ticker,
                                    or None if a major error occurs.
    """
    try:
        fetcher = YFinanceDataFetcher(ticker_symbol)
        if not fetcher.valid_ticker:
            logging.warning(f"get_company_news_response: Fetcher for {ticker_symbol} is invalid. Returning empty CompanyNewsResponse.")
            return CompanyNewsResponse(news=[])
            
        news_list_raw = fetcher.get_news()
        if not news_list_raw: 
            logging.info(f"No news data found for {ticker_symbol}.")
            return CompanyNewsResponse(news=[])
            
        news_list = []
        for item in news_list_raw: # item is a dict from yfinance
            publish_time, date_str = item.get('providerPublishTime'), None
            if publish_time: # Convert Unix timestamp to ISO format string
                try: date_str = datetime.fromtimestamp(int(publish_time)).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError) as e_ts: logging.warning(f"Error converting news timestamp for {ticker_symbol}: {e_ts}. Timestamp: {publish_time}")
            news_list.append(CompanyNews(
                ticker=ticker_symbol, title=item.get('title'), author=item.get('publisher'), # Using publisher as author
                source=item.get('publisher'), date=date_str, url=item.get('link'), 
                sentiment=None # Explicitly None for GAP
            ))
        return CompanyNewsResponse(news=news_list)
    except Exception as e:
        logging.error(f"Error in get_company_news_response for {ticker_symbol}: {e}", exc_info=True)
        return None

def get_insider_trades_response(ticker_symbol: str) -> InsiderTradeResponse | None:
    """
    Fetches and transforms insider trading data into an InsiderTradeResponse object.
    Some fields like `shares_owned_before_transaction` and `filing_date` are GAPs.

    Args:
        ticker_symbol (str): The stock ticker symbol.

    Returns:
        InsiderTradeResponse | None: An InsiderTradeResponse with transaction data,
                                      or an empty one if data is unavailable/invalid ticker,
                                      or None if a major error occurs.
    """
    try:
        fetcher = YFinanceDataFetcher(ticker_symbol)
        if not fetcher.valid_ticker:
            logging.warning(f"get_insider_trades_response: Fetcher for {ticker_symbol} is invalid. Returning empty InsiderTradeResponse.")
            return InsiderTradeResponse(insider_trades=[])
            
        transactions_df = fetcher.get_insider_transactions()
        if transactions_df is None or transactions_df.empty:
            logging.info(f"No insider trading data found for {ticker_symbol}.")
            return InsiderTradeResponse(insider_trades=[])
        
        insider_trades_list = []
        # Helper to safely get data from DataFrame row, considering yfinance column name variations
        def get_row_val(r, potential_names, default=None):
            for name in potential_names:
                val = r.get(name) # Use .get for Series to avoid KeyError if name is not present
                if pd.notna(val): return val
            return default

        for _, row in transactions_df.iterrows(): # Iterate through DataFrame rows
            name = get_row_val(row, ['Insider', 'Name'])
            title = get_row_val(row, ['Position', 'Title']) # 'Position' is more common in yfinance
            is_dir = bool("director" in str(title).lower() or "dir" in str(title).lower()) if title else False
            date_val = get_row_val(row, ['Start Date', 'Date'])
            date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d') if pd.notna(date_val) else None
            
            shares_val = get_row_val(row, ['Shares'], 0.0) # Default to 0.0 if not found
            value_val = get_row_val(row, ['Value']) # No default, might be missing
            
            shares = pd.to_numeric(shares_val, errors='coerce')
            value = pd.to_numeric(value_val, errors='coerce') if value_val is not None else None # Coerce only if value_val exists
            
            shares = 0.0 if pd.isna(shares) else shares # Ensure shares is a float
            price_ps = value / shares if shares != 0 and value is not None and pd.notna(value) else None
            
            after_shares_val = get_row_val(row, ['Shares Held After', 'Post Transaction Shares'])
            after_shares = pd.to_numeric(after_shares_val, errors='coerce')
            after_shares = int(after_shares) if pd.notna(after_shares) else None
            
            insider_trades_list.append(InsiderTrade(
                ticker=ticker_symbol, issuer=ticker_symbol, name=name, title=title, is_board_director=is_dir,
                transaction_date=date_str, transaction_shares=float(shares) if shares is not None else None,
                transaction_price_per_share=float(price_ps) if price_ps is not None else None,
                transaction_value=float(value) if value is not None and pd.notna(value) else None,
                shares_owned_after_transaction=after_shares, security_title="Common Stock", # Default security title
                # Explicitly None for GAPs
                shares_owned_before_transaction=None, filing_date=None 
            ))
        return InsiderTradeResponse(insider_trades=insider_trades_list)
    except Exception as e:
        logging.error(f"Error in get_insider_trades_response for {ticker_symbol}: {e}", exc_info=True)
        return None

def get_financial_statements_response(ticker_symbol: str) -> LineItemResponse | None:
    """
    Fetches and transforms financial statements (income, balance sheet, cash flow)
    for both annual and quarterly periods into a LineItemResponse object.
    Account names are dynamically added as fields to the LineItem model.

    Args:
        ticker_symbol (str): The stock ticker symbol.

    Returns:
        LineItemResponse | None: A LineItemResponse with financial statement line items,
                                  or an empty one if data is unavailable/invalid ticker,
                                  or None if a major error occurs.
    """
    try:
        fetcher = YFinanceDataFetcher(ticker_symbol)
        if not fetcher.valid_ticker:
            logging.warning(f"get_financial_statements_response: Fetcher for {ticker_symbol} is invalid. Returning empty LineItemResponse.")
            return LineItemResponse(search_results=[])
            
        info_data = fetcher.get_info() # Re-fetch info to get currency; fetcher handles caching
        currency = info_data.get('currency', 'USD') if info_data else 'USD'
        
        all_items = []
        for stype in ["income", "balance_sheet", "cashflow"]: # Statement types
            for speriod in ["annual", "quarterly"]: # Statement periods
                try:
                    stmt_df = fetcher.get_financials(stype, speriod)
                    if stmt_df is None or stmt_df.empty: 
                        logging.info(f"No data for {ticker_symbol}, statement: {stype}, period: {speriod}")
                        continue # Skip if no data for this specific statement/period
                    
                    # yfinance financial statement DataFrames have report dates as columns
                    for col_name in stmt_df.columns: 
                        report_period_str = pd.to_datetime(col_name).strftime('%Y-%m-%d') if pd.notna(col_name) else None
                        if not report_period_str: 
                            logging.warning(f"Could not parse date from column name '{col_name}' for {ticker_symbol}, {stype}, {speriod}.")
                            continue
                        
                        dynamic_fields = {} # To hold account names and their values for this period
                        for acc_name, val in stmt_df[col_name].items(): # acc_name is index (e.g., "Total Revenue")
                            # Sanitize account name to be a valid Python identifier for Pydantic field
                            valid_acc_name = ''.join(c for c in str(acc_name).replace(' ', '_').replace('-', '_').replace('/', '_').replace('&', 'and') if c.isalnum() or c == '_')
                            if valid_acc_name and valid_acc_name[0].isdigit(): valid_acc_name = "_" + valid_acc_name # Prepend underscore if starts with digit
                            
                            # Store value, converting to basic types if necessary
                            if pd.isna(val): dyn_fields[valid_acc_name] = None
                            elif isinstance(val, (int, float, bool, str)): dyn_fields[valid_acc_name] = val
                            else: # Attempt to convert other numeric types (e.g., np.int64) to float
                                try: dyn_fields[valid_acc_name] = float(val)
                                except (ValueError, TypeError): 
                                    logging.warning(f"Could not convert value '{val}' for account '{acc_name}' to float. Storing as string.")
                                    dyn_fields[valid_acc_name] = str(val)
                        
                        all_items.append(LineItem(
                            ticker=ticker_symbol, report_period=report_period_str, period=speriod, 
                            currency=currency, statement_type=stype, **dynamic_fields
                        ))
                except Exception as e_inner: 
                    logging.error(f"Error processing statement {stype} ({speriod}) for {ticker_symbol}: {e_inner}", exc_info=True)
        
        if not all_items:
            logging.info(f"No financial statement line items generated for {ticker_symbol}.")
        return LineItemResponse(search_results=all_items) # Return all collected items, even if empty
    except Exception as e_outer:
        logging.error(f"Major error in get_financial_statements_response for {ticker_symbol}: {e_outer}", exc_info=True)
        return None

# --- Helper functions for financial metrics ---
def _get_statement_value(df: pd.DataFrame | None, item_name: str, col_name_or_idx=0, default=None, prefer_float=True):
    """Safely extracts a value from a DataFrame row/column."""
    if df is None or df.empty or item_name not in df.index: return default
    try:
        # Determine the target column: if col_name_or_idx is a string, use it directly; otherwise, assume it's an integer index.
        target_col = col_name_or_idx if isinstance(col_name_or_idx, str) else df.columns[col_name_or_idx]
        if target_col not in df.columns: # Check if the resolved column name exists
            logging.debug(f"Column '{target_col}' not found in DataFrame for item '{item_name}'.")
            return default
        value = df.loc[item_name, target_col]
        if pd.isna(value) or value is None: return default
        if prefer_float:
            try: return float(value)
            except (ValueError, TypeError): # If float conversion fails, return as string
                logging.debug(f"Could not convert value '{value}' to float for item '{item_name}', returning as string.")
                return str(value) 
        return value 
    except (KeyError, IndexError) as e: # Catch errors if item_name or col_idx is invalid
        logging.debug(f"Error accessing item '{item_name}' at column '{col_name_or_idx}': {e}")
        return default

def _sum_last_n_quarters(df: pd.DataFrame | None, item_name: str, n: int = 4, default=None):
    """Sums the 'n' most recent quarterly values for an item, ensuring numeric conversion."""
    if df is None or df.empty or item_name not in df.index or len(df.columns) == 0 : return default
    num_cols_to_sum = min(n, len(df.columns)) # Use available columns if less than n
    if num_cols_to_sum == 0: return default
    try:
        # Select first num_cols_to_sum columns, convert to numeric (errors to NaN), fill NaN with 0, then sum.
        relevant_values = pd.to_numeric(df.loc[item_name].iloc[:num_cols_to_sum], errors='coerce').fillna(0)
        return relevant_values.sum() if relevant_values.notna().any() else default # Return sum or default if all were NaN
    except Exception as e: # Catch any other unexpected error during calculation
        logging.debug(f"Exception in _sum_last_n_quarters for item '{item_name}': {e}")
        return default

def get_financial_metrics_response(ticker_symbol: str, include_historical: bool = False) -> FinancialMetricsResponse | None:
    """
    Calculates and transforms financial metrics into a FinancialMetricsResponse object.
    Includes latest/TTM metrics and optionally historical annual metrics.
    Many metrics are sourced directly from `ticker.info`, while others are calculated
    from financial statements.

    Args:
        ticker_symbol (str): The stock ticker symbol.
        include_historical (bool): If True, attempts to calculate and include historical
                                   annual metrics. Defaults to False.

    Returns:
        FinancialMetricsResponse | None: A response object containing a list of
                                          FinancialMetrics objects (TTM and optionally historical),
                                          or an empty one if data is insufficient/invalid ticker,
                                          or None if a major error occurs.
    """
    try:
        fetcher = YFinanceDataFetcher(ticker_symbol)
        if not fetcher.valid_ticker:
            logging.warning(f"get_financial_metrics_response: Fetcher for {ticker_symbol} is invalid. Returning empty FinancialMetricsResponse.")
            return FinancialMetricsResponse(financial_metrics=[])

        info = fetcher.get_info()
        if not info: # If info is still empty even if valid_ticker was true (e.g. only history worked in init)
            logging.warning(f"No company info data found for {ticker_symbol} via get_info. Cannot calculate all metrics.")
            return FinancialMetricsResponse(financial_metrics=[]) # Return empty list if info is crucial and missing

        currency = info.get('currency', 'USD')
        all_metrics_list = []
        report_period_latest = datetime.today().strftime('%Y-%m-%d') # TTM report date is approximated as today
        period_latest = "ttm"

        # Fetch all necessary statements for TTM and historical calculations
        income_q, income_a = fetcher.get_financials('income', 'quarterly'), fetcher.get_financials('income', 'annual')
        bs_q, bs_a = fetcher.get_financials('balance_sheet', 'quarterly'), fetcher.get_financials('balance_sheet', 'annual')
        cf_q, cf_a = fetcher.get_financials('cashflow', 'quarterly'), fetcher.get_financials('cashflow', 'annual')

        shares_outstanding, market_cap = info.get('sharesOutstanding'), info.get('marketCap')

        # Initialize with all fields from Pydantic model to ensure all are present, defaulting to None
        latest_metrics_dict = {field: None for field in FinancialMetrics.__annotations__}
        # Update with basic info and directly available metrics from yfinance ticker.info
        latest_metrics_dict.update({
            "ticker": ticker_symbol, "report_period": report_period_latest, "period": period_latest, "currency": currency,
            "market_cap": market_cap, "enterprise_value": info.get('enterpriseValue'),
            "price_to_earnings_ratio": info.get('trailingPE'), "price_to_book_ratio": info.get('priceToBook'),
            "price_to_sales_ratio": info.get('priceToSalesTrailing12Months'),
            "enterprise_value_to_ebitda_ratio": info.get('enterpriseToEbitda'),
            "enterprise_value_to_revenue_ratio": info.get('enterpriseToRevenue'),
            "peg_ratio": info.get('pegRatio'), "return_on_equity": info.get('returnOnEquity'),
            "return_on_assets": info.get('returnOnAssets'), "current_ratio": info.get('currentRatio'),
            "quick_ratio": info.get('quickRatio'), "debt_to_equity": info.get('debtToEquity'),
            "revenue_growth": info.get('revenueGrowth'), "earnings_growth": info.get('earningsGrowth'),
            "earnings_per_share_growth": info.get('earningsQuarterlyGrowth'), 
            "payout_ratio": info.get('payoutRatio'), "earnings_per_share": info.get('trailingEps'),
            "book_value_per_share": info.get('bookValue'), "net_margin": info.get('profitMargins') # profitMargins is often net margin
        })

        # Calculate TTM metrics using helper functions
        revenue_ttm = _sum_last_n_quarters(income_q, 'Total Revenue', 4, _get_statement_value(income_a, 'Total Revenue'))
        cogs_ttm = _sum_last_n_quarters(income_q, 'Cost Of Revenue', 4, _get_statement_value(income_a, 'Cost Of Revenue', default=0.0))
        op_income_ttm = _sum_last_n_quarters(income_q, 'Operating Income', 4, _get_statement_value(income_a, 'Operating Income'))
        
        if revenue_ttm and revenue_ttm != 0: # Avoid division by zero for margin calculations
            if cogs_ttm is not None: latest_metrics_dict["gross_margin"] = (revenue_ttm - cogs_ttm) / revenue_ttm
            if op_income_ttm: latest_metrics_dict["operating_margin"] = op_income_ttm / revenue_ttm
        
        op_cashflow_ttm = _sum_last_n_quarters(cf_q, 'Total Cash From Operating Activities', 4, _get_statement_value(cf_a, 'Total Cash From Operating Activities'))
        capex_ttm = _sum_last_n_quarters(cf_q, 'Capital Expenditures', 4, _get_statement_value(cf_a, 'Capital Expenditures')) # Capex is usually negative
        
        fcf_ttm = None
        if op_cashflow_ttm is not None and capex_ttm is not None:
            fcf_ttm = op_cashflow_ttm + capex_ttm # Adding because capex is typically negative
            latest_metrics_dict["free_cash_flow"] = fcf_ttm
            if shares_outstanding and shares_outstanding != 0:
                latest_metrics_dict["free_cash_flow_per_share"] = fcf_ttm / shares_outstanding
            if market_cap and market_cap != 0 and fcf_ttm is not None: # Ensure fcf_ttm was calculated
                latest_metrics_dict["free_cash_flow_yield"] = fcf_ttm / market_cap

        total_assets_latest_annual = _get_statement_value(bs_a, 'Total Assets') # Use latest annual assets for turnover
        if revenue_ttm and total_assets_latest_annual and total_assets_latest_annual != 0:
            latest_metrics_dict["asset_turnover"] = revenue_ttm / total_assets_latest_annual
        
        # Final type conversion for all numeric metric fields
        for key, value in latest_metrics_dict.items():
            if key not in ["ticker", "report_period", "period", "currency"]: # Skip non-metric fields
                if isinstance(value, (int, float, np.number)) and pd.notna(value): # Check if it's a number and not NaN
                    latest_metrics_dict[key] = float(value)
                else: # Ensure it's None if not a valid number (e.g. string, or was already None/NaN)
                    latest_metrics_dict[key] = None
        
        all_metrics_list.append(FinancialMetrics(**latest_metrics_dict))

        # --- Historical Metrics (Annual) ---
        if include_historical and income_a is not None and not income_a.empty:
            # Get unique, sorted dates from annual income statement columns
            historical_statement_dates = sorted([pd.to_datetime(col, errors='coerce') for col in income_a.columns if pd.to_datetime(col, errors='coerce') is not pd.NaT])

            for col_date_obj in historical_statement_dates:
                report_period_hist = col_date_obj.strftime('%Y-%m-%d')
                # Find original column name string for this date to pass to _get_statement_value
                # This handles cases where column names might be full timestamps or just date strings
                original_col_name_options = [
                    col for col in income_a.columns if pd.to_datetime(col, errors='coerce') == col_date_obj
                ]
                if not original_col_name_options: continue # Should not happen if date came from columns
                col_name_str = original_col_name_options[0]


                hist_metrics_data = {field: None for field in FinancialMetrics.__annotations__} # Initialize all fields
                hist_metrics_data.update({ # Basic info for this historical period
                    "ticker": ticker_symbol, "report_period": report_period_hist, "period": "annual", "currency": currency,
                    # Price-based ratios require historical prices, not calculated here, so explicitly None
                    "market_cap": None, "price_to_earnings_ratio": None, 
                    "price_to_book_ratio": None, "price_to_sales_ratio": None,
                })

                # Calculate metrics using data for this specific historical column (col_name_str)
                revenue_h = _get_statement_value(income_a, 'Total Revenue', col_name_str)
                cogs_h = _get_statement_value(income_a, 'Cost Of Revenue', col_name_str, default=0.0)
                op_income_h = _get_statement_value(income_a, 'Operating Income', col_name_str)
                net_income_h = _get_statement_value(income_a, 'Net Income', col_name_str)
                
                if revenue_h and revenue_h != 0: # Avoid division by zero
                    if cogs_h is not None: hist_metrics_data["gross_margin"] = (revenue_h - cogs_h) / revenue_h
                    if op_income_h: hist_metrics_data["operating_margin"] = op_income_h / revenue_h
                    if net_income_h: hist_metrics_data["net_margin"] = net_income_h / revenue_h
                
                assets_h = _get_statement_value(bs_a, 'Total Assets', col_name_str) # Match balance sheet date
                liab_h = _get_statement_value(bs_a, 'Total Liab', col_name_str)   # Match balance sheet date
                if assets_h and liab_h: hist_metrics_data["book_value"] = assets_h - liab_h # Book value for that year

                op_cf_h = _get_statement_value(cf_a, 'Total Cash From Operating Activities', col_name_str) # Match cash flow date
                capex_h = _get_statement_value(cf_a, 'Capital Expenditures', col_name_str)           # Match cash flow date
                if op_cf_h and capex_h: hist_metrics_data["free_cash_flow"] = op_cf_h + capex_h

                if revenue_h and assets_h and assets_h != 0:
                    hist_metrics_data["asset_turnover"] = revenue_h / assets_h
                
                # Final type conversion for all numeric metric fields
                for key, value in hist_metrics_data.items():
                     if key not in ["ticker", "report_period", "period", "currency"]:
                        if isinstance(value, (int, float, np.number)) and pd.notna(value): hist_metrics_data[key] = float(value)
                        else: hist_metrics_data[key] = None
                
                all_metrics_list.append(FinancialMetrics(**hist_metrics_data))
        
        if not all_metrics_list: # Should at least have TTM if info was available
            logging.info(f"No financial metrics could be generated for {ticker_symbol}.")
            return FinancialMetricsResponse(financial_metrics=[])

        return FinancialMetricsResponse(financial_metrics=all_metrics_list)

    except Exception as e:
        logging.error(f"Major error in get_financial_metrics_response for {ticker_symbol}: {e}", exc_info=True)
        return None
