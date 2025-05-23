"""
Unit tests for the yfinance_transformers module.

This test suite uses pytest and unittest.mock to test the transformer functions
that convert data fetched by YFinanceDataFetcher into Pydantic models.
It focuses on mocking the underlying yfinance library calls to ensure
transformers behave correctly under various scenarios, including valid data,
empty data, invalid tickers, and API errors.
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import numpy as np # For NaN

# Expected Pydantic Models
from src.data.models import (
    PriceResponse, Price,
    FinancialMetricsResponse, FinancialMetrics,
    CompanyFactsResponse, CompanyFacts
)
# Functions to test
from src.data.yfinance_transformers import (
    get_price_response,
    get_financial_metrics_response,
    get_company_facts_response
)
# The class we are effectively mocking the instance of
from src.data.yfinance_fetcher import YFinanceDataFetcher


# --- Mock Data ---
# This section defines reusable mock data structures that mimic responses
# from yfinance library calls, used across multiple test cases.

MOCK_VALID_AAPL_INFO = {
    'shortName': 'Apple Inc.',
    'currency': 'USD',
    'regularMarketPrice': 150.0, 
    'marketCap': 2.0e12,
    'trailingPE': 25.0,
    'priceToBook': 5.0,
    'priceToSalesTrailing12Months': 6.0,
    'enterpriseToEbitda': 20.0,
    'enterpriseToRevenue': 7.0,
    'pegRatio': 1.8,
    'returnOnEquity': 0.5,
    'returnOnAssets': 0.15,
    'currentRatio': 1.5,
    'quickRatio': 1.0,
    'debtToEquity': 120.0,
    'revenueGrowth': 0.1,
    'earningsGrowth': 0.12,
    'earningsQuarterlyGrowth': 0.05,
    'payoutRatio': 0.2,
    'trailingEps': 6.0,
    'bookValue': 30.0,
    'profitMargins': 0.21,
    'sharesOutstanding': 16e9,
    'enterpriseValue': 2.1e12,
    'industry': 'Technology',
    'sector': 'Consumer Electronics',
    'exchange': 'NMS',
    'fullTimeEmployees': 150000,
    'website': 'http://www.apple.com',
    'city': 'Cupertino',
    'state': 'CA',
    'country': 'United States',
}

MOCK_INVALID_TICKER_INFO = {} 

MOCK_AAPL_HISTORY = pd.DataFrame({
    'Open': [150.0, 151.0], 'High': [152.0, 152.5], 'Low': [149.0, 150.5],
    'Close': [151.0, 151.5], 'Volume': [1000000, 1200000]
}, index=[pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')])

MOCK_AAPL_INCOME_ANNUAL = pd.DataFrame({
    pd.Timestamp('2023-12-31'): {'Total Revenue': 300e9, 'Cost Of Revenue': 180e9, 'Operating Income': 70e9, 'Net Income': 63e9},
    pd.Timestamp('2022-12-31'): {'Total Revenue': 280e9, 'Cost Of Revenue': 170e9, 'Operating Income': 65e9, 'Net Income': 60e9}
})

MOCK_AAPL_INCOME_QUARTERLY = pd.DataFrame({
    pd.Timestamp('2023-12-31'): {'Total Revenue': 80e9, 'Cost Of Revenue': 45e9, 'Operating Income': 20e9, 'Net Income': 18e9},
    pd.Timestamp('2023-09-30'): {'Total Revenue': 75e9, 'Cost Of Revenue': 40e9, 'Operating Income': 18e9, 'Net Income': 16e9},
    pd.Timestamp('2023-06-30'): {'Total Revenue': 70e9, 'Cost Of Revenue': 35e9, 'Operating Income': 17e9, 'Net Income': 15e9},
    pd.Timestamp('2023-03-31'): {'Total Revenue': 75e9, 'Cost Of Revenue': 40e9, 'Operating Income': 15e9, 'Net Income': 14e9}
})

MOCK_AAPL_BALANCE_SHEET_ANNUAL = pd.DataFrame({
    pd.Timestamp('2023-12-31'): {'Total Assets': 350e9, 'Total Liab': 200e9},
    pd.Timestamp('2022-12-31'): {'Total Assets': 340e9, 'Total Liab': 190e9}
})
MOCK_AAPL_BALANCE_SHEET_QUARTERLY = pd.DataFrame()

MOCK_AAPL_CASHFLOW_ANNUAL = pd.DataFrame({
    pd.Timestamp('2023-12-31'): {'Total Cash From Operating Activities': 90e9, 'Capital Expenditures': -15e9},
    pd.Timestamp('2022-12-31'): {'Total Cash From Operating Activities': 85e9, 'Capital Expenditures': -14e9}
})
MOCK_AAPL_CASHFLOW_QUARTERLY = pd.DataFrame({
    pd.Timestamp('2023-12-31'): {'Total Cash From Operating Activities': 25e9, 'Capital Expenditures': -4e9},
    pd.Timestamp('2023-09-30'): {'Total Cash From Operating Activities': 22e9, 'Capital Expenditures': -3e9},
    pd.Timestamp('2023-06-30'): {'Total Cash From Operating Activities': 20e9, 'Capital Expenditures': -3e9},
    pd.Timestamp('2023-03-31'): {'Total Cash From Operating Activities': 23e9, 'Capital Expenditures': -5e9}
})


# --- Test Cases ---

@patch('src.data.yfinance_fetcher.yf.Ticker')
def test_get_price_response_success(mock_yf_ticker_class):
    """
    Tests `get_price_response` for a successful scenario where valid historical
    price data is returned and transformed.
    """
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = MOCK_VALID_AAPL_INFO # Needed for YFinanceDataFetcher init validation
    mock_ticker_instance.history.return_value = MOCK_AAPL_HISTORY
    mock_yf_ticker_class.return_value = mock_ticker_instance

    result = get_price_response('AAPL')

    assert result is not None
    assert isinstance(result, PriceResponse)
    assert result.ticker == 'AAPL'
    assert len(result.prices) == 2
    assert isinstance(result.prices[0], Price)
    assert result.prices[0].open == 150.0
    assert result.prices[0].time == '2023-01-01T00:00:00'
    mock_yf_ticker_class.assert_called_once_with('AAPL')
    mock_ticker_instance.history.assert_called_once()


@patch('src.data.yfinance_fetcher.yf.Ticker')
def test_get_price_response_empty_history(mock_yf_ticker_class):
    """
    Tests `get_price_response` when `ticker.history()` returns an empty DataFrame.
    Expected behavior is an empty PriceResponse (prices list is empty).
    """
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = MOCK_VALID_AAPL_INFO
    mock_ticker_instance.history.return_value = pd.DataFrame()
    mock_yf_ticker_class.return_value = mock_ticker_instance

    result = get_price_response('AAPL_EMPTY')

    assert result is not None
    assert isinstance(result, PriceResponse)
    assert result.ticker == 'AAPL_EMPTY'
    assert len(result.prices) == 0
    mock_yf_ticker_class.assert_called_once_with('AAPL_EMPTY')


@patch('src.data.yfinance_fetcher.yf.Ticker')
def test_get_price_response_invalid_ticker(mock_yf_ticker_class):
    """
    Tests `get_price_response` when the ticker is considered invalid by
    `YFinanceDataFetcher` (e.g., `ticker.info` is empty).
    Expected behavior is an empty PriceResponse.
    """
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = MOCK_INVALID_TICKER_INFO 
    mock_ticker_instance.history.return_value = pd.DataFrame() # For YFinanceDataFetcher init
    mock_yf_ticker_class.return_value = mock_ticker_instance
    
    result = get_price_response('INVALIDTICKER')

    assert result is not None
    assert isinstance(result, PriceResponse)
    assert result.ticker == 'INVALIDTICKER'
    assert len(result.prices) == 0 
    mock_yf_ticker_class.assert_called_once_with('INVALIDTICKER')


@patch('src.data.yfinance_fetcher.yf.Ticker')
def test_get_financial_metrics_response_basic_ttm(mock_yf_ticker_class):
    """
    Tests `get_financial_metrics_response` for TTM (Trailing Twelve Months) metrics.
    Focuses on metrics derived directly from `ticker.info` and basic calculated TTM metrics
    like gross margin and FCF per share.
    """
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = MOCK_VALID_AAPL_INFO

    # Mocking the financial statement attributes directly on the ticker instance
    mock_ticker_instance.income_stmt = MOCK_AAPL_INCOME_ANNUAL
    mock_ticker_instance.quarterly_income_stmt = MOCK_AAPL_INCOME_QUARTERLY
    mock_ticker_instance.balance_sheet = MOCK_AAPL_BALANCE_SHEET_ANNUAL
    mock_ticker_instance.quarterly_balance_sheet = MOCK_AAPL_BALANCE_SHEET_QUARTERLY
    mock_ticker_instance.cashflow = MOCK_AAPL_CASHFLOW_ANNUAL
    mock_ticker_instance.quarterly_cashflow = MOCK_AAPL_CASHFLOW_QUARTERLY
    # Fallbacks for older yfinance versions potentially accessed by YFinanceDataFetcher
    mock_ticker_instance.financials = MOCK_AAPL_INCOME_ANNUAL
    mock_ticker_instance.quarterly_financials = MOCK_AAPL_INCOME_QUARTERLY

    mock_yf_ticker_class.return_value = mock_ticker_instance

    result = get_financial_metrics_response('AAPL', include_historical=False)

    assert result is not None
    assert isinstance(result, FinancialMetricsResponse)
    assert len(result.financial_metrics) == 1
    
    metrics = result.financial_metrics[0]
    assert isinstance(metrics, FinancialMetrics)
    assert metrics.ticker == 'AAPL'
    assert metrics.period == "ttm"
    
    assert metrics.market_cap == 2.0e12
    assert metrics.price_to_earnings_ratio == 25.0
    assert metrics.net_margin == 0.21 

    expected_revenue_ttm = MOCK_AAPL_INCOME_QUARTERLY.iloc[:, :4].loc['Total Revenue'].sum()
    expected_cogs_ttm = MOCK_AAPL_INCOME_QUARTERLY.iloc[:, :4].loc['Cost Of Revenue'].sum()
    expected_gross_margin = (expected_revenue_ttm - expected_cogs_ttm) / expected_revenue_ttm
    assert metrics.gross_margin == pytest.approx(expected_gross_margin)

    expected_opcf_ttm = MOCK_AAPL_CASHFLOW_QUARTERLY.iloc[:,:4].loc['Total Cash From Operating Activities'].sum()
    expected_capex_ttm = MOCK_AAPL_CASHFLOW_QUARTERLY.iloc[:,:4].loc['Capital Expenditures'].sum()
    expected_fcf_ttm = expected_opcf_ttm + expected_capex_ttm
    expected_fcf_per_share = expected_fcf_ttm / MOCK_VALID_AAPL_INFO['sharesOutstanding']
    assert metrics.free_cash_flow == pytest.approx(expected_fcf_ttm)
    assert metrics.free_cash_flow_per_share == pytest.approx(expected_fcf_per_share)
    assert metrics.free_cash_flow_yield == pytest.approx(expected_fcf_ttm / MOCK_VALID_AAPL_INFO['marketCap'])


@patch('src.data.yfinance_fetcher.yf.Ticker')
def test_get_company_facts_gap_fields(mock_yf_ticker_class):
    """
    Tests `get_company_facts_response` to ensure that fields identified as GAPs
    (not typically provided by yfinance, e.g., CIK, SIC codes) are correctly `None`.
    """
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {
        'shortName': 'Test Co', 'currency': 'USD', 'regularMarketPrice': 10.0,
        'industry': 'Testing', 'sector': 'Automation'
    }
    mock_yf_ticker_class.return_value = mock_ticker_instance

    result = get_company_facts_response('TESTGAP')

    assert result is not None
    assert isinstance(result, CompanyFactsResponse)
    facts = result.company_facts
    assert facts.ticker == 'TESTGAP'
    assert facts.name == 'Test Co'
    assert facts.industry == 'Testing'
    assert facts.cik is None 
    assert facts.sic_code is None
    assert facts.listing_date is None


@patch('src.data.yfinance_fetcher.yf.Ticker')
def test_get_financial_metrics_response_historical(mock_yf_ticker_class):
    """
    Tests `get_financial_metrics_response` with `include_historical=True`.
    Verifies that historical annual metrics are calculated and appended correctly,
    in addition to the TTM metrics.
    """
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = MOCK_VALID_AAPL_INFO

    mock_ticker_instance.income_stmt = MOCK_AAPL_INCOME_ANNUAL
    mock_ticker_instance.quarterly_income_stmt = MOCK_AAPL_INCOME_QUARTERLY
    mock_ticker_instance.balance_sheet = MOCK_AAPL_BALANCE_SHEET_ANNUAL
    mock_ticker_instance.quarterly_balance_sheet = pd.DataFrame() 
    mock_ticker_instance.cashflow = MOCK_AAPL_CASHFLOW_ANNUAL
    mock_ticker_instance.quarterly_cashflow = MOCK_AAPL_CASHFLOW_QUARTERLY
    mock_ticker_instance.financials = MOCK_AAPL_INCOME_ANNUAL
    mock_ticker_instance.quarterly_financials = MOCK_AAPL_INCOME_QUARTERLY

    mock_yf_ticker_class.return_value = mock_ticker_instance

    result = get_financial_metrics_response('AAPL', include_historical=True)

    assert result is not None
    assert len(result.financial_metrics) == 3 # 1 TTM + 2 Annual periods from mock data
    
    metrics_2023 = result.financial_metrics[1] # TTM is [0], 2023 is [1]
    assert metrics_2023.period == "annual"
    assert metrics_2023.report_period == "2023-12-31"
    rev_2023 = MOCK_AAPL_INCOME_ANNUAL[pd.Timestamp('2023-12-31')]['Total Revenue']
    cogs_2023 = MOCK_AAPL_INCOME_ANNUAL[pd.Timestamp('2023-12-31')]['Cost Of Revenue']
    assert metrics_2023.gross_margin == pytest.approx((rev_2023 - cogs_2023) / rev_2023)
    
    op_cf_2023 = MOCK_AAPL_CASHFLOW_ANNUAL[pd.Timestamp('2023-12-31')]['Total Cash From Operating Activities']
    capex_2023 = MOCK_AAPL_CASHFLOW_ANNUAL[pd.Timestamp('2023-12-31')]['Capital Expenditures']
    assert metrics_2023.free_cash_flow == pytest.approx(op_cf_2023 + capex_2023)

    metrics_2022 = result.financial_metrics[2] # 2022 is [2]
    assert metrics_2022.period == "annual"
    assert metrics_2022.report_period == "2022-12-31"
    rev_2022 = MOCK_AAPL_INCOME_ANNUAL[pd.Timestamp('2022-12-31')]['Total Revenue']
    cogs_2022 = MOCK_AAPL_INCOME_ANNUAL[pd.Timestamp('2022-12-31')]['Cost Of Revenue']
    assert metrics_2022.gross_margin == pytest.approx((rev_2022 - cogs_2022) / rev_2022)

    assert metrics_2023.price_to_earnings_ratio is None # Historical price-based ratios are not calculated
    assert metrics_2022.market_cap is None


@patch('src.data.yfinance_fetcher.yf.Ticker', side_effect=Exception("Simulated yf.Ticker init error"))
def test_fetcher_init_exception_propagates_to_transformer(mock_yf_ticker_class_ex):
    """
    Tests the behavior of transformer functions when the underlying `yf.Ticker`
    initialization itself raises an exception within `YFinanceDataFetcher`.
    Transformers should gracefully handle this by returning their defined "empty" or None states.
    """
    price_res = get_price_response('ANYTICKER')
    assert price_res is not None 
    assert price_res.ticker == 'ANYTICKER'
    assert len(price_res.prices) == 0

    metrics_res = get_financial_metrics_response('ANYTICKER')
    assert metrics_res is not None
    assert metrics_res.financial_metrics == []

    facts_res = get_company_facts_response('ANYTICKER')
    assert facts_res is not None
    assert facts_res.company_facts.ticker == 'ANYTICKER'
    assert facts_res.company_facts.name is None


@patch('src.data.yfinance_fetcher.yf.Ticker')
def test_fetcher_init_missing_validation_field(mock_yf_ticker_class):
    """
    Tests transformer behavior when `YFinanceDataFetcher` fails its internal validation
    (e.g., `ticker.info` is missing key fields and history is empty),
    resulting in `valid_ticker=False`.
    Transformers should return their defined "empty" or None states.
    """
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.info = {'shortName': 'Test Co'} # Missing 'regularMarketPrice'
    mock_ticker_instance.history.return_value = pd.DataFrame() # Empty history
    mock_yf_ticker_class.return_value = mock_ticker_instance
        
    price_res = get_price_response('MISSINGFIELD')
    assert price_res is not None
    assert price_res.ticker == 'MISSINGFIELD'
    assert len(price_res.prices) == 0

    metrics_res = get_financial_metrics_response('MISSINGFIELD')
    assert metrics_res is not None
    assert metrics_res.financial_metrics == []

    facts_res = get_company_facts_response('MISSINGFIELD')
    assert facts_res is not None
    assert facts_res.company_facts.ticker == 'MISSINGFIELD'
    assert facts_res.company_facts.name is None 

