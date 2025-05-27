from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.tools.api import get_prices, prices_to_df
import json
import logging # Added import


##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors for multiple tickers."""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    # Initialize risk analysis for each ticker
    risk_analysis = {}
    current_prices = {}  # Store prices here to avoid redundant API calls

    # First, fetch prices for all relevant tickers
    all_tickers = set(tickers) | set(portfolio.get("positions", {}).keys())
    
    for ticker in all_tickers:
        progress.update_status("risk_management_agent", ticker, "Fetching price data")
        
        # Logging before get_prices call
        logging.info(f"RiskManager: Calling get_prices for {ticker} with start_date={data['end_date']}, end_date={data['end_date']}")
        
        prices_list, fetch_error = get_prices(
            ticker=ticker,
            start_date=data["end_date"],  # Just get the latest price
            end_date=data["end_date"],
        )

        # Logging after get_prices call
        logging.info(f"RiskManager: Received for {ticker}: prices_list (count={len(prices_list)}), fetch_error='{fetch_error}'")

        error_for_ticker = None # To store specific error for this ticker for risk_analysis

        if fetch_error:
            error_for_ticker = fetch_error
            progress.update_status("risk_management_agent", ticker, f"Warning: Fetch error - {fetch_error}")
            # current_prices[ticker] will not be set, handled later
        elif not prices_list:
            error_for_ticker = "No price data found for the specific date after filtering."
            progress.update_status("risk_management_agent", ticker, f"Warning: {error_for_ticker}")
            # current_prices[ticker] will not be set
        else:
            prices_df = prices_to_df(prices_list)
            logging.info(f"RiskManager: For {ticker}, prices_df is empty: {prices_df.empty}. Head: {prices_df.head().to_string() if not prices_df.empty else 'N/A'}")
            
            if prices_df.empty:
                error_for_ticker = "Price data converted to empty DataFrame."
                progress.update_status("risk_management_agent", ticker, f"Warning: {error_for_ticker}")
                # current_prices[ticker] will not be set
            else:
                current_price = prices_df["close"].iloc[-1]
                current_prices[ticker] = current_price # Populate current_prices only on success
                progress.update_status("risk_management_agent", ticker, f"Current price: {current_price}")
        
        # If there was an error, store it in risk_analysis for this ticker, even if loop continues for others for portfolio value.
        # This part will be used in the next loop.
        if error_for_ticker and ticker in tickers: # Only initialize risk_analysis for main tickers here if error
             risk_analysis[ticker] = {
                "remaining_position_limit": 0.0,
                "current_price": 0.0,
                "reasoning": {"error": error_for_ticker}
            }


    # Calculate total portfolio value based on current market prices (Net Liquidation Value)
    # This loop uses current_prices, which is populated only with successfully fetched prices.
    total_portfolio_value = portfolio.get("cash", 0.0)
    
    for ticker, position in portfolio.get("positions", {}).items():
        if ticker in current_prices:
            # Add market value of long positions
            total_portfolio_value += position.get("long", 0) * current_prices[ticker]
            # Subtract market value of short positions
            total_portfolio_value -= position.get("short", 0) * current_prices[ticker]
    
    progress.update_status("risk_management_agent", None, f"Total portfolio value: {total_portfolio_value}")

    # Calculate risk limits for each ticker in the universe
    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Calculating position limits")
        
        if ticker not in current_prices:
            # If an error was already recorded for this ticker from the fetching loop, use that.
            # Otherwise, it's a generic missing price data error.
            if ticker not in risk_analysis: # Should only happen if ticker was not in all_tickers (unlikely) or error init failed
                logging.warning(f"RiskManager: Ticker {ticker} has no price and no prior error in risk_analysis. Setting generic error.")
                risk_analysis[ticker] = {
                    "remaining_position_limit": 0.0,
                    "current_price": 0.0,
                    "reasoning": {"error": "Missing price data for risk calculation (not found in current_prices)."}
                }
            # Update progress with the specific error if available, or a generic one.
            error_reason = risk_analysis[ticker]["reasoning"].get("error", "Failed: No price data available")
            progress.update_status("risk_management_agent", ticker, error_reason)
            logging.warning(f"RiskManager: Ticker {ticker} not in current_prices. Reason: {error_reason}")
            continue # Skip to next ticker as price is missing for calculations.
            
        current_price = current_prices[ticker] # This ticker is guaranteed to be in current_prices here
        
        # Calculate current market value of this position
        position = portfolio.get("positions", {}).get(ticker, {})
        long_value = position.get("long", 0) * current_price
        short_value = position.get("short", 0) * current_price
        current_position_value = abs(long_value - short_value)  # Use absolute exposure
        
        # Calculate position limit (20% of total portfolio)
        position_limit = total_portfolio_value * 0.20
        
        # Calculate remaining limit for this position
        remaining_position_limit = position_limit - current_position_value
        
        # Ensure we don't exceed available cash
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))
        
        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position_value": float(current_position_value),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
            },
        }
        
        progress.update_status("risk_management_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name="risk_management_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
