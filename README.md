# AI Hedge Fund (Free Edition) 🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![yfinance](https://img.shields.io/badge/data-yfinance-green.svg)](https://github.com/ranaroussi/yfinance)
[![Alpha Vantage](https://img.shields.io/badge/news-Alpha%20Vantage-orange.svg)](https://www.alphavantage.co/)

This is a proof of concept for an AI-powered hedge fund using **100% free APIs**.  The goal of this project is to explore the use of AI to make trading decisions.  This project is for **educational** purposes only and is not intended for real trading or investment.

## 🆓 What Makes This "Free Edition"?

| Feature | Status | Details |
|---------|--------|---------|
| 📊 **Stock Prices** | ✅ FREE | Unlimited via yfinance |
| 📈 **Financial Statements** | ✅ FREE | Quarterly data via yfinance |
| 💼 **Insider Trades** | ✅ FREE | Real-time via yfinance |
| 📰 **News Sentiment** | ✅ FREE | 500/day via Alpha Vantage (optional) |
| 🔓 **Ticker Restrictions** | ✅ NONE | Any stock, any market |
| 💰 **API Costs** | ✅ $0 | Completely free for financial data |

### Key Highlights
- 🌍 **Global Coverage**: Analyze stocks from any market (US, international, crypto)
- 🚀 **No Limitations**: Not restricted to only AAPL, GOOGL, MSFT, NVDA, TSLA
- 💡 **Smart Sentiment**: Pre-computed sentiment scores with Alpha Vantage (saves 80% of LLM costs)
- 🛠️ **Easy Setup**: Only need one LLM API key to get started
- 📦 **No Dependencies**: No paid financial data subscriptions required

This system employs several agents working together:

1. Aswath Damodaran Agent - The Dean of Valuation, focuses on story, numbers, and disciplined valuation
2. Ben Graham Agent - The godfather of value investing, only buys hidden gems with a margin of safety
3. Bill Ackman Agent - An activist investor, takes bold positions and pushes for change
4. Cathie Wood Agent - The queen of growth investing, believes in the power of innovation and disruption
5. Charlie Munger Agent - Warren Buffett's partner, only buys wonderful businesses at fair prices
6. Michael Burry Agent - The Big Short contrarian who hunts for deep value
7. Mohnish Pabrai Agent - The Dhandho investor, who looks for doubles at low risk
8. Peter Lynch Agent - Practical investor who seeks "ten-baggers" in everyday businesses
9. Phil Fisher Agent - Meticulous growth investor who uses deep "scuttlebutt" research 
10. Rakesh Jhunjhunwala Agent - The Big Bull of India
11. Stanley Druckenmiller Agent - Macro legend who hunts for asymmetric opportunities with growth potential
12. Warren Buffett Agent - The oracle of Omaha, seeks wonderful companies at a fair price
13. Valuation Agent - Calculates the intrinsic value of a stock and generates trading signals
14. Sentiment Agent - Analyzes market sentiment and generates trading signals
15. Fundamentals Agent - Analyzes fundamental data and generates trading signals
16. Technicals Agent - Analyzes technical indicators and generates trading signals
17. Risk Manager - Calculates risk metrics and sets position limits
18. Portfolio Manager - Makes final trading decisions and generates orders

<img width="1042" alt="Screenshot 2025-03-22 at 6 19 07 PM" src="https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4" />

Note: the system does not actually make any trades.

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions
- Past performance does not indicate future results

By using this software, you agree to use it solely for learning purposes.

## Table of Contents
- [Quick Start Example](#-quick-start-example)
- [How to Install](#how-to-install)
- [How to Run](#how-to-run)
  - [⌨️ Command Line Interface](#️-command-line-interface)
  - [🖥️ Web Application](#️-web-application)
- [What's Different](#-whats-different-from-the-original)
- [Troubleshooting](#-troubleshooting)
- [How to Contribute](#how-to-contribute)
- [Credits](#-credits--acknowledgments)
- [License](#license)

## ⚡ Quick Start Example

Get started in 3 simple steps:

```bash
# 1. Clone and install
git clone https://github.com/uditrpanchal/ai-hedge-fund-yfinance-free.git
cd ai-hedge-fund-yfinance-free
poetry install

# 2. Set up your LLM API key (required)
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (or other LLM key)

# 3. Run analysis on any stock
poetry run python src/main.py --ticker NVDA
```

**That's it!** No financial data API key required. 🎉

**Want better news sentiment?** (Optional but recommended)
```bash
# Add Alpha Vantage key to .env for pre-computed sentiment scores
ALPHA_VANTAGE_API_KEY=your-key-here
```

## How to Install

Before you can run the AI Hedge Fund, you'll need to install it and set up your API keys. These steps are common to both the full-stack web application and command line interface.

### 1. Clone the Repository

```bash
git clone https://github.com/uditrpanchal/ai-hedge-fund-yfinance-free.git
cd ai-hedge-fund-yfinance-free
```

Or if you're cloning from the original repository:
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

### 2. Set up API keys

Create a `.env` file for your API keys:
```bash
# Create .env file for your API keys (in the root directory)
cp .env.example .env
```

Open and edit the `.env` file to add your LLM API key:
```bash
# Required: At least one LLM API key
OPENAI_API_KEY=your-openai-api-key
# OR
ANTHROPIC_API_KEY=your-anthropic-api-key
# OR
DEEPSEEK_API_KEY=your-deepseek-api-key
# OR
GROQ_API_KEY=your-groq-api-key

# Optional: For enhanced news sentiment analysis (recommended)
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-api-key
```

#### Required: LLM API Key

You **must** set at least one LLM API key for the hedge fund to work:
- **OpenAI** (GPT-4, GPT-4o, etc.): Get key from https://platform.openai.com/
- **Anthropic** (Claude): Get key from https://anthropic.com/
- **DeepSeek** (DeepSeek-R1): Get key from https://deepseek.com/
- **Groq** (Fast inference): Get key from https://groq.com/

#### Optional: News Sentiment API Key

For high-quality news sentiment analysis with pre-computed scores:

**Alpha Vantage** (Recommended):
- **Get it here**: https://www.alphavantage.co/support/#api-key
- **Free tier**: 500 API calls per day
- **Benefits**: 
  - Pre-computed sentiment scores (saves LLM calls)
  - Higher reliability than yfinance news
  - Reduces costs by ~80% for sentiment analysis
- **Fallback**: If not configured, system uses yfinance news API (may have limited availability)

#### Financial Data (No API Key Required!)

All stock prices, financial statements, metrics, and insider trades are fetched using the free **yfinance** library:
- ✅ **No API key needed**
- ✅ **No ticker restrictions** (analyze any stock)
- ✅ **Unlimited usage**
- ✅ **Completely free**

## How to Run

### ⌨️ Command Line Interface

You can run the AI Hedge Fund directly via terminal. This approach offers more granular control and is useful for automation, scripting, and integration purposes.

<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

#### Quick Start

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

#### Run the AI Hedge Fund

**Analyze any stock** (no restrictions):
```bash
# Single ticker
poetry run python src/main.py --ticker NVDA

# Multiple tickers
poetry run python src/main.py --ticker AAPL,MSFT,NVDA,TSLA,GOOGL

# International stocks
poetry run python src/main.py --ticker TSM,BABA,NVO
```

**Use local LLMs** (Ollama):
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama
```

**Analyze specific time period**:
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

**Example Output with Alpha Vantage:**
```
Fetching news for NVDA from Alpha Vantage...
✓ Retrieved 50 articles from Alpha Vantage with sentiment scores
✓ Aswath Damodaran    [NVDA] Done
✓ Fundamentals Analyst[NVDA] Done
✓ News Sentiment      [NVDA] Done (0 LLM calls - sentiments pre-computed)
✓ Technical Analyst   [NVDA] Done
✓ Valuation Analyst   [NVDA] Done
✓ Portfolio Manager   [NVDA] Done

TRADING DECISION: [NVDA]
Action: SHORT | Quantity: 99 | Confidence: 80%
Reasoning: Strong bearish valuation signals outweigh bullish sentiment
```

#### Run the Backtester
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

**Example Output:**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />


Note: The `--ollama`, `--start-date`, and `--end-date` flags work for the backtester, as well!

### 🖥️ Web Application

The new way to run the AI Hedge Fund is through our web application that provides a user-friendly interface. This is recommended for users who prefer visual interfaces over command line tools.

Please see detailed instructions on how to install and run the web application [here](https://github.com/virattt/ai-hedge-fund/tree/main/app).

<img width="1721" alt="Screenshot 2025-06-28 at 6 41 03 PM" src="https://github.com/user-attachments/assets/b95ab696-c9f4-416c-9ad1-51feb1f5374b" />


## 🔄 What's Different from the Original?

This fork replaces paid APIs with free alternatives while maintaining full functionality:

### Financial Data Migration
| **Feature** | **Original** | **This Fork** |
|------------|-------------|---------------|
| Price Data | Financial Datasets API | yfinance (free) |
| Financial Statements | Financial Datasets API | yfinance (free) |
| Insider Trades | Financial Datasets API | yfinance (free) |
| Company Metrics | Financial Datasets API | yfinance (free) |
| News Sentiment | Financial Datasets API | Alpha Vantage (free) + yfinance fallback |
| **API Key Required** | ✅ Yes (paid after free tier) | ❌ No (optional for news) |
| **Ticker Restrictions** | Only 5 free tickers | ✅ All tickers unlimited |

### Benefits of This Fork
- 💰 **Lower Costs**: No financial data API subscription needed
- 🌍 **Global Tickers**: Analyze any stock from any market
- 🚀 **Faster Setup**: One less API key to configure
- 📊 **Same Quality**: Equivalent data from trusted sources
- 🔄 **Better News**: Alpha Vantage provides pre-computed sentiment scores

## 🐛 Troubleshooting

### Common Issues

**Problem**: "No news articles found" or yfinance HTTP 401 errors
- **Solution**: Add `ALPHA_VANTAGE_API_KEY` to your `.env` file for reliable news sentiment
- **Why**: yfinance news API can be unreliable due to Yahoo Finance authentication changes

**Problem**: "Module not found" errors
- **Solution**: Run `poetry install` to install all dependencies including yfinance

**Problem**: Financial metrics showing `None` values
- **Solution**: Some tickers may have incomplete data in yfinance. Try major US stocks (AAPL, MSFT, GOOGL) first

**Problem**: Slow performance
- **Solution**: Alpha Vantage significantly reduces LLM calls for sentiment analysis. Add the API key for better performance.

## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused.  This will make it easier to review and merge.

## Feature Requests

If you have a feature request, please open an issue and make sure it is tagged with `enhancement`.

## 🙏 Credits & Acknowledgments

This fork is based on the original **AI Hedge Fund** by [@virattt](https://github.com/virattt):
- **Original Repository**: https://github.com/virattt/ai-hedge-fund
- **Twitter**: [@virattt](https://twitter.com/virattt)

### What This Fork Adds
- Migration to free yfinance for financial data
- Alpha Vantage integration for news sentiment
- Improved error handling and fallback mechanisms
- Support for unlimited stock tickers
- Enhanced documentation for free API setup

**Thank you** to the original creator for building this amazing educational project! 🚀

## 📝 Technical Details

### Data Sources
- **Stock Prices**: [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance wrapper
- **Financial Statements**: yfinance quarterly data
- **News Sentiment**: [Alpha Vantage NEWS_SENTIMENT API](https://www.alphavantage.co/documentation/#news-sentiment)
- **Insider Trades**: yfinance insider transactions data

### Dependencies Added
- `yfinance ^0.2.40` - Financial data fetching
- `requests` - HTTP requests for Alpha Vantage API

## License

This project is licensed under the MIT License - see the LICENSE file for details.
