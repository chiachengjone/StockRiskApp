# Stock Risk Analysis App 

A professional-grade **Stock Risk Modeling & Portfolio Analysis** web application built with Streamlit, featuring advanced quantitative risk metrics, AI-powered predictions, and stress testing capabilities.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-4.3-brightgreen.svg)

## What's New in v4.3

### 🔮 Portfolio Digital Twin
- **Scenario Simulation** - Monte Carlo simulation of rebalancing strategies
- **Strategy Comparison** - Buy-and-hold vs Monthly/Quarterly/Threshold rebalancing
- **Tax-Loss Harvesting** - Simulate tax-optimized portfolio management
- **Portfolio Health Score** - Comprehensive 0-100 health assessment with recommendations
- **Correlation Monitoring** - Real-time alerts when normally uncorrelated assets converge

### 🧠 Behavioral Finance & Sentiment Analysis
- **NLP Sentiment Scoring** - TextBlob + VADER dual-analyzer with financial lexicon
- **Sentiment-Adjusted VaR** - VaR adjustment based on market sentiment (0.7x-1.5x multiplier)
- **News Sentiment Tracking** - Real-time sentiment analysis from financial news
- **Whale Activity Tracking** - Monitor large institutional transactions
- **Portfolio Sentiment Heatmap** - Visualize sentiment across holdings

### 🎯 Interactive What-If Analysis
- **Real-time Weight Adjustment** - Interactive sliders with immediate metric updates
- **Efficient Frontier Overlay** - See your portfolio position relative to optimal frontier
- **Trade Recommendations** - Automatic rebalancing trade list with cost estimates
- **Scenario Builder** - Save and compare multiple portfolio scenarios

### ⚡ Enhanced Real-time Data Pipeline
- **Polygon.io Integration** - Professional market data with WebSocket streaming
- **Alpaca Markets Integration** - Real-time quotes with paper/live trading support
- **WebSocket Price Streaming** - Sub-second price updates during market hours
- **Multi-Provider Fallback** - Automatic failover between data sources

### 📊 Enhanced Visualizations
- **3D Volatility Surfaces** - Interactive rotation, slicing, and ATM term structure
- **Volatility Smile Cross-Sections** - Time-horizon slices of IV surface
- **Fan Charts** - Probability distribution over simulation horizon
- **Correlation Change Charts** - Rolling average correlation with stress indicators

---

## What's New in v4.2

### TA Signals Extension (NEW)
Switch between **Risk Analysis** and **TA Signals** modes via the sidebar:

- **Technical Indicators** - SMA, EMA, RSI, MACD, Bollinger Bands, ADX, Stochastic, ATR
- **Signal Generation** - MA Crossover, RSI, MACD, Bollinger breakouts with 0-100 scoring
- **Risk-Filtered Signals** - Filter by volatility (<30%), beta (<1.8), Sharpe (>0.3)
- **Interactive Charts** - Candlestick with indicator overlays and signal markers
- **Stock Screener** - Scan multiple symbols for active signals
- **Backtesting** - Test signal performance with equity curves and trade history
- **Portfolio Signals** - Aggregate signals for portfolio holdings

## What's New in v4.1

### Enhanced Analytics
- **VaR Backtesting** - Kupiec and Christoffersen tests for model validation
- **Regime Detection** - GMM-based Bull/Bear/Sideways market identification
- **Ensemble VaR** - Combine XGBoost, GARCH, Historical, Parametric, and EWMA models
- **Confidence Intervals** - Bootstrap-based uncertainty quantification

### Advanced Portfolio Features
- **Risk Parity Optimization** - Equal risk contribution allocation
- **Hierarchical Risk Parity (HRP)** - Clustering-based portfolio construction
- **Black-Litterman Model** - Combine market equilibrium with investor views
- **Transaction Cost Analysis** - Estimate rebalancing costs
- **Threshold Rebalancing** - Drift-based rebalancing recommendations
- **Tax-Loss Harvesting** - Identify tax optimization opportunities

### Real-time Features
- **Live Market Quotes** - Real-time price updates
- **Market Hours Detection** - NYSE, NASDAQ, LSE schedule awareness
- **Live P&L Tracking** - Real-time profit/loss calculations

### Enhanced Visualizations
- **Interactive Correlation Heatmaps** - Clustered correlation matrices
- **VaR Cone Projections** - Monte Carlo confidence cones
- **3D Volatility Surfaces** - Options IV surface visualization
- **Downloadable Charts** - Export charts as HTML

---

## Features

### Single Stock Analysis
- **Value at Risk (VaR)** - Multiple methods: Parametric (Normal & t-distribution), Historical, Monte Carlo
- **Conditional VaR (CVaR)** - Expected Shortfall for tail risk
- **GARCH(1,1)** - Volatility forecasting with clustering effects
- **Extreme Value Theory (EVT)** - Tail risk modeling using Generalized Pareto Distribution
- **Risk Metrics** - Sharpe, Sortino, Calmar ratios, Max Drawdown, Beta, Alpha

### Portfolio Analysis
- **Multi-asset portfolio construction** with custom weights
- **Risk contribution analysis** - Marginal VaR by asset
- **Correlation matrices** with interactive heatmaps
- **Mean-Variance Optimization (MPT)** - Efficient frontier generation
- **Portfolio stress testing** across historical scenarios

### Advanced Analytics
- **Monte Carlo Simulation** - Up to 50,000 paths with confidence cones
- **Stress Testing** - Pre-configured scenarios (Black Monday, GFC 2008, COVID 2020, etc.)
- **Fama-French 5-Factor Model** - Factor regression with alpha extraction
- **Kelly Criterion** - Optimal position sizing for risk management
- **ESG Ratings** - Environmental, Social, Governance scores
- **XGBoost ML Predictor** - AI-powered next-day VaR prediction with feature importance

### Data Coverage
- **Stocks** - US equities, global markets
- **ETFs** - Major indices and sector funds
- **Cryptocurrencies** - BTC, ETH, SOL, ADA, XRP
- **Benchmarks** - S&P 500, NASDAQ, Dow Jones, VIX, international indices

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/chiachengjone/StockRiskAnalysis.git
cd StockRiskAnalysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run stock_risk_app.py
```

The app will open automatically in your default browser at `http://localhost:8501`

## Tech Stack

- **Frontend**: Streamlit (interactive web UI)
- **Data**: yfinance (Yahoo Finance API)
- **Visualization**: Plotly (interactive charts)
- **Risk Models**: 
  - `arch` - GARCH volatility modeling
  - `scipy.stats` - Statistical distributions (Normal, t, GPD)
  - Custom VaR/CVaR implementations
- **Machine Learning**: XGBoost (tree-based regression for VaR prediction)
- **Factor Models**: Fama-French 5-Factor (via pandas-datareader)
- **Optimization**: scipy.optimize (Mean-Variance portfolio optimization)

## Key Modules

### `stock_risk_app.py`
Main Streamlit application with UI components and workflow orchestration.

### `risk_engine.py`
Core risk calculations:
- VaR methods (Parametric, Historical, Monte Carlo)
- GARCH(1,1) volatility forecasting
- Extreme Value Theory (EVT)
- Portfolio optimization & efficient frontier
- Stress testing framework

### `ml_predictor.py`
Machine learning module:
- XGBoost regression for VaR prediction
- Feature engineering (RSI, volatility ratios, momentum, skewness)
- Model backtesting & validation
- Method comparison (ML vs traditional VaR)

### `factors.py`
Factor analysis & advanced metrics:
- Fama-French 5-Factor regression
- Kelly Criterion position sizing
- ESG rating integration
- Performance attribution

### `utils/` (NEW in v4.1)
Enhanced utilities package:

#### `utils/performance.py`
- Cached data fetching with TTL
- Lazy loading decorators
- Progress tracking utilities

#### `utils/validation.py`
- Robust ticker validation
- Retry logic with exponential backoff
- Data quality checks (outliers, missing data)
- Corporate action adjustments

#### `utils/analytics.py`
- VaR backtesting (Kupiec, Christoffersen tests)
- Regime detection with Gaussian Mixture Models
- Time-varying beta (rolling, EWMA, Kalman)
- CoVaR systemic risk measures
- Enhanced stress testing

#### `utils/portfolio.py`
- Risk parity optimization
- Hierarchical Risk Parity (HRP)
- Black-Litterman model
- Transaction cost modeling
- Threshold rebalancing
- Tax-loss harvesting

#### `utils/visualization.py`
- Interactive correlation heatmaps
- 3D volatility surfaces
- VaR cone projections
- Risk contribution charts
- Chart export utilities

#### `utils/realtime.py`
- Live quote fetching
- Market hours detection
- Real-time P&L tracking

### `services/` (NEW in v4.2)
TA Signals extension services:

#### `services/ta_service.py`
- Technical indicator calculations (SMA, EMA, RSI, MACD, etc.)
- Bollinger Bands, ADX, Stochastic, ATR
- Indicator summary and interpretation

#### `services/signals_service.py`
- Signal generation (MA crossover, RSI, MACD, Bollinger)
- Combined signal scoring (0-100)
- Risk-based signal filtering
- Backtesting engine with equity curve tracking

### `ta_signals_app.py` (NEW in v4.2)
Main TA Signals extension:
- Candlestick charts with indicators
- Signal dashboard and history
- Stock screener
- Backtest module
- Portfolio signal aggregation

## Use Cases

1. **Risk Management** - Quantify portfolio risk with institutional-grade metrics
2. **Position Sizing** - Kelly Criterion for optimal capital allocation
3. **Stress Testing** - Evaluate portfolio resilience under historical crash scenarios
4. **Factor Analysis** - Decompose returns into systematic factors + alpha
5. **ESG Screening** - Incorporate sustainability ratings into investment decisions
6. **Backtesting** - Compare strategy performance against benchmarks

## Screenshots

### Single Stock Analysis
- Real-time price charts with technical indicators
- VaR comparison across multiple methodologies
- Monte Carlo simulation with confidence intervals
- GARCH volatility forecasting

### Portfolio Mode
- Multi-asset correlation heatmaps
- Risk contribution pie charts
- Efficient frontier visualization
- Stress test impact analysis

## Methodology

### Value at Risk (VaR)
```
Parametric VaR = μ + σ × Z_α × √t
Historical VaR = Empirical α-quantile
Monte Carlo VaR = Simulated distribution quantile
```

### GARCH(1,1)
```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
```

### Fama-French 5-Factor
```
R_excess = α + β₁×Mkt-RF + β₂×SMB + β₃×HML + β₄×RMW + β₅×CMA + ε
```

### Kelly Criterion
```
f* = (p × b - q) / b
where p = win probability, q = loss probability, b = win/loss ratio
```

## Disclaimer

This application is for **educational and research purposes only**. It should not be considered financial advice. Always consult with a qualified financial advisor before making investment decisions.

- Past performance does not guarantee future results
- All models have limitations and assumptions
- Market conditions can change rapidly
- Use at your own risk

## License

MIT License - feel free to use and modify for your projects.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**Chia-Cheng (Jone)**
- GitHub: [@chiachengjone](https://github.com/chiachengjone)
- Repository: [StockRiskAnalysis](https://github.com/chiachengjone/StockRiskAnalysis)

## Acknowledgments

- Yahoo Finance for market data API
- Kenneth French Data Library for factor data
- Streamlit team for the amazing framework
- Open-source community for incredible libraries

---

**Built with ❤️ using Python & Streamlit**
