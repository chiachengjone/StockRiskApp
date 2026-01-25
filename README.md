# Stock Risk Analysis App 

A professional-grade **Stock Risk Modeling & Portfolio Analysis** web application built with Streamlit, featuring advanced quantitative risk metrics, AI-powered predictions, and stress testing capabilities.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

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
