# Stock Risk Modelling App v1.0

A modern, decoupled stock risk analysis application built with **FastAPI** (backend) and **React** (frontend).

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│                    React + Vite + Tailwind                   │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│    │Dashboard│  │ Single  │  │Portfolio│  │   TA    │      │
│    │         │  │ Stock   │  │         │  │         │      │
│    └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │
└─────────┼────────────┼────────────┼────────────┼────────────┘
          │            │            │            │
          └────────────┴─────┬──────┴────────────┘
                             │
                    ┌────────▼────────┐
                    │   REST API      │
                    │   (localhost)   │
                    └────────┬────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                         Backend                              │
│                    FastAPI + Pydantic                        │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│    │  Risk   │  │   ML    │  │Portfolio│  │   TA    │      │
│    │ Service │  │ Service │  │ Service │  │ Service │      │
│    └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
│                     ┌─────────────┐                         │
│                     │Data Service │                         │
│                     │  (yfinance) │                         │
│                     └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. OpenAPI docs at `http://localhost:8000/docs`.

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The app will be available at `http://localhost:5173`.

## 📊 Features

### Single Stock Analysis
- **Risk Metrics**: Annualized return, volatility, Sharpe ratio, Sortino ratio, max drawdown
- **VaR Analysis**: Parametric VaR, Historical VaR, CVaR (Expected Shortfall)
- **GARCH Modeling**: GARCH(1,1) volatility forecasting with conditional VaR
- **Monte Carlo**: 10,000 simulations with percentile distribution
- **Stress Testing**: Pre-built scenarios (2008 Crisis, COVID Crash, etc.)
- **ML Predictions**: XGBoost-based VaR prediction with feature importance

### Portfolio Analysis
- **Risk Parity**: Equal risk contribution portfolio
- **HRP (Hierarchical Risk Parity)**: Cluster-based allocation
- **Black-Litterman**: Incorporate investor views
- **Efficient Frontier**: Mean-variance optimization
- **Correlation Analysis**: Heatmap with high correlation detection
- **Portfolio VaR**: Component and marginal VaR

### Technical Analysis
- **Indicators**: RSI, MACD, Bollinger Bands, ADX, Stochastic, ATR
- **Moving Averages**: SMA, EMA with multiple periods
- **Signals**: Buy/Sell signals from each indicator
- **Charts**: Candlestick with overlays

## 🔌 API Endpoints

### Risk Endpoints (`/api/risk`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/metrics` | Calculate risk metrics |
| POST | `/var` | Calculate VaR (parametric, historical, CVaR) |
| POST | `/garch` | Fit GARCH(1,1) model |
| POST | `/evt` | Extreme Value Theory tail risk |
| POST | `/monte-carlo` | Monte Carlo simulation |
| POST | `/stress-test` | Run stress test scenario |
| GET | `/stress-scenarios` | List available scenarios |
| POST | `/rolling` | Rolling risk metrics |

### ML Endpoints (`/api/ml`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | ML-based VaR prediction |
| POST | `/ensemble` | Ensemble prediction |
| POST | `/backtest` | Backtest VaR model |
| POST | `/volatility-forecast` | Volatility forecast |
| POST | `/bootstrap` | Bootstrap confidence intervals |
| POST | `/feature-importance` | Get feature importance |

### Portfolio Endpoints (`/api/portfolio`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/risk-parity` | Risk parity optimization |
| POST | `/hrp` | Hierarchical Risk Parity |
| POST | `/black-litterman` | Black-Litterman optimization |
| POST | `/efficient-frontier` | Efficient frontier calculation |
| POST | `/transaction-costs` | Transaction cost analysis |
| POST | `/portfolio-var` | Portfolio VaR |
| POST | `/marginal-var` | Marginal VaR per asset |
| POST | `/correlation-analysis` | Correlation matrix |

### Data Endpoints (`/api/data`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/historical` | Get historical prices |
| GET | `/info/{ticker}` | Get stock info |
| GET | `/quote/{ticker}` | Get current quote |
| POST | `/multiple` | Get data for multiple tickers |

### TA Endpoints (`/api/ta`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/full` | All technical indicators |
| GET | `/rsi/{ticker}` | RSI indicator |
| GET | `/macd/{ticker}` | MACD indicator |
| GET | `/bollinger/{ticker}` | Bollinger Bands |
| GET | `/signals/{ticker}` | Buy/Sell signals |

## 🛠 Technology Stack

### Backend
- **FastAPI** - High-performance async web framework
- **Pydantic** - Data validation and settings management
- **yfinance** - Yahoo Finance data source
- **pandas/numpy** - Data manipulation
- **scipy** - Statistical functions
- **scikit-learn** - Machine learning
- **xgboost** - Gradient boosting
- **arch** - GARCH modeling

### Frontend
- **React 18** - UI library
- **Vite** - Fast build tool
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first CSS
- **React Query** - Data fetching & caching
- **React Router** - Client-side routing
- **ApexCharts** - Interactive charts
- **Recharts** - Additional chart components
- **Headless UI** - Accessible components

## 📁 Project Structure

```
StockRiskApp/
├── backend/
│   ├── app/
│   │   ├── api/           # API routes
│   │   │   ├── risk.py
│   │   │   ├── ml.py
│   │   │   ├── portfolio.py
│   │   │   ├── data.py
│   │   │   └── ta.py
│   │   ├── core/          # Configuration
│   │   │   └── config.py
│   │   ├── schemas/       # Pydantic models
│   │   │   ├── risk.py
│   │   │   ├── ml.py
│   │   │   └── portfolio.py
│   │   ├── services/      # Business logic
│   │   │   ├── risk_service.py
│   │   │   ├── ml_service.py
│   │   │   ├── portfolio_service.py
│   │   │   ├── ta_service.py
│   │   │   └── data_service.py
│   │   └── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/    # Reusable components
│   │   │   ├── Layout.tsx
│   │   │   ├── MetricCard.tsx
│   │   │   ├── LoadingSpinner.tsx
│   │   │   └── charts/    # Chart components
│   │   │       ├── PriceChart.tsx
│   │   │       ├── VaRChart.tsx
│   │   │       ├── MonteCarloChart.tsx
│   │   │       ├── PieChart.tsx
│   │   │       ├── EfficientFrontierChart.tsx
│   │   │       ├── CorrelationHeatmap.tsx
│   │   │       ├── CandlestickChart.tsx
│   │   │       └── TAIndicatorChart.tsx
│   │   ├── pages/         # Page components
│   │   │   ├── Dashboard.tsx
│   │   │   ├── SingleStock.tsx
│   │   │   ├── Portfolio.tsx
│   │   │   └── TechnicalAnalysis.tsx
│   │   ├── services/      # API client
│   │   │   └── api.ts
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── tsconfig.json
│
└── README.md
```

## 🔧 Configuration

### Environment Variables

Backend (`.env` in `backend/` folder):
```env
ENVIRONMENT=development
DEBUG=true
```

Frontend (`.env` in `frontend/` folder):
```env
VITE_API_URL=http://localhost:8000
```

## 📈 Mathematical Models

### Value at Risk (VaR)
- **Parametric**: $VaR_\alpha = \mu + z_\alpha \sigma$
- **Historical**: Percentile of historical returns
- **CVaR**: $E[X | X < VaR_\alpha]$

### GARCH(1,1)
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

### Risk Parity
Minimize: $\sum_{i,j} (w_i \cdot (\Sigma w)_i - w_j \cdot (\Sigma w)_j)^2$

### Black-Litterman
$$E[R] = [(\tau\Sigma)^{-1} + P'\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\Pi + P'\Omega^{-1}Q]$$

## 📝 License

MIT License

## 🙏 Acknowledgments

- Original Streamlit app architecture
- yfinance for market data
- FastAPI for the excellent web framework
