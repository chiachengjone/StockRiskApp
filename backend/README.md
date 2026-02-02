# Stock Risk Modelling App - Backend

FastAPI backend for stock risk analysis.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --port 8000
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

### Risk Analysis
- `POST /api/risk/metrics` - Calculate risk metrics
- `POST /api/risk/var` - Calculate VaR
- `POST /api/risk/garch` - GARCH(1,1) model
- `POST /api/risk/monte-carlo` - Monte Carlo simulation
- `POST /api/risk/stress-test` - Stress testing

### Machine Learning
- `POST /api/ml/predict` - ML VaR prediction
- `POST /api/ml/ensemble` - Ensemble prediction
- `POST /api/ml/backtest` - Backtest VaR

### Portfolio
- `POST /api/portfolio/risk-parity` - Risk parity weights
- `POST /api/portfolio/hrp` - Hierarchical Risk Parity
- `POST /api/portfolio/black-litterman` - Black-Litterman
- `POST /api/portfolio/efficient-frontier` - Efficient frontier

### Data
- `POST /api/data/historical` - Get historical data
- `GET /api/data/info/{ticker}` - Stock info

### Technical Analysis
- `POST /api/ta/full` - All indicators
- `GET /api/ta/signals/{ticker}` - Trading signals
