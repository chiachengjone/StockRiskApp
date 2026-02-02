import axios from 'axios'

const API_BASE_URL = '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Risk API
export const riskApi = {
  getMetrics: (ticker: string, startDate: string, endDate: string) =>
    api.post('/risk/metrics', { ticker, start_date: startDate, end_date: endDate }),
  
  getVaR: (ticker: string, startDate: string, endDate: string, confidence = 0.95) =>
    api.post('/risk/var', { ticker, start_date: startDate, end_date: endDate, confidence }),
  
  getGARCH: (ticker: string, startDate: string, endDate: string, p = 1, q = 1) =>
    api.post('/risk/garch', { ticker, start_date: startDate, end_date: endDate, p, q }),
  
  getEVT: (ticker: string, startDate: string, endDate: string, threshold = 0.95) =>
    api.post('/risk/evt', { ticker, start_date: startDate, end_date: endDate, threshold_percentile: threshold }),
  
  getMonteCarlo: (ticker: string, startDate: string, endDate: string, nSims = 10000, horizon = 10) =>
    api.post('/risk/monte-carlo', { ticker, start_date: startDate, end_date: endDate, n_simulations: nSims, horizon_days: horizon }),
  
  getStressTest: (ticker: string, startDate: string, endDate: string, scenario?: string, customShock?: number) =>
    api.post('/risk/stress-test', { ticker, start_date: startDate, end_date: endDate, scenario, custom_shock: customShock }),
  
  getStressScenarios: () =>
    api.get('/risk/stress-scenarios'),
  
  getRollingMetrics: (ticker: string, startDate: string, endDate: string, window = 21) =>
    api.post('/risk/rolling', { ticker, start_date: startDate, end_date: endDate, window_days: window }),
}

// ML API
export const mlApi = {
  predict: (ticker: string, startDate: string, endDate: string) =>
    api.post('/ml/predict', { ticker, start_date: startDate, end_date: endDate }),
  
  ensemble: (ticker: string, startDate: string, endDate: string, models?: string[]) =>
    api.post('/ml/ensemble', { ticker, start_date: startDate, end_date: endDate, models }),
  
  backtest: (ticker: string, startDate: string, endDate: string, window = 252, confidence = 0.95) =>
    api.post('/ml/backtest', { ticker, start_date: startDate, end_date: endDate, window, confidence }),
  
  volatilityForecast: (ticker: string, startDate: string, endDate: string, horizon = 10) =>
    api.post('/ml/volatility-forecast', { ticker, start_date: startDate, end_date: endDate, horizon }),
  
  bootstrap: (ticker: string, startDate: string, endDate: string, nBootstrap = 100) =>
    api.post('/ml/bootstrap', { ticker, start_date: startDate, end_date: endDate, n_bootstrap: nBootstrap }),
  
  featureImportance: (ticker: string, startDate: string, endDate: string) =>
    api.post('/ml/feature-importance', { ticker, start_date: startDate, end_date: endDate }),
}

// Portfolio API
export const portfolioApi = {
  riskParity: (tickers: string[], startDate: string, endDate: string, targetRisk = 0.1) =>
    api.post('/portfolio/risk-parity', { tickers, start_date: startDate, end_date: endDate, target_risk: targetRisk }),
  
  hrp: (tickers: string[], startDate: string, endDate: string, method = 'complete') =>
    api.post('/portfolio/hrp', { tickers, start_date: startDate, end_date: endDate, method }),
  
  blackLitterman: (tickers: string[], views: Record<string, number>, startDate: string, endDate: string, riskFreeRate = 0.05) =>
    api.post('/portfolio/black-litterman', { tickers, views, start_date: startDate, end_date: endDate, risk_free_rate: riskFreeRate }),
  
  efficientFrontier: (tickers: string[], startDate: string, endDate: string, riskFreeRate = 0.05, nPoints = 50) =>
    api.post('/portfolio/efficient-frontier', { tickers, start_date: startDate, end_date: endDate, risk_free_rate: riskFreeRate, n_points: nPoints }),
  
  transactionCosts: (currentWeights: Record<string, number>, targetWeights: Record<string, number>, portfolioValue: number, prices: Record<string, number>) =>
    api.post('/portfolio/transaction-costs', { current_weights: currentWeights, target_weights: targetWeights, portfolio_value: portfolioValue, prices }),
  
  portfolioVaR: (tickers: string[], weights: Record<string, number>, startDate: string, endDate: string, confidence = 0.95) =>
    api.post('/portfolio/portfolio-var', { tickers, weights, start_date: startDate, end_date: endDate, confidence }),
  
  marginalVaR: (tickers: string[], weights: Record<string, number>, startDate: string, endDate: string, confidence = 0.95) =>
    api.post('/portfolio/marginal-var', { tickers, weights, start_date: startDate, end_date: endDate, confidence }),
  
  correlationAnalysis: (tickers: string[], startDate: string, endDate: string) =>
    api.post('/portfolio/correlation-analysis', { tickers, start_date: startDate, end_date: endDate }),
}

// Data API
export const dataApi = {
  getHistorical: (ticker: string, startDate: string, endDate: string) =>
    api.post('/data/historical', { ticker, start_date: startDate, end_date: endDate }),
  
  getInfo: (ticker: string) =>
    api.get(`/data/info/${ticker}`),
  
  getQuote: (ticker: string) =>
    api.get(`/data/quote/${ticker}`),
  
  getMultiple: (tickers: string[], startDate: string, endDate: string) =>
    api.post('/data/multiple', { tickers, start_date: startDate, end_date: endDate }),
}

// Technical Analysis API
export const taApi = {
  getFull: (ticker: string, startDate: string, endDate: string) =>
    api.post('/ta/full', { ticker, start_date: startDate, end_date: endDate }),
  
  getRSI: (ticker: string, days = 252, period = 14) =>
    api.get(`/ta/rsi/${ticker}?days=${days}&period=${period}`),
  
  getMACD: (ticker: string, days = 252) =>
    api.get(`/ta/macd/${ticker}?days=${days}`),
  
  getBollinger: (ticker: string, days = 252) =>
    api.get(`/ta/bollinger/${ticker}?days=${days}`),
  
  getSignals: (ticker: string, days = 252) =>
    api.get(`/ta/signals/${ticker}?days=${days}`),
}

export default api
