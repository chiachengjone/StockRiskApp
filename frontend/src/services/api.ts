/**
 * Complete API Service Layer
 * ==========================
 * All backend endpoints with TypeScript types
 */

import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// ============================================================================
// Risk API
// ============================================================================

export const riskApi = {
  getMetrics: (ticker: string, startDate: string, endDate: string) =>
    api.post('/risk/metrics', { ticker, start_date: startDate, end_date: endDate }),
  
  getVaR: (ticker: string, startDate: string, endDate: string, confidence = 0.95) =>
    api.post('/risk/var', { ticker, start_date: startDate, end_date: endDate, confidence_level: confidence }),
  
  getAllVaR: (ticker: string, startDate: string, endDate: string, confidence = 0.95) =>
    api.post('/risk/all-var', { ticker, start_date: startDate, end_date: endDate, confidence }),
  
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

// ============================================================================
// Factors API
// ============================================================================

export const factorsApi = {
  famaFrench: (ticker: string, startDate: string, endDate: string, benchmark = '^GSPC') =>
    api.post('/factors/fama-french', { ticker, start_date: startDate, end_date: endDate, benchmark }),
  
  kelly: (ticker: string, startDate: string, endDate: string, fraction = 0.5) =>
    api.post('/factors/kelly', { ticker, start_date: startDate, end_date: endDate, fraction }),
  
  esg: (ticker: string) =>
    api.get(`/factors/esg/${ticker}`),
  
  styleFactor: (ticker: string, startDate: string, endDate: string, benchmark = '^GSPC') =>
    api.post('/factors/style', { ticker, start_date: startDate, end_date: endDate, benchmark }),
  
  quickStyle: (ticker: string, days = 252, benchmark = '^GSPC') =>
    api.get(`/factors/style/${ticker}?days=${days}&benchmark=${benchmark}`),
}

// ============================================================================
// Options API
// ============================================================================

export const optionsApi = {
  blackScholes: (spotPrice: number, strikePrice: number, timeToExpiry: number, riskFreeRate: number, volatility: number, optionType: 'call' | 'put') =>
    api.post('/options/black-scholes', { spot_price: spotPrice, strike_price: strikePrice, time_to_expiry: timeToExpiry, risk_free_rate: riskFreeRate, volatility, option_type: optionType }),
  
  greeks: (spotPrice: number, strikePrice: number, timeToExpiry: number, riskFreeRate: number, volatility: number, optionType: 'call' | 'put') =>
    api.post('/options/greeks', { spot_price: spotPrice, strike_price: strikePrice, time_to_expiry: timeToExpiry, risk_free_rate: riskFreeRate, volatility, option_type: optionType }),
  
  impliedVolatility: (marketPrice: number, spotPrice: number, strikePrice: number, timeToExpiry: number, riskFreeRate: number, optionType: 'call' | 'put') =>
    api.post('/options/implied-volatility', { market_price: marketPrice, spot_price: spotPrice, strike_price: strikePrice, time_to_expiry: timeToExpiry, risk_free_rate: riskFreeRate, option_type: optionType }),
  
  analyze: (ticker: string, spotPrice: number, strikePrice: number, timeToExpiry: number, riskFreeRate: number, volatility: number, optionType: 'call' | 'put', positionSize = 100) =>
    api.post('/options/analyze', { ticker, spot_price: spotPrice, strike_price: strikePrice, time_to_expiry: timeToExpiry, risk_free_rate: riskFreeRate, volatility, option_type: optionType, position_size: positionSize }),
  
  volatilitySurface: (ticker: string, spotPrice: number, strikes: number[], expiries: number[]) =>
    api.post('/options/volatility-surface', { ticker, spot_price: spotPrice, strikes, expiries }),
  
  quick: (ticker: string, strikePct = 1.0, expiryDays = 30, optionType: 'call' | 'put' = 'call') =>
    api.get(`/options/quick/${ticker}?strike_pct=${strikePct}&expiry_days=${expiryDays}&option_type=${optionType}`),
}

// ============================================================================
// Fundamentals API
// ============================================================================

export const fundamentalsApi = {
  get: (ticker: string) =>
    api.get(`/fundamentals/${ticker}`),
  
  analyze: (ticker: string) =>
    api.post('/fundamentals/analyze', { ticker }),
  
  dcf: (ticker: string, growthRate = 0.05, terminalGrowth = 0.02, discountRate = 0.10, projectionYears = 5) =>
    api.post('/fundamentals/dcf', { ticker, growth_rate: growthRate, terminal_growth: terminalGrowth, discount_rate: discountRate, projection_years: projectionYears }),
  
  peerComparison: (ticker: string, peers?: string[]) =>
    api.post('/fundamentals/peer-comparison', { ticker, peers }),
  
  qualityScore: (ticker: string) =>
    api.get(`/fundamentals/quality-score/${ticker}`),
  
  quick: (ticker: string) =>
    api.get(`/fundamentals/quick/${ticker}`),
}

// ============================================================================
// Sentiment API
// ============================================================================

export const sentimentApi = {
  analyze: (ticker: string, startDate: string, endDate: string, benchmark = '^GSPC') =>
    api.post('/sentiment/analyze', { ticker, start_date: startDate, end_date: endDate, benchmark }),
  
  quickAnalyze: (ticker: string, days = 60, benchmark = '^GSPC') =>
    api.get(`/sentiment/analyze/${ticker}?days=${days}&benchmark=${benchmark}`),
  
  var: (ticker: string, startDate: string, endDate: string, confidenceLevel = 0.95) =>
    api.post('/sentiment/var', { ticker, start_date: startDate, end_date: endDate, confidence_level: confidenceLevel }),
  
  portfolioVar: (tickers: string[], weights: number[], startDate: string, endDate: string, confidenceLevel = 0.95) =>
    api.post('/sentiment/portfolio-var', { tickers, weights, start_date: startDate, end_date: endDate, confidence_level: confidenceLevel }),
  
  signals: (ticker: string, days = 60) =>
    api.get(`/sentiment/signals/${ticker}?days=${days}`),
}

// ============================================================================
// Digital Twin API
// ============================================================================

export const digitalTwinApi = {
  scenario: (tickers: string[], weights: number[], strategy: string, years = 1, initialValue = 100000) =>
    api.post('/digital-twin/scenario', { tickers, weights, strategy, years, initial_value: initialValue }),
  
  compareScenarios: (tickers: string[], weights: number[], strategies: string[], years = 1, initialValue = 100000) =>
    api.post('/digital-twin/compare-scenarios', { tickers, weights, strategies, years, initial_value: initialValue }),
  
  healthScore: (tickers: string[], weights: number[], days = 252) =>
    api.post('/digital-twin/health-score', { tickers, weights, days }),
  
  whatIf: (currentWeights: Record<string, number>, proposedWeights: Record<string, number>) =>
    api.post('/digital-twin/what-if', { current_weights: currentWeights, proposed_weights: proposedWeights }),
  
  rebalanceTrades: (tickers: string[], currentWeights: number[], targetWeights: number[], portfolioValue: number, minTradePct = 0.01) =>
    api.post('/digital-twin/rebalance-trades', { tickers, current_weights: currentWeights, target_weights: targetWeights, portfolio_value: portfolioValue, min_trade_pct: minTradePct }),
  
  optimizeTarget: (tickers: string[], currentWeights: number[], target = 'max_sharpe', targetValue?: number, constraints?: Record<string, unknown>) =>
    api.post('/digital-twin/optimize-target', { tickers, current_weights: currentWeights, target, target_value: targetValue, constraints }),
  
  stressTest: (scenario: string, tickers: string, weights: string) =>
    api.get(`/digital-twin/stress-test/${scenario}?tickers=${tickers}&weights=${weights}`),
}

// ============================================================================
// Reports API
// ============================================================================

export const reportsApi = {
  checkAvailability: () =>
    api.get('/reports/available'),
  
  singleStock: (ticker: string, startDate: string, endDate: string) =>
    api.post('/reports/single-stock', { ticker, start_date: startDate, end_date: endDate }, { responseType: 'blob' }),
  
  portfolio: (portfolioName: string, tickers: string[], weights: number[], startDate: string, endDate: string) =>
    api.post('/reports/portfolio', { portfolio_name: portfolioName, tickers, weights, start_date: startDate, end_date: endDate }, { responseType: 'blob' }),
  
  comparison: (tickers: string[], startDate: string, endDate: string) =>
    api.post('/reports/comparison', { tickers, start_date: startDate, end_date: endDate }, { responseType: 'blob' }),
  
  quick: (ticker: string, days = 252) =>
    api.get(`/reports/quick/${ticker}?days=${days}`, { responseType: 'blob' }),
}

export default api
