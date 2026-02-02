/**
 * React Query Hooks
 * =================
 * Custom hooks for server state management
 */

import { useQuery, useMutation, UseQueryOptions } from '@tanstack/react-query'
import { 
  riskApi, 
  mlApi, 
  portfolioApi, 
  dataApi, 
  taApi,
  factorsApi,
  optionsApi,
  fundamentalsApi,
  sentimentApi,
  digitalTwinApi,
  reportsApi
} from './api'

// ============================================================================
// Risk Hooks
// ============================================================================

export const useRiskMetrics = (ticker: string, startDate: string, endDate: string, options?: UseQueryOptions) =>
  useQuery({
    queryKey: ['risk', 'metrics', ticker, startDate, endDate],
    queryFn: () => riskApi.getMetrics(ticker, startDate, endDate).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 5 * 60 * 1000,
    ...options
  })

export const useVaR = (ticker: string, startDate: string, endDate: string, confidence = 0.95) =>
  useQuery({
    queryKey: ['risk', 'var', ticker, startDate, endDate, confidence],
    queryFn: () => riskApi.getVaR(ticker, startDate, endDate, confidence).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 5 * 60 * 1000,
  })

export const useGARCH = (ticker: string, startDate: string, endDate: string, p = 1, q = 1) =>
  useQuery({
    queryKey: ['risk', 'garch', ticker, startDate, endDate, p, q],
    queryFn: () => riskApi.getGARCH(ticker, startDate, endDate, p, q).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 5 * 60 * 1000,
  })

export const useEVT = (ticker: string, startDate: string, endDate: string, threshold = 0.95) =>
  useQuery({
    queryKey: ['risk', 'evt', ticker, startDate, endDate, threshold],
    queryFn: () => riskApi.getEVT(ticker, startDate, endDate, threshold).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 5 * 60 * 1000,
  })

export const useMonteCarlo = (ticker: string, startDate: string, endDate: string, nSims = 10000, horizon = 10) =>
  useQuery({
    queryKey: ['risk', 'monte-carlo', ticker, startDate, endDate, nSims, horizon],
    queryFn: () => riskApi.getMonteCarlo(ticker, startDate, endDate, nSims, horizon).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 5 * 60 * 1000,
  })

export const useStressTest = (ticker: string, startDate: string, endDate: string, scenario?: string) =>
  useQuery({
    queryKey: ['risk', 'stress-test', ticker, startDate, endDate, scenario],
    queryFn: () => riskApi.getStressTest(ticker, startDate, endDate, scenario).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 5 * 60 * 1000,
  })

export const useStressScenarios = () =>
  useQuery({
    queryKey: ['risk', 'stress-scenarios'],
    queryFn: () => riskApi.getStressScenarios().then(res => res.data),
    staleTime: 60 * 60 * 1000,
  })

// ============================================================================
// ML Hooks
// ============================================================================

export const useMLPredict = (ticker: string, startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['ml', 'predict', ticker, startDate, endDate],
    queryFn: () => mlApi.predict(ticker, startDate, endDate).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

export const useMLEnsemble = (ticker: string, startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['ml', 'ensemble', ticker, startDate, endDate],
    queryFn: () => mlApi.ensemble(ticker, startDate, endDate).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

export const useVolatilityForecast = (ticker: string, startDate: string, endDate: string, horizon = 10) =>
  useQuery({
    queryKey: ['ml', 'volatility-forecast', ticker, startDate, endDate, horizon],
    queryFn: () => mlApi.volatilityForecast(ticker, startDate, endDate, horizon).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

export const useFeatureImportance = (ticker: string, startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['ml', 'feature-importance', ticker, startDate, endDate],
    queryFn: () => mlApi.featureImportance(ticker, startDate, endDate).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

// ============================================================================
// Portfolio Hooks
// ============================================================================

export const useRiskParity = (tickers: string[], startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['portfolio', 'risk-parity', tickers, startDate, endDate],
    queryFn: () => portfolioApi.riskParity(tickers, startDate, endDate).then(res => res.data),
    enabled: tickers.length > 0 && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

export const useHRP = (tickers: string[], startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['portfolio', 'hrp', tickers, startDate, endDate],
    queryFn: () => portfolioApi.hrp(tickers, startDate, endDate).then(res => res.data),
    enabled: tickers.length > 0 && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

export const useEfficientFrontier = (tickers: string[], startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['portfolio', 'efficient-frontier', tickers, startDate, endDate],
    queryFn: () => portfolioApi.efficientFrontier(tickers, startDate, endDate).then(res => res.data),
    enabled: tickers.length > 0 && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

export const useCorrelationAnalysis = (tickers: string[], startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['portfolio', 'correlation', tickers, startDate, endDate],
    queryFn: () => portfolioApi.correlationAnalysis(tickers, startDate, endDate).then(res => res.data),
    enabled: tickers.length > 0 && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

// ============================================================================
// Data Hooks
// ============================================================================

export const useHistoricalData = (ticker: string, startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['data', 'historical', ticker, startDate, endDate],
    queryFn: () => dataApi.getHistorical(ticker, startDate, endDate).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 5 * 60 * 1000,
  })

export const useStockInfo = (ticker: string) =>
  useQuery({
    queryKey: ['data', 'info', ticker],
    queryFn: () => dataApi.getInfo(ticker).then(res => res.data),
    enabled: !!ticker,
    staleTime: 60 * 60 * 1000,
  })

export const useQuote = (ticker: string) =>
  useQuery({
    queryKey: ['data', 'quote', ticker],
    queryFn: () => dataApi.getQuote(ticker).then(res => res.data),
    enabled: !!ticker,
    staleTime: 60 * 1000,
    refetchInterval: 60 * 1000,
  })

// ============================================================================
// Technical Analysis Hooks
// ============================================================================

export const useTAFull = (ticker: string, startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['ta', 'full', ticker, startDate, endDate],
    queryFn: () => taApi.getFull(ticker, startDate, endDate).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 5 * 60 * 1000,
  })

export const useTASignals = (ticker: string, days = 252) =>
  useQuery({
    queryKey: ['ta', 'signals', ticker, days],
    queryFn: () => taApi.getSignals(ticker, days).then(res => res.data),
    enabled: !!ticker,
    staleTime: 5 * 60 * 1000,
  })

// ============================================================================
// Factors Hooks
// ============================================================================

export const useFamaFrench = (ticker: string, startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['factors', 'fama-french', ticker, startDate, endDate],
    queryFn: () => factorsApi.famaFrench(ticker, startDate, endDate).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

export const useKelly = (ticker: string, startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['factors', 'kelly', ticker, startDate, endDate],
    queryFn: () => factorsApi.kelly(ticker, startDate, endDate).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

export const useESG = (ticker: string) =>
  useQuery({
    queryKey: ['factors', 'esg', ticker],
    queryFn: () => factorsApi.esg(ticker).then(res => res.data),
    enabled: !!ticker,
    staleTime: 60 * 60 * 1000,
  })

export const useStyleFactor = (ticker: string, startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['factors', 'style', ticker, startDate, endDate],
    queryFn: () => factorsApi.styleFactor(ticker, startDate, endDate).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

// ============================================================================
// Options Hooks
// ============================================================================

export const useOptionQuick = (ticker: string, strikePct = 1.0, expiryDays = 30, optionType: 'call' | 'put' = 'call') =>
  useQuery({
    queryKey: ['options', 'quick', ticker, strikePct, expiryDays, optionType],
    queryFn: () => optionsApi.quick(ticker, strikePct, expiryDays, optionType).then(res => res.data),
    enabled: !!ticker,
    staleTime: 5 * 60 * 1000,
  })

export const useBlackScholesMutation = () =>
  useMutation({
    mutationFn: (params: { spotPrice: number; strikePrice: number; timeToExpiry: number; riskFreeRate: number; volatility: number; optionType: 'call' | 'put' }) =>
      optionsApi.blackScholes(params.spotPrice, params.strikePrice, params.timeToExpiry, params.riskFreeRate, params.volatility, params.optionType).then(res => res.data)
  })

export const useGreeksMutation = () =>
  useMutation({
    mutationFn: (params: { spotPrice: number; strikePrice: number; timeToExpiry: number; riskFreeRate: number; volatility: number; optionType: 'call' | 'put' }) =>
      optionsApi.greeks(params.spotPrice, params.strikePrice, params.timeToExpiry, params.riskFreeRate, params.volatility, params.optionType).then(res => res.data)
  })

// ============================================================================
// Fundamentals Hooks
// ============================================================================

export const useFundamentals = (ticker: string) =>
  useQuery({
    queryKey: ['fundamentals', ticker],
    queryFn: () => fundamentalsApi.get(ticker).then(res => res.data),
    enabled: !!ticker,
    staleTime: 60 * 60 * 1000,
  })

export const useDCF = (ticker: string) =>
  useQuery({
    queryKey: ['fundamentals', 'dcf', ticker],
    queryFn: () => fundamentalsApi.dcf(ticker).then(res => res.data),
    enabled: !!ticker,
    staleTime: 60 * 60 * 1000,
  })

export const usePeerComparison = (ticker: string) =>
  useQuery({
    queryKey: ['fundamentals', 'peer-comparison', ticker],
    queryFn: () => fundamentalsApi.peerComparison(ticker).then(res => res.data),
    enabled: !!ticker,
    staleTime: 60 * 60 * 1000,
  })

export const useQualityScore = (ticker: string) =>
  useQuery({
    queryKey: ['fundamentals', 'quality-score', ticker],
    queryFn: () => fundamentalsApi.qualityScore(ticker).then(res => res.data),
    enabled: !!ticker,
    staleTime: 60 * 60 * 1000,
  })

// ============================================================================
// Sentiment Hooks
// ============================================================================

export const useSentiment = (ticker: string, startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['sentiment', 'analyze', ticker, startDate, endDate],
    queryFn: () => sentimentApi.analyze(ticker, startDate, endDate).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

export const useSentimentVaR = (ticker: string, startDate: string, endDate: string) =>
  useQuery({
    queryKey: ['sentiment', 'var', ticker, startDate, endDate],
    queryFn: () => sentimentApi.var(ticker, startDate, endDate).then(res => res.data),
    enabled: !!ticker && !!startDate && !!endDate,
    staleTime: 10 * 60 * 1000,
  })

export const useMomentumSignals = (ticker: string) =>
  useQuery({
    queryKey: ['sentiment', 'signals', ticker],
    queryFn: () => sentimentApi.signals(ticker).then(res => res.data),
    enabled: !!ticker,
    staleTime: 5 * 60 * 1000,
  })

// ============================================================================
// Digital Twin Hooks
// ============================================================================

export const useDigitalTwinScenario = (tickers: string[], weights: number[], strategy: string, years = 1) =>
  useQuery({
    queryKey: ['digital-twin', 'scenario', tickers, weights, strategy, years],
    queryFn: () => digitalTwinApi.scenario(tickers, weights, strategy, years).then(res => res.data),
    enabled: tickers.length > 0 && weights.length > 0,
    staleTime: 10 * 60 * 1000,
  })

export const useCompareScenarios = (tickers: string[], weights: number[], strategies: string[], years = 1) =>
  useQuery({
    queryKey: ['digital-twin', 'compare-scenarios', tickers, weights, strategies, years],
    queryFn: () => digitalTwinApi.compareScenarios(tickers, weights, strategies, years).then(res => res.data),
    enabled: tickers.length > 0 && weights.length > 0,
    staleTime: 10 * 60 * 1000,
  })

export const useHealthScore = (tickers: string[], weights: number[]) =>
  useQuery({
    queryKey: ['digital-twin', 'health-score', tickers, weights],
    queryFn: () => digitalTwinApi.healthScore(tickers, weights).then(res => res.data),
    enabled: tickers.length > 0 && weights.length > 0,
    staleTime: 10 * 60 * 1000,
  })

// ============================================================================
// Reports Hooks
// ============================================================================

export const useReportAvailability = () =>
  useQuery({
    queryKey: ['reports', 'available'],
    queryFn: () => reportsApi.checkAvailability().then(res => res.data),
    staleTime: 60 * 60 * 1000,
  })

export const useDownloadStockReport = () =>
  useMutation({
    mutationFn: async ({ ticker, startDate, endDate }: { ticker: string; startDate: string; endDate: string }) => {
      const response = await reportsApi.singleStock(ticker, startDate, endDate)
      const blob = new Blob([response.data], { type: 'application/pdf' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${ticker}_risk_report.pdf`
      link.click()
      window.URL.revokeObjectURL(url)
    }
  })

export const useDownloadPortfolioReport = () =>
  useMutation({
    mutationFn: async ({ name, tickers, weights, startDate, endDate }: { name: string; tickers: string[]; weights: Record<string, number>; startDate: string; endDate: string }) => {
      // Convert weights object to array matching ticker order
      const weightsArray = tickers.map(t => weights[t] || 0)
      const response = await reportsApi.portfolio(name, tickers, weightsArray, startDate, endDate)
      const blob = new Blob([response.data], { type: 'application/pdf' })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${name}_portfolio_report.pdf`
      link.click()
      window.URL.revokeObjectURL(url)
    }
  })
