/**
 * Single Stock Analysis Page
 * ==========================
 * Comprehensive stock risk analysis with tabbed interface
 */

import { useState, useEffect } from 'react'
import { Tab } from '@headlessui/react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  ChartBarIcon, 
  ShieldExclamationIcon, 
  BeakerIcon,
  CurrencyDollarIcon,
  DocumentChartBarIcon,
  CpuChipIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline'
import clsx from 'clsx'
import toast from 'react-hot-toast'

import { useCurrentTicker, useDateRange, useRiskSettings, useUIState, usePortfolioStore } from '../store/portfolioStore'
import {
  useRiskMetrics,
  useVaR,
  useGARCH,
  useEVT,
  useMonteCarlo,
  useMLPredict,
  useMLEnsemble,
  useHistoricalData,
  useStockInfo,
  useFamaFrench,
  useKelly,
  useESG,
  useOptionQuick,
  useFundamentals,
  useDCF,
  useQualityScore,
  useSentiment,
} from '../services/hooks'

import LoadingSpinner from '../components/LoadingSpinner'
import MetricCard from '../components/MetricCard'
import PriceChart from '../components/charts/PriceChart'
import MonteCarloChart from '../components/charts/MonteCarloChart'

// Tab Configuration
const TABS = [
  { name: 'Overview', icon: ChartBarIcon },
  { name: 'Risk Analysis', icon: ShieldExclamationIcon },
  { name: 'Factor Analysis', icon: BeakerIcon },
  { name: 'Options', icon: CurrencyDollarIcon },
  { name: 'Fundamentals', icon: DocumentChartBarIcon },
  { name: 'ML Predictions', icon: CpuChipIcon },
]

export default function SingleStock() {
  const { darkMode } = useUIState()
  const currentTicker = useCurrentTicker()
  const dateRange = useDateRange()
  const riskSettings = useRiskSettings()
  const setCurrentTicker = usePortfolioStore((state) => state.setCurrentTicker)
  const setActiveTab = usePortfolioStore((state) => state.setActiveTab)
  
  // Local state for ticker input
  const [tickerInput, setTickerInput] = useState(currentTicker)
  const [selectedTab, setSelectedTab] = useState(0)

  // Sync ticker input with store
  useEffect(() => {
    setTickerInput(currentTicker)
  }, [currentTicker])

  // Persist tab selection
  useEffect(() => {
    setActiveTab(`single-stock-${TABS[selectedTab].name}`)
  }, [selectedTab, setActiveTab])

  // Data Queries - Overview
  const { data: stockInfo } = useStockInfo(currentTicker)
  const { data: historicalData, isLoading: histLoading } = useHistoricalData(currentTicker, dateRange.startDate, dateRange.endDate)
  const { data: metricsData, isLoading: metricsLoading } = useRiskMetrics(currentTicker, dateRange.startDate, dateRange.endDate)

  // Data Queries - Risk Analysis (enabled on tab)
  const { data: varData, isLoading: varLoading } = useVaR(
    currentTicker, dateRange.startDate, dateRange.endDate, riskSettings.confidenceLevel
  )
  const { data: garchData, isLoading: garchLoading } = useGARCH(
    currentTicker, dateRange.startDate, dateRange.endDate, riskSettings.garchP, riskSettings.garchQ
  )
  const { data: evtData, isLoading: evtLoading } = useEVT(
    currentTicker, dateRange.startDate, dateRange.endDate
  )
  const { data: mcData, isLoading: mcLoading } = useMonteCarlo(
    currentTicker, dateRange.startDate, dateRange.endDate, riskSettings.monteCarloSims, riskSettings.varHorizon
  )

  // Data Queries - Factors
  const { data: famaFrenchData, isLoading: ffLoading } = useFamaFrench(currentTicker, dateRange.startDate, dateRange.endDate)
  const { data: kellyData, isLoading: kellyLoading } = useKelly(currentTicker, dateRange.startDate, dateRange.endDate)
  const { data: esgData, isLoading: esgLoading } = useESG(currentTicker)

  // Data Queries - Options
  const { data: optionsData, isLoading: optionsLoading } = useOptionQuick(currentTicker)

  // Data Queries - Fundamentals
  const { data: fundamentalsData, isLoading: fundLoading } = useFundamentals(currentTicker)
  const { data: dcfData, isLoading: dcfLoading } = useDCF(currentTicker)
  const { data: qualityData, isLoading: qualityLoading } = useQualityScore(currentTicker)

  // Data Queries - ML
  const { data: mlData, isLoading: mlLoading } = useMLPredict(currentTicker, dateRange.startDate, dateRange.endDate)
  const { data: ensembleData, isLoading: ensembleLoading } = useMLEnsemble(currentTicker, dateRange.startDate, dateRange.endDate)

  // Sentiment
  const { data: sentimentData } = useSentiment(currentTicker, dateRange.startDate, dateRange.endDate)

  // Handle ticker search
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    const ticker = tickerInput.trim().toUpperCase()
    if (ticker) {
      setCurrentTicker(ticker)
      toast.success(`Analyzing ${ticker}`)
    }
  }

  // Extract data - hooks return data directly from API response
  const info = stockInfo || {} as Record<string, any>
  const metrics = (metricsData as Record<string, any>)?.metrics || {} as Record<string, any>
  const historical = historicalData || {} as Record<string, any>
  const varResult = varData || {} as Record<string, any>
  const garchResult = garchData || {} as Record<string, any>
  const evtResult = evtData || {} as Record<string, any>
  const mcResult = mcData || {} as Record<string, any>
  const ffResult = famaFrenchData || {} as Record<string, any>
  const kellyResult = kellyData || {} as Record<string, any>
  const esgResult = esgData || {} as Record<string, any>
  const optionsResult = optionsData || {} as Record<string, any>
  const fundResult = fundamentalsData || {} as Record<string, any>
  const dcfResult = dcfData || {} as Record<string, any>
  const qualityResult = qualityData || {} as Record<string, any>
  const mlResult = mlData || {} as Record<string, any>
  const ensembleResult = ensembleData || {}

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header with Search */}
      <div className="flex flex-col sm:flex-row justify-between items-start gap-4">
        <div>
          <h1 className={clsx('text-3xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
            Single Stock Analysis
          </h1>
          <p className={clsx('mt-1', darkMode ? 'text-gray-400' : 'text-gray-500')}>
            Comprehensive risk and factor analysis for {currentTicker}
          </p>
        </div>
        
        <form onSubmit={handleSearch} className="flex items-center gap-2">
          <input
            type="text"
            value={tickerInput}
            onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
            placeholder="Ticker symbol..."
            className={clsx(
              'input w-36 uppercase font-mono',
              darkMode && 'bg-gray-800 border-gray-600 text-white'
            )}
          />
          <button type="submit" className="btn-primary">
            Analyze
          </button>
        </form>
      </div>

      {/* Stock Header Card */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}
      >
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h2 className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                {currentTicker}
              </h2>
              {sentimentData && (
                <span className={clsx(
                  'badge',
                  sentimentData.sentiment_score > 0 ? 'badge-success' : sentimentData.sentiment_score < 0 ? 'badge-danger' : 'badge-info'
                )}>
                  {sentimentData.sentiment_label || 'Neutral'}
                </span>
              )}
            </div>
            <p className={clsx('text-lg', darkMode ? 'text-gray-300' : 'text-gray-600')}>
              {info.name || 'Loading...'}
            </p>
            <p className={clsx('text-sm', darkMode ? 'text-gray-500' : 'text-gray-400')}>
              {info.sector} {info.industry && `• ${info.industry}`}
            </p>
          </div>
          <div className="text-right">
            <div className="flex items-center justify-end gap-2">
              <p className={clsx('text-4xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                ${info.current_price?.toFixed(2) || '—'}
              </p>
              {info.change_percent && (
                <span className={clsx(
                  'flex items-center text-sm font-medium px-2 py-1 rounded',
                  info.change_percent >= 0 
                    ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-400' 
                    : 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-400'
                )}>
                  {info.change_percent >= 0 ? <ArrowTrendingUpIcon className="h-4 w-4 mr-1" /> : <ArrowTrendingDownIcon className="h-4 w-4 mr-1" />}
                  {info.change_percent?.toFixed(2)}%
                </span>
              )}
            </div>
            <p className={clsx('text-sm mt-1', darkMode ? 'text-gray-400' : 'text-gray-500')}>
              Beta: {info.beta?.toFixed(2) || '—'} | Market Cap: {info.market_cap ? `$${(info.market_cap / 1e9).toFixed(1)}B` : '—'}
            </p>
          </div>
        </div>
      </motion.div>

      {/* Tab Navigation */}
      <Tab.Group selectedIndex={selectedTab} onChange={setSelectedTab}>
        <Tab.List className={clsx('tab-group', darkMode && 'bg-gray-800')}>
          {TABS.map((tab) => (
            <Tab
              key={tab.name}
              className={({ selected }) =>
                clsx(
                  'tab flex items-center justify-center gap-2',
                  selected
                    ? darkMode ? 'tab-active bg-gray-700 text-emerald-400' : 'tab-active'
                    : darkMode ? 'tab-inactive text-gray-400' : 'tab-inactive'
                )
              }
            >
              <tab.icon className="h-4 w-4" />
              <span className="hidden sm:inline">{tab.name}</span>
            </Tab>
          ))}
        </Tab.List>

        <Tab.Panels className="mt-6">
          <AnimatePresence mode="wait">
            {/* Overview Tab */}
            <Tab.Panel as={motion.div} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              {metricsLoading || histLoading ? (
                <div className="flex justify-center py-12">
                  <LoadingSpinner size="lg" />
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Key Metrics Grid */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricCard
                      label="Annualized Return"
                      value={`${((metrics.annualized_return || 0) * 100).toFixed(2)}%`}
                      positive={(metrics.annualized_return || 0) >= 0}
                    />
                    <MetricCard
                      label="Volatility (Ann.)"
                      value={`${((metrics.annualized_volatility || 0) * 100).toFixed(2)}%`}
                    />
                    <MetricCard
                      label="Sharpe Ratio"
                      value={(metrics.sharpe_ratio || 0).toFixed(3)}
                      positive={(metrics.sharpe_ratio || 0) >= 1}
                    />
                    <MetricCard
                      label="Sortino Ratio"
                      value={(metrics.sortino_ratio || 0).toFixed(3)}
                      positive={(metrics.sortino_ratio || 0) >= 1}
                    />
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricCard
                      label="Max Drawdown"
                      value={`${((metrics.max_drawdown || 0) * 100).toFixed(2)}%`}
                      positive={false}
                    />
                    <MetricCard
                      label="Calmar Ratio"
                      value={(metrics.calmar_ratio || 0).toFixed(3)}
                      positive={(metrics.calmar_ratio || 0) >= 1}
                    />
                    <MetricCard
                      label="Skewness"
                      value={(metrics.skewness || 0).toFixed(3)}
                    />
                    <MetricCard
                      label="Kurtosis"
                      value={(metrics.kurtosis || 0).toFixed(3)}
                    />
                  </div>

                  {/* Price Chart */}
                  <div className={clsx('chart-container', darkMode && 'bg-gray-800 border-gray-700')}>
                    <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                      Price History
                    </h3>
                    {historical.dates && historical.close ? (
                      <PriceChart
                        dates={historical.dates}
                        prices={historical.close}
                        ticker={currentTicker}
                      />
                    ) : (
                      <p className={clsx('text-center py-8', darkMode ? 'text-gray-500' : 'text-gray-400')}>
                        No price data available
                      </p>
                    )}
                  </div>
                </div>
              )}
            </Tab.Panel>

            {/* Risk Analysis Tab */}
            <Tab.Panel as={motion.div} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="space-y-6">
                {/* VaR Section */}
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    Value at Risk (VaR) Analysis
                  </h3>
                  {varLoading ? (
                    <LoadingSpinner />
                  ) : (
                    <div className="grid grid-cols-3 gap-4">
                      <div className={clsx('p-4 rounded-lg text-center', darkMode ? 'bg-gray-700' : 'bg-red-50')}>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Parametric VaR</p>
                        <p className="text-3xl font-bold text-red-500">
                          {((varResult.var_parametric || 0) * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div className={clsx('p-4 rounded-lg text-center', darkMode ? 'bg-gray-700' : 'bg-red-50')}>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Historical VaR</p>
                        <p className="text-3xl font-bold text-red-500">
                          {((varResult.var_historical || 0) * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div className={clsx('p-4 rounded-lg text-center', darkMode ? 'bg-gray-700' : 'bg-red-50')}>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>CVaR (ES)</p>
                        <p className="text-3xl font-bold text-red-600">
                          {((varResult.cvar || 0) * 100).toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  )}
                  {varResult.var_interpretation && (
                    <p className={clsx('mt-4 text-sm', darkMode ? 'text-gray-400' : 'text-gray-600')}>
                      <InformationCircleIcon className="h-4 w-4 inline mr-1" />
                      {varResult.var_interpretation}
                    </p>
                  )}
                </div>

                {/* GARCH Section */}
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    GARCH Volatility Model
                  </h3>
                  {garchLoading ? (
                    <LoadingSpinner />
                  ) : (
                    <>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <div>
                          <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Model</p>
                          <p className={clsx('text-lg font-mono', darkMode ? 'text-white' : 'text-gray-900')}>
                            GARCH({riskSettings.garchP},{riskSettings.garchQ})
                          </p>
                        </div>
                        <div>
                          <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Persistence</p>
                          <p className={clsx('text-lg font-mono', darkMode ? 'text-white' : 'text-gray-900')}>
                            {(garchResult.persistence || 0).toFixed(4)}
                          </p>
                        </div>
                        <div>
                          <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Current Vol</p>
                          <p className="text-lg font-mono text-amber-500">
                            {((garchResult.current_volatility || 0) * 100).toFixed(2)}%
                          </p>
                        </div>
                        <div>
                          <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Forecast Vol</p>
                          <p className="text-lg font-mono text-emerald-500">
                            {((garchResult.forecast_volatility || 0) * 100).toFixed(2)}%
                          </p>
                        </div>
                      </div>
                      <div className={clsx('p-4 rounded-lg', darkMode ? 'bg-gray-700' : 'bg-gray-50')}>
                        <p className={clsx('text-sm mb-2', darkMode ? 'text-gray-400' : 'text-gray-600')}>Parameters</p>
                        <div className="flex gap-8 font-mono text-sm">
                          <span>ω = {(garchResult.omega || 0).toExponential(4)}</span>
                          <span>α = {(garchResult.alpha || 0).toFixed(4)}</span>
                          <span>β = {(garchResult.beta || 0).toFixed(4)}</span>
                        </div>
                      </div>
                    </>
                  )}
                </div>

                {/* EVT Section */}
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    Extreme Value Theory (EVT)
                  </h3>
                  {evtLoading ? (
                    <LoadingSpinner />
                  ) : (
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Shape (ξ)</p>
                        <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                          {(evtResult.shape || 0).toFixed(4)}
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>EVT VaR 99%</p>
                        <p className="text-2xl font-bold text-red-500">
                          {((evtResult.evt_var_99 || 0) * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>EVT CVaR 99%</p>
                        <p className="text-2xl font-bold text-red-600">
                          {((evtResult.evt_cvar_99 || 0) * 100).toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Monte Carlo Section */}
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    Monte Carlo Simulation
                  </h3>
                  {mcLoading ? (
                    <LoadingSpinner />
                  ) : (
                    <>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                        <MetricCard label="Simulations" value={mcResult.n_simulations?.toLocaleString() || '10,000'} />
                        <MetricCard label="Horizon" value={`${mcResult.horizon_days || 10} days`} />
                        <MetricCard label="MC VaR 95%" value={`${((mcResult.var_95 || 0) * 100).toFixed(2)}%`} positive={false} />
                        <MetricCard label="MC VaR 99%" value={`${((mcResult.var_99 || 0) * 100).toFixed(2)}%`} positive={false} />
                      </div>
                      {mcResult.percentiles && (
                        <MonteCarloChart percentiles={mcResult.percentiles} />
                      )}
                    </>
                  )}
                </div>
              </div>
            </Tab.Panel>

            {/* Factor Analysis Tab */}
            <Tab.Panel as={motion.div} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="space-y-6">
                {/* Fama-French */}
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    Fama-French Three Factor Model
                  </h3>
                  {ffLoading ? (
                    <LoadingSpinner />
                  ) : (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Alpha (α)</p>
                        <p className={clsx('text-2xl font-bold', ffResult.alpha >= 0 ? 'text-emerald-500' : 'text-red-500')}>
                          {((ffResult.alpha || 0) * 100).toFixed(3)}%
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Market Beta</p>
                        <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                          {(ffResult.market_beta || 0).toFixed(3)}
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>SMB (Size)</p>
                        <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                          {(ffResult.smb_beta || 0).toFixed(3)}
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>HML (Value)</p>
                        <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                          {(ffResult.hml_beta || 0).toFixed(3)}
                        </p>
                      </div>
                    </div>
                  )}
                  {ffResult.r_squared && (
                    <p className={clsx('mt-4 text-sm', darkMode ? 'text-gray-400' : 'text-gray-600')}>
                      Model R² = {(ffResult.r_squared * 100).toFixed(1)}%
                    </p>
                  )}
                </div>

                {/* Kelly Criterion */}
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    Kelly Criterion
                  </h3>
                  {kellyLoading ? (
                    <LoadingSpinner />
                  ) : (
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Full Kelly</p>
                        <p className="text-2xl font-bold text-emerald-500">
                          {((kellyResult.full_kelly || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Half Kelly</p>
                        <p className="text-2xl font-bold text-amber-500">
                          {((kellyResult.half_kelly || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Quarter Kelly</p>
                        <p className="text-2xl font-bold text-blue-500">
                          {((kellyResult.quarter_kelly || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  )}
                  <p className={clsx('mt-4 text-sm', darkMode ? 'text-gray-400' : 'text-gray-600')}>
                    <InformationCircleIcon className="h-4 w-4 inline mr-1" />
                    Kelly Criterion suggests optimal position sizing based on edge and odds
                  </p>
                </div>

                {/* ESG */}
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    ESG Scores
                  </h3>
                  {esgLoading ? (
                    <LoadingSpinner />
                  ) : esgResult.total_score ? (
                    <div className="grid grid-cols-4 gap-4">
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Total ESG</p>
                        <p className={clsx('text-2xl font-bold', esgResult.total_score >= 50 ? 'text-emerald-500' : 'text-amber-500')}>
                          {esgResult.total_score?.toFixed(1)}
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Environmental</p>
                        <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                          {esgResult.environment_score?.toFixed(1) || '—'}
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Social</p>
                        <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                          {esgResult.social_score?.toFixed(1) || '—'}
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Governance</p>
                        <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                          {esgResult.governance_score?.toFixed(1) || '—'}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <p className={clsx('text-center py-4', darkMode ? 'text-gray-500' : 'text-gray-400')}>
                      ESG data not available for this ticker
                    </p>
                  )}
                </div>
              </div>
            </Tab.Panel>

            {/* Options Tab */}
            <Tab.Panel as={motion.div} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="space-y-6">
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    Options Analysis (ATM, 30-Day)
                  </h3>
                  {optionsLoading ? (
                    <LoadingSpinner />
                  ) : optionsResult.call_price ? (
                    <>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                        <div>
                          <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Call Price</p>
                          <p className="text-2xl font-bold text-emerald-500">
                            ${optionsResult.call_price?.toFixed(2)}
                          </p>
                        </div>
                        <div>
                          <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Put Price</p>
                          <p className="text-2xl font-bold text-red-500">
                            ${optionsResult.put_price?.toFixed(2)}
                          </p>
                        </div>
                        <div>
                          <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Implied Volatility</p>
                          <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                            {((optionsResult.implied_volatility || 0) * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div>
                          <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Spot Price</p>
                          <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                            ${optionsResult.spot_price?.toFixed(2)}
                          </p>
                        </div>
                      </div>

                      {/* Greeks */}
                      <h4 className={clsx('text-md font-semibold mb-3', darkMode ? 'text-gray-300' : 'text-gray-700')}>
                        Option Greeks (Call)
                      </h4>
                      <div className="grid grid-cols-5 gap-4">
                        <div className={clsx('p-3 rounded-lg text-center', darkMode ? 'bg-gray-700' : 'bg-gray-50')}>
                          <p className={clsx('text-xs', darkMode ? 'text-gray-400' : 'text-gray-500')}>Delta (Δ)</p>
                          <p className={clsx('text-lg font-mono', darkMode ? 'text-white' : 'text-gray-900')}>
                            {optionsResult.greeks?.delta?.toFixed(4) || '—'}
                          </p>
                        </div>
                        <div className={clsx('p-3 rounded-lg text-center', darkMode ? 'bg-gray-700' : 'bg-gray-50')}>
                          <p className={clsx('text-xs', darkMode ? 'text-gray-400' : 'text-gray-500')}>Gamma (Γ)</p>
                          <p className={clsx('text-lg font-mono', darkMode ? 'text-white' : 'text-gray-900')}>
                            {optionsResult.greeks?.gamma?.toFixed(4) || '—'}
                          </p>
                        </div>
                        <div className={clsx('p-3 rounded-lg text-center', darkMode ? 'bg-gray-700' : 'bg-gray-50')}>
                          <p className={clsx('text-xs', darkMode ? 'text-gray-400' : 'text-gray-500')}>Theta (Θ)</p>
                          <p className={clsx('text-lg font-mono', darkMode ? 'text-white' : 'text-gray-900')}>
                            {optionsResult.greeks?.theta?.toFixed(4) || '—'}
                          </p>
                        </div>
                        <div className={clsx('p-3 rounded-lg text-center', darkMode ? 'bg-gray-700' : 'bg-gray-50')}>
                          <p className={clsx('text-xs', darkMode ? 'text-gray-400' : 'text-gray-500')}>Vega (ν)</p>
                          <p className={clsx('text-lg font-mono', darkMode ? 'text-white' : 'text-gray-900')}>
                            {optionsResult.greeks?.vega?.toFixed(4) || '—'}
                          </p>
                        </div>
                        <div className={clsx('p-3 rounded-lg text-center', darkMode ? 'bg-gray-700' : 'bg-gray-50')}>
                          <p className={clsx('text-xs', darkMode ? 'text-gray-400' : 'text-gray-500')}>Rho (ρ)</p>
                          <p className={clsx('text-lg font-mono', darkMode ? 'text-white' : 'text-gray-900')}>
                            {optionsResult.greeks?.rho?.toFixed(4) || '—'}
                          </p>
                        </div>
                      </div>
                    </>
                  ) : (
                    <p className={clsx('text-center py-8', darkMode ? 'text-gray-500' : 'text-gray-400')}>
                      Options data not available
                    </p>
                  )}
                </div>
              </div>
            </Tab.Panel>

            {/* Fundamentals Tab */}
            <Tab.Panel as={motion.div} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="space-y-6">
                {/* Key Ratios */}
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    Key Financial Ratios
                  </h3>
                  {fundLoading ? (
                    <LoadingSpinner />
                  ) : (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>P/E Ratio</p>
                        <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                          {fundResult.pe_ratio?.toFixed(2) || '—'}
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>P/B Ratio</p>
                        <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                          {fundResult.pb_ratio?.toFixed(2) || '—'}
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>ROE</p>
                        <p className="text-2xl font-bold text-emerald-500">
                          {fundResult.roe ? `${(fundResult.roe * 100).toFixed(1)}%` : '—'}
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Debt/Equity</p>
                        <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                          {fundResult.debt_to_equity?.toFixed(2) || '—'}
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* DCF Valuation */}
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    DCF Valuation
                  </h3>
                  {dcfLoading ? (
                    <LoadingSpinner />
                  ) : dcfResult.intrinsic_value ? (
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Intrinsic Value</p>
                        <p className="text-2xl font-bold text-emerald-500">
                          ${dcfResult.intrinsic_value?.toFixed(2)}
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Current Price</p>
                        <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                          ${dcfResult.current_price?.toFixed(2)}
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Upside/Downside</p>
                        <p className={clsx(
                          'text-2xl font-bold',
                          (dcfResult.upside || 0) >= 0 ? 'text-emerald-500' : 'text-red-500'
                        )}>
                          {dcfResult.upside >= 0 ? '+' : ''}{((dcfResult.upside || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  ) : (
                    <p className={clsx('text-center py-4', darkMode ? 'text-gray-500' : 'text-gray-400')}>
                      DCF data not available
                    </p>
                  )}
                </div>

                {/* Quality Score */}
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    Quality Score
                  </h3>
                  {qualityLoading ? (
                    <LoadingSpinner />
                  ) : qualityResult.overall_score !== undefined ? (
                    <div className="flex items-center gap-8">
                      <div className="text-center">
                        <div className={clsx(
                          'w-24 h-24 rounded-full flex items-center justify-center text-3xl font-bold',
                          qualityResult.overall_score >= 70 ? 'bg-emerald-100 text-emerald-600' :
                          qualityResult.overall_score >= 50 ? 'bg-amber-100 text-amber-600' :
                          'bg-red-100 text-red-600'
                        )}>
                          {qualityResult.overall_score}
                        </div>
                        <p className={clsx('mt-2 text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Overall</p>
                      </div>
                      <div className="flex-1 grid grid-cols-2 gap-4">
                        <div>
                          <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Profitability</p>
                          <p className={clsx('text-xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                            {qualityResult.profitability_score}/100
                          </p>
                        </div>
                        <div>
                          <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Growth</p>
                          <p className={clsx('text-xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                            {qualityResult.growth_score}/100
                          </p>
                        </div>
                        <div>
                          <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Safety</p>
                          <p className={clsx('text-xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                            {qualityResult.safety_score}/100
                          </p>
                        </div>
                        <div>
                          <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Value</p>
                          <p className={clsx('text-xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                            {qualityResult.value_score}/100
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className={clsx('text-center py-4', darkMode ? 'text-gray-500' : 'text-gray-400')}>
                      Quality score not available
                    </p>
                  )}
                </div>
              </div>
            </Tab.Panel>

            {/* ML Predictions Tab */}
            <Tab.Panel as={motion.div} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="space-y-6">
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    ML Risk Prediction
                  </h3>
                  {mlLoading ? (
                    <LoadingSpinner />
                  ) : (
                    <>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                        <MetricCard label="Model" value={mlResult.model_type || 'XGBoost'} />
                        <MetricCard label="Predicted VaR" value={`${((mlResult.predicted_var || 0) * 100).toFixed(2)}%`} positive={false} />
                        <MetricCard label="Train Score" value={(mlResult.training_score || 0).toFixed(3)} />
                        <MetricCard label="Test Score" value={(mlResult.test_score || 0).toFixed(3)} />
                      </div>

                      {mlResult.feature_importance && (
                        <div>
                          <h4 className={clsx('text-md font-semibold mb-3', darkMode ? 'text-gray-300' : 'text-gray-700')}>
                            Feature Importance
                          </h4>
                          <div className="space-y-2">
                            {Object.entries(mlResult.feature_importance)
                              .sort((a, b) => (b[1] as number) - (a[1] as number))
                              .slice(0, 8)
                              .map(([feature, importance]) => (
                                <div key={feature} className="flex items-center">
                                  <span className={clsx('w-28 text-sm truncate', darkMode ? 'text-gray-400' : 'text-gray-600')}>
                                    {feature}
                                  </span>
                                  <div className={clsx('flex-1 rounded-full h-2 mx-2', darkMode ? 'bg-gray-700' : 'bg-gray-200')}>
                                    <div
                                      className="bg-gradient-to-r from-emerald-500 to-teal-500 h-2 rounded-full"
                                      style={{ width: `${(importance as number) * 100}%` }}
                                    />
                                  </div>
                                  <span className={clsx('text-sm w-14 text-right', darkMode ? 'text-gray-400' : 'text-gray-500')}>
                                    {((importance as number) * 100).toFixed(1)}%
                                  </span>
                                </div>
                              ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>

                {/* Ensemble */}
                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
                    Ensemble Model
                  </h3>
                  {ensembleLoading ? (
                    <LoadingSpinner />
                  ) : ensembleResult.ensemble_prediction ? (
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Ensemble VaR</p>
                        <p className="text-2xl font-bold text-red-500">
                          {((ensembleResult.ensemble_prediction || 0) * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Model Agreement</p>
                        <p className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                          {((ensembleResult.model_agreement || 0) * 100).toFixed(0)}%
                        </p>
                      </div>
                      <div>
                        <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Confidence</p>
                        <p className={clsx(
                          'text-2xl font-bold',
                          (ensembleResult.confidence || 0) >= 0.7 ? 'text-emerald-500' : 'text-amber-500'
                        )}>
                          {((ensembleResult.confidence || 0) * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                  ) : (
                    <p className={clsx('text-center py-4', darkMode ? 'text-gray-500' : 'text-gray-400')}>
                      Ensemble prediction not available
                    </p>
                  )}
                </div>
              </div>
            </Tab.Panel>
          </AnimatePresence>
        </Tab.Panels>
      </Tab.Group>
    </div>
  )
}
