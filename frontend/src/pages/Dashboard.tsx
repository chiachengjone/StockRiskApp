/**
 * Dashboard Page
 * ==============
 * Main overview with quick access to key features
 */

import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  ChartBarIcon, 
  ChartPieIcon, 
  PresentationChartLineIcon,
  CubeIcon,
  DocumentTextIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ShieldExclamationIcon,
  BoltIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'
import clsx from 'clsx'
import toast from 'react-hot-toast'

import { useCurrentTicker, useDateRange, useUIState, usePortfolioStore } from '../store/portfolioStore'
import { useRiskMetrics, useVaR, useStockInfo, useHistoricalData, useTASignals } from '../services/hooks'
import MetricCard from '../components/MetricCard'
import LoadingSpinner from '../components/LoadingSpinner'
import PriceChart from '../components/charts/PriceChart'

// Quick action cards
const quickActions = [
  {
    title: 'Single Stock Analysis',
    description: 'Comprehensive risk metrics, VaR, GARCH, Monte Carlo',
    href: '/single-stock',
    icon: PresentationChartLineIcon,
    gradient: 'from-emerald-500 to-teal-600',
  },
  {
    title: 'Portfolio Optimization',
    description: 'Risk Parity, HRP, Efficient Frontier analysis',
    href: '/portfolio',
    icon: ChartPieIcon,
    gradient: 'from-purple-500 to-indigo-600',
  },
  {
    title: 'Technical Analysis',
    description: 'RSI, MACD, Bollinger Bands, trading signals',
    href: '/technical-analysis',
    icon: CubeIcon,
    gradient: 'from-amber-500 to-orange-600',
  },
  {
    title: 'Generate Reports',
    description: 'Export comprehensive PDF risk reports',
    href: '/reports',
    icon: DocumentTextIcon,
    gradient: 'from-blue-500 to-cyan-600',
  },
]

export default function Dashboard() {
  const { darkMode } = useUIState()
  const currentTicker = useCurrentTicker()
  const dateRange = useDateRange()
  const setCurrentTicker = usePortfolioStore((state) => state.setCurrentTicker)

  // Data queries
  const { data: stockInfo, isLoading: infoLoading } = useStockInfo(currentTicker)
  const { data: metricsData, isLoading: metricsLoading } = useRiskMetrics(currentTicker, dateRange.startDate, dateRange.endDate)
  const { data: varData, isLoading: varLoading } = useVaR(currentTicker, dateRange.startDate, dateRange.endDate)
  const { data: historicalData, isLoading: histLoading } = useHistoricalData(currentTicker, dateRange.startDate, dateRange.endDate)
  const { data: taSignals } = useTASignals(currentTicker)

  const isLoading = infoLoading || metricsLoading || varLoading || histLoading

  // Extract data
  const info = stockInfo || {} as Record<string, any>
  const metrics = (metricsData as Record<string, any>)?.metrics || {} as Record<string, any>
  const varResult = varData || {} as Record<string, any>
  const historical = historicalData || {} as Record<string, any>
  const signals = taSignals || {} as Record<string, any>

  // Handle ticker search
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    const formData = new FormData(e.target as HTMLFormElement)
    const ticker = (formData.get('ticker') as string).trim().toUpperCase()
    if (ticker) {
      setCurrentTicker(ticker)
      toast.success(`Analyzing ${ticker}`)
    }
  }

  // Calculate overall signal
  const overallSignal = signals.overall_signal || 'neutral'
  // Signal color for future use
  // const signalColor = overallSignal === 'bullish' ? 'text-emerald-500' : overallSignal === 'bearish' ? 'text-red-500' : 'text-gray-500'

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header with Search */}
      <div className="flex flex-col sm:flex-row justify-between items-start gap-4">
        <div>
          <h1 className={clsx('text-3xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
            Dashboard
          </h1>
          <p className={clsx('mt-1', darkMode ? 'text-gray-400' : 'text-gray-500')}>
            Quick overview of stock risk metrics
          </p>
        </div>
        
        <form onSubmit={handleSearch} className="flex items-center gap-2">
          <input
            type="text"
            name="ticker"
            defaultValue={currentTicker}
            placeholder="Ticker..."
            className={clsx(
              'input w-28 uppercase font-mono',
              darkMode && 'bg-gray-800 border-gray-600 text-white'
            )}
          />
          <button type="submit" className="btn-primary">
            <BoltIcon className="h-4 w-4 mr-1" />
            Analyze
          </button>
        </form>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-16">
          <LoadingSpinner size="lg" />
        </div>
      ) : (
        <>
          {/* Stock Header with Signal */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={clsx(
              'card relative overflow-hidden',
              darkMode && 'bg-gray-800 border-gray-700'
            )}
          >
            <div className="absolute top-0 right-0 w-48 h-48 bg-gradient-to-br from-emerald-500/10 to-teal-500/10 rounded-full -translate-y-1/2 translate-x-1/2" />
            <div className="relative flex items-center justify-between">
              <div className="flex items-center gap-6">
                <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                  <ChartBarIcon className="h-8 w-8 text-white" />
                </div>
                <div>
                  <div className="flex items-center gap-3">
                    <h2 className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                      {currentTicker}
                    </h2>
                    <span className={clsx(
                      'badge',
                      overallSignal === 'bullish' ? 'badge-success' : 
                      overallSignal === 'bearish' ? 'badge-danger' : 'badge-info'
                    )}>
                      <SparklesIcon className="h-3 w-3 mr-1" />
                      {overallSignal.charAt(0).toUpperCase() + overallSignal.slice(1)}
                    </span>
                  </div>
                  <p className={clsx('text-lg', darkMode ? 'text-gray-300' : 'text-gray-600')}>
                    {info.name || 'Loading...'}
                  </p>
                  <p className={clsx('text-sm', darkMode ? 'text-gray-500' : 'text-gray-400')}>
                    {info.sector} {info.industry && `• ${info.industry}`}
                  </p>
                </div>
              </div>
              <div className="text-right">
                <div className="flex items-center justify-end gap-2">
                  <p className={clsx('text-4xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                    ${info.current_price?.toFixed(2) || '—'}
                  </p>
                  {info.change_percent !== undefined && (
                    <span className={clsx(
                      'flex items-center text-sm font-semibold px-2 py-1 rounded-lg',
                      info.change_percent >= 0 
                        ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/50 dark:text-emerald-400' 
                        : 'bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-400'
                    )}>
                      {info.change_percent >= 0 ? <ArrowTrendingUpIcon className="h-4 w-4 mr-1" /> : <ArrowTrendingDownIcon className="h-4 w-4 mr-1" />}
                      {info.change_percent >= 0 ? '+' : ''}{info.change_percent?.toFixed(2)}%
                    </span>
                  )}
                </div>
                <p className={clsx('text-sm mt-2', darkMode ? 'text-gray-400' : 'text-gray-500')}>
                  Beta: {info.beta?.toFixed(2) || '—'} | 52W High: ${info.fifty_two_week_high?.toFixed(2) || '—'}
                </p>
              </div>
            </div>
          </motion.div>

          {/* Key Metrics Grid */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-4"
          >
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
              label="Max Drawdown"
              value={`${((metrics.max_drawdown || 0) * 100).toFixed(2)}%`}
              positive={false}
            />
          </motion.div>

          {/* VaR Section */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}
          >
            <div className="flex items-center gap-3 mb-4">
              <ShieldExclamationIcon className="h-6 w-6 text-red-500" />
              <h3 className={clsx('text-lg font-semibold', darkMode ? 'text-white' : 'text-gray-900')}>
                Value at Risk (95% Confidence)
              </h3>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div className={clsx('text-center p-4 rounded-xl', darkMode ? 'bg-gray-700' : 'bg-red-50')}>
                <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Parametric VaR</p>
                <p className="text-3xl font-bold text-red-500">
                  {((varResult.var_parametric || 0) * 100).toFixed(2)}%
                </p>
              </div>
              <div className={clsx('text-center p-4 rounded-xl', darkMode ? 'bg-gray-700' : 'bg-red-50')}>
                <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Historical VaR</p>
                <p className="text-3xl font-bold text-red-500">
                  {((varResult.var_historical || 0) * 100).toFixed(2)}%
                </p>
              </div>
              <div className={clsx('text-center p-4 rounded-xl', darkMode ? 'bg-gray-700' : 'bg-red-50')}>
                <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>CVaR (ES)</p>
                <p className="text-3xl font-bold text-red-600">
                  {((varResult.cvar || 0) * 100).toFixed(2)}%
                </p>
              </div>
            </div>
            {varResult.var_interpretation && (
              <p className={clsx('mt-4 text-sm', darkMode ? 'text-gray-400' : 'text-gray-600')}>
                {varResult.var_interpretation}
              </p>
            )}
          </motion.div>

          {/* Price Chart */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className={clsx('chart-container', darkMode && 'bg-gray-800 border-gray-700')}
          >
            <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
              Price History (1 Year)
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
          </motion.div>

          {/* Quick Actions */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
              Quick Actions
            </h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              {quickActions.map((action) => (
                <Link 
                  key={action.title}
                  to={action.href}
                  className={clsx(
                    'group card hover:shadow-xl transition-all duration-300 hover:-translate-y-1',
                    darkMode && 'bg-gray-800 border-gray-700 hover:border-gray-600'
                  )}
                >
                  <div className="flex items-start gap-4">
                    <div className={clsx('w-12 h-12 rounded-lg bg-gradient-to-br flex items-center justify-center flex-shrink-0', action.gradient)}>
                      <action.icon className="h-6 w-6 text-white" />
                    </div>
                    <div>
                      <h4 className={clsx(
                        'font-semibold group-hover:text-emerald-500 transition-colors',
                        darkMode ? 'text-white' : 'text-gray-900'
                      )}>
                        {action.title}
                      </h4>
                      <p className={clsx('text-sm mt-1', darkMode ? 'text-gray-400' : 'text-gray-500')}>
                        {action.description}
                      </p>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </motion.div>
        </>
      )}
    </div>
  )
}
