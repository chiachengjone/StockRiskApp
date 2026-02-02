import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Tab } from '@headlessui/react'
import { motion } from 'framer-motion'
import clsx from 'clsx'
import { portfolioApi } from '@/services/api'
import { useUIState } from '@/store/portfolioStore'
import LoadingSpinner from '@/components/LoadingSpinner'
import MetricCard from '@/components/MetricCard'
import PieChart from '@/components/charts/PieChart'
import EfficientFrontierChart from '@/components/charts/EfficientFrontierChart'
import CorrelationHeatmap from '@/components/charts/CorrelationHeatmap'

const TABS = [
  'Risk Parity',
  'HRP',
  'Black-Litterman',
  'Efficient Frontier',
  'Correlation',
  'Portfolio VaR',
]

export default function Portfolio() {
  const { darkMode } = useUIState()
  const [tickers, setTickers] = useState(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
  const [tickerInput, setTickerInput] = useState(tickers.join(', '))
  const [selectedTab, setSelectedTab] = useState(0)
  const [riskFreeRate, setRiskFreeRate] = useState(0.05)
  const [targetReturn, setTargetReturn] = useState(0.12)
  
  const endDate = new Date().toISOString().split('T')[0]
  const startDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]

  // Risk Parity
  const { data: riskParityData, isLoading: rpLoading } = useQuery({
    queryKey: ['riskParity', tickers, startDate, endDate],
    queryFn: () => portfolioApi.riskParity(tickers, startDate, endDate, riskFreeRate),
    enabled: tickers.length > 1 && selectedTab === 0
  })

  // HRP
  const { data: hrpData, isLoading: hrpLoading } = useQuery({
    queryKey: ['hrp', tickers, startDate, endDate],
    queryFn: () => portfolioApi.hrp(tickers, startDate, endDate),
    enabled: tickers.length > 1 && selectedTab === 1
  })

  // Black-Litterman (with default views)
  const blViews: Record<string, number> = tickers.reduce((acc, t, i) => ({
    ...acc,
    [t]: i === 0 ? 0.10 : 0.05  // Example: 10% expected return for first, 5% for others
  }), {} as Record<string, number>)

  const { data: blData, isLoading: blLoading } = useQuery({
    queryKey: ['blackLitterman', tickers, startDate, endDate],
    queryFn: () => portfolioApi.blackLitterman(
      tickers,
      blViews,
      startDate,
      endDate,
      riskFreeRate
    ),
    enabled: tickers.length > 1 && selectedTab === 2
  })

  // Efficient Frontier
  const { data: efData, isLoading: efLoading } = useQuery({
    queryKey: ['efficientFrontier', tickers, startDate, endDate],
    queryFn: () => portfolioApi.efficientFrontier(tickers, startDate, endDate, riskFreeRate),
    enabled: tickers.length > 1 && selectedTab === 3
  })

  // Correlation
  const { data: corrData, isLoading: corrLoading } = useQuery({
    queryKey: ['correlation', tickers, startDate, endDate],
    queryFn: () => portfolioApi.correlationAnalysis(tickers, startDate, endDate),
    enabled: tickers.length > 1 && selectedTab === 4
  })

  // Portfolio VaR
  const weights = tickers.reduce((acc, t) => ({
    ...acc,
    [t]: 1 / tickers.length  // Equal weight
  }), {} as Record<string, number>)

  const { data: varData, isLoading: varLoading } = useQuery({
    queryKey: ['portfolioVaR', tickers, startDate, endDate],
    queryFn: () => portfolioApi.portfolioVaR(tickers, weights, startDate, endDate),
    enabled: tickers.length > 1 && selectedTab === 5
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const newTickers = tickerInput
      .split(',')
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t.length > 0)
    setTickers(newTickers)
  }

  // Type-safe data extraction
  type AnyData = Record<string, any>
  const riskParity = (riskParityData as AnyData)?.data || {} as AnyData
  const hrp = (hrpData as AnyData)?.data || {} as AnyData
  const bl = (blData as AnyData)?.data || {} as AnyData
  const ef = (efData as AnyData)?.data || {} as AnyData
  const corr = (corrData as AnyData)?.data || {} as AnyData
  const pVar = (varData as AnyData)?.data || {} as AnyData

  return (
    <motion.div 
      className="space-y-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className={clsx(
            'text-2xl font-bold',
            darkMode ? 'text-white' : 'text-gray-900'
          )}>Portfolio Analysis</h1>
          <p className={clsx(darkMode ? 'text-gray-400' : 'text-gray-500')}>
            Analyze {tickers.length} assets: {tickers.join(', ')}
          </p>
        </div>
        
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <input
            type="text"
            value={tickerInput}
            onChange={(e) => setTickerInput(e.target.value)}
            placeholder="AAPL, MSFT, GOOGL..."
            className={clsx(
              'input w-64',
              darkMode && 'bg-gray-800 border-gray-700 text-white placeholder-gray-400'
            )}
          />
          <button type="submit" className="btn-primary">
            Analyze
          </button>
        </form>
      </div>

      {/* Parameters */}
      <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className={clsx('label', darkMode && 'text-gray-300')}>Risk-Free Rate</label>
            <input
              type="range"
              min="0"
              max="10"
              step="0.5"
              value={riskFreeRate * 100}
              onChange={(e) => setRiskFreeRate(parseFloat(e.target.value) / 100)}
              className="w-full accent-emerald-500"
            />
            <span className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>
              {(riskFreeRate * 100).toFixed(1)}%
            </span>
          </div>
          <div>
            <label className={clsx('label', darkMode && 'text-gray-300')}>Target Return (Black-Litterman)</label>
            <input
              type="range"
              min="5"
              max="30"
              step="1"
              value={targetReturn * 100}
              onChange={(e) => setTargetReturn(parseFloat(e.target.value) / 100)}
              className="w-full accent-emerald-500"
            />
            <span className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>
              {(targetReturn * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <Tab.Group selectedIndex={selectedTab} onChange={setSelectedTab}>
        <Tab.List className={clsx(
          'flex space-x-1 rounded-xl p-1',
          darkMode ? 'bg-gray-800' : 'bg-gray-100'
        )}>
          {TABS.map((tab) => (
            <Tab
              key={tab}
              className={({ selected }) =>
                clsx(
                  'w-full rounded-lg py-2.5 text-sm font-medium leading-5 transition-all',
                  'ring-white ring-opacity-60 ring-offset-2 focus:outline-none focus:ring-2',
                  selected
                    ? darkMode 
                      ? 'bg-emerald-600 shadow text-white'
                      : 'bg-white shadow text-emerald-700'
                    : darkMode
                      ? 'text-gray-400 hover:bg-gray-700 hover:text-gray-200'
                      : 'text-gray-600 hover:bg-white/[0.5] hover:text-gray-800'
                )
              }
            >
              {tab}
            </Tab>
          ))}
        </Tab.List>

        <Tab.Panels className="mt-4">
          {/* Risk Parity */}
          <Tab.Panel>
            {rpLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    label="Portfolio Return"
                    value={`${((riskParity.portfolio_return || 0) * 100).toFixed(2)}%`}
                    positive={(riskParity.portfolio_return || 0) >= 0}
                  />
                  <MetricCard
                    label="Portfolio Risk"
                    value={`${((riskParity.portfolio_volatility || 0) * 100).toFixed(2)}%`}
                  />
                  <MetricCard
                    label="Sharpe Ratio"
                    value={(riskParity.sharpe_ratio || 0).toFixed(2)}
                    positive={(riskParity.sharpe_ratio || 0) >= 1}
                  />
                  <MetricCard
                    label="Diversification"
                    value={(riskParity.diversification_ratio || 0).toFixed(2)}
                  />
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                    <h3 className={clsx(
                      'text-lg font-semibold mb-4',
                      darkMode ? 'text-white' : 'text-gray-900'
                    )}>Optimal Weights</h3>
                    {riskParity.weights && (
                      <PieChart
                        data={Object.entries(riskParity.weights).map(([name, value]) => ({
                          name,
                          value: (value as number) * 100,
                        }))}
                      />
                    )}
                  </div>
                  <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                    <h3 className={clsx(
                      'text-lg font-semibold mb-4',
                      darkMode ? 'text-white' : 'text-gray-900'
                    )}>Risk Contribution</h3>
                    {riskParity.risk_contributions && (
                      <PieChart
                        data={Object.entries(riskParity.risk_contributions).map(([name, value]) => ({
                          name,
                          value: (value as number) * 100,
                        }))}
                      />
                    )}
                  </div>
                </div>
              </div>
            )}
          </Tab.Panel>

          {/* HRP */}
          <Tab.Panel>
            {hrpLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    label="Portfolio Return"
                    value={`${((hrp.portfolio_return || 0) * 100).toFixed(2)}%`}
                    positive={(hrp.portfolio_return || 0) >= 0}
                  />
                  <MetricCard
                    label="Portfolio Risk"
                    value={`${((hrp.portfolio_volatility || 0) * 100).toFixed(2)}%`}
                  />
                  <MetricCard
                    label="Sharpe Ratio"
                    value={(hrp.sharpe_ratio || 0).toFixed(2)}
                  />
                  <MetricCard
                    label="Diversification"
                    value={(hrp.diversification_ratio || 0).toFixed(2)}
                  />
                </div>

                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx(
                    'text-lg font-semibold mb-4',
                    darkMode ? 'text-white' : 'text-gray-900'
                  )}>HRP Weights</h3>
                  {hrp.weights && (
                    <PieChart
                      data={Object.entries(hrp.weights).map(([name, value]) => ({
                        name,
                        value: (value as number) * 100,
                      }))}
                    />
                  )}
                </div>

                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx(
                    'text-lg font-semibold mb-4',
                    darkMode ? 'text-white' : 'text-gray-900'
                  )}>Cluster Order</h3>
                  <div className="flex flex-wrap gap-2">
                    {(hrp.cluster_order || []).map((ticker: string, i: number) => (
                      <span
                        key={ticker}
                        className={clsx(
                          'px-3 py-1 rounded-full text-sm',
                          darkMode 
                            ? 'bg-emerald-900/50 text-emerald-300' 
                            : 'bg-emerald-100 text-emerald-700'
                        )}
                      >
                        {i + 1}. {ticker}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </Tab.Panel>

          {/* Black-Litterman */}
          <Tab.Panel>
            {blLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    label="Expected Return"
                    value={`${((bl.expected_return || 0) * 100).toFixed(2)}%`}
                    positive={(bl.expected_return || 0) >= 0}
                  />
                  <MetricCard
                    label="Portfolio Risk"
                    value={`${((bl.portfolio_volatility || 0) * 100).toFixed(2)}%`}
                  />
                  <MetricCard
                    label="Sharpe Ratio"
                    value={(bl.sharpe_ratio || 0).toFixed(2)}
                  />
                  <MetricCard
                    label="Information Ratio"
                    value={(bl.information_ratio || 0).toFixed(2)}
                  />
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                    <h3 className={clsx(
                      'text-lg font-semibold mb-4',
                      darkMode ? 'text-white' : 'text-gray-900'
                    )}>Optimal Weights</h3>
                    {bl.weights && (
                      <PieChart
                        data={Object.entries(bl.weights).map(([name, value]) => ({
                          name,
                          value: (value as number) * 100,
                        }))}
                      />
                    )}
                  </div>
                  <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                    <h3 className={clsx(
                      'text-lg font-semibold mb-4',
                      darkMode ? 'text-white' : 'text-gray-900'
                    )}>Prior vs Posterior</h3>
                    {bl.prior_returns && bl.posterior_returns && (
                      <div className="space-y-2">
                        {tickers.map((ticker) => (
                          <div key={ticker} className="flex items-center gap-4">
                            <span className={clsx(
                              'w-16 font-medium',
                              darkMode ? 'text-gray-200' : 'text-gray-900'
                            )}>{ticker}</span>
                            <div className="flex-1 flex gap-2 items-center">
                              <span className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Prior:</span>
                              <span className={clsx('text-sm', darkMode ? 'text-gray-300' : 'text-gray-700')}>
                                {((bl.prior_returns[ticker] || 0) * 100).toFixed(1)}%
                              </span>
                              <span className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>→</span>
                              <span className="text-sm font-medium text-emerald-500">
                                {((bl.posterior_returns[ticker] || 0) * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </Tab.Panel>

          {/* Efficient Frontier */}
          <Tab.Panel>
            {efLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    label="Max Sharpe Return"
                    value={`${((ef.max_sharpe_return || 0) * 100).toFixed(2)}%`}
                    positive={true}
                  />
                  <MetricCard
                    label="Max Sharpe Risk"
                    value={`${((ef.max_sharpe_volatility || 0) * 100).toFixed(2)}%`}
                  />
                  <MetricCard
                    label="Min Vol Return"
                    value={`${((ef.min_vol_return || 0) * 100).toFixed(2)}%`}
                  />
                  <MetricCard
                    label="Min Vol Risk"
                    value={`${((ef.min_vol_volatility || 0) * 100).toFixed(2)}%`}
                  />
                </div>

                {ef.frontier_returns && ef.frontier_volatilities && (
                  <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                    <h3 className={clsx(
                      'text-lg font-semibold mb-4',
                      darkMode ? 'text-white' : 'text-gray-900'
                    )}>Efficient Frontier</h3>
                    <EfficientFrontierChart
                      returns={ef.frontier_returns}
                      volatilities={ef.frontier_volatilities}
                      maxSharpePoint={{
                        x: ef.max_sharpe_volatility,
                        y: ef.max_sharpe_return,
                      }}
                      minVolPoint={{
                        x: ef.min_vol_volatility,
                        y: ef.min_vol_return,
                      }}
                    />
                  </div>
                )}

                <div className="grid grid-cols-2 gap-6">
                  <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                    <h3 className={clsx(
                      'text-lg font-semibold mb-4',
                      darkMode ? 'text-white' : 'text-gray-900'
                    )}>Max Sharpe Weights</h3>
                    {ef.max_sharpe_weights && (
                      <PieChart
                        data={Object.entries(ef.max_sharpe_weights).map(([name, value]) => ({
                          name,
                          value: (value as number) * 100,
                        }))}
                      />
                    )}
                  </div>
                  <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                    <h3 className={clsx(
                      'text-lg font-semibold mb-4',
                      darkMode ? 'text-white' : 'text-gray-900'
                    )}>Min Volatility Weights</h3>
                    {ef.min_vol_weights && (
                      <PieChart
                        data={Object.entries(ef.min_vol_weights).map(([name, value]) => ({
                          name,
                          value: (value as number) * 100,
                        }))}
                      />
                    )}
                  </div>
                </div>
              </div>
            )}
          </Tab.Panel>

          {/* Correlation */}
          <Tab.Panel>
            {corrLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <MetricCard
                    label="Average Correlation"
                    value={(corr.avg_correlation || 0).toFixed(3)}
                  />
                  <MetricCard
                    label="Max Correlation"
                    value={(corr.max_correlation || 0).toFixed(3)}
                  />
                  <MetricCard
                    label="Min Correlation"
                    value={(corr.min_correlation || 0).toFixed(3)}
                  />
                </div>

                {corr.correlation_matrix && (
                  <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                    <h3 className={clsx(
                      'text-lg font-semibold mb-4',
                      darkMode ? 'text-white' : 'text-gray-900'
                    )}>Correlation Matrix</h3>
                    <CorrelationHeatmap
                      matrix={corr.correlation_matrix}
                      labels={tickers}
                    />
                  </div>
                )}

                {corr.high_correlations && corr.high_correlations.length > 0 && (
                  <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                    <h3 className={clsx(
                      'text-lg font-semibold mb-4',
                      darkMode ? 'text-white' : 'text-gray-900'
                    )}>High Correlations (&gt;0.7)</h3>
                    <div className="space-y-2">
                      {corr.high_correlations.map((pair: any, i: number) => (
                        <div key={i} className={clsx(
                          'flex items-center justify-between p-2 rounded',
                          darkMode ? 'bg-yellow-900/30' : 'bg-yellow-50'
                        )}>
                          <span className={darkMode ? 'text-gray-200' : 'text-gray-900'}>
                            {pair.asset1} ↔ {pair.asset2}
                          </span>
                          <span className={clsx(
                            'font-bold',
                            darkMode ? 'text-yellow-400' : 'text-yellow-700'
                          )}>
                            {pair.correlation.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </Tab.Panel>

          {/* Portfolio VaR */}
          <Tab.Panel>
            {varLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-3 gap-4">
                  <div className={clsx(
                    'card text-center',
                    darkMode && 'bg-gray-800 border-gray-700'
                  )}>
                    <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Parametric VaR (95%)</p>
                    <p className="text-3xl font-bold text-red-500">
                      {((pVar.var_parametric || 0) * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className={clsx(
                    'card text-center',
                    darkMode && 'bg-gray-800 border-gray-700'
                  )}>
                    <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Historical VaR (95%)</p>
                    <p className="text-3xl font-bold text-red-500">
                      {((pVar.var_historical || 0) * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className={clsx(
                    'card text-center',
                    darkMode && 'bg-gray-800 border-gray-700'
                  )}>
                    <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>CVaR (Expected Shortfall)</p>
                    <p className="text-3xl font-bold text-red-500">
                      {((pVar.cvar || 0) * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <MetricCard
                    label="Portfolio Return"
                    value={`${((pVar.portfolio_return || 0) * 100).toFixed(2)}%`}
                    positive={(pVar.portfolio_return || 0) >= 0}
                  />
                  <MetricCard
                    label="Portfolio Volatility"
                    value={`${((pVar.portfolio_volatility || 0) * 100).toFixed(2)}%`}
                  />
                </div>

                <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                  <h3 className={clsx(
                    'text-lg font-semibold mb-4',
                    darkMode ? 'text-white' : 'text-gray-900'
                  )}>Component VaR</h3>
                  {pVar.component_var && (
                    <div className="space-y-2">
                      {Object.entries(pVar.component_var)
                        .sort((a, b) => Math.abs(b[1] as number) - Math.abs(a[1] as number))
                        .map(([ticker, var_]) => (
                          <div key={ticker} className="flex items-center">
                            <span className={clsx(
                              'w-16 font-medium',
                              darkMode ? 'text-gray-200' : 'text-gray-900'
                            )}>{ticker}</span>
                            <div className={clsx(
                              'flex-1 rounded-full h-3 mx-2',
                              darkMode ? 'bg-gray-700' : 'bg-gray-200'
                            )}>
                              <div
                                className="bg-red-500 h-3 rounded-full"
                                style={{ width: `${Math.abs(var_ as number) * 100 * 10}%` }}
                              />
                            </div>
                            <span className="text-sm text-red-500 w-20 text-right">
                              {((var_ as number) * 100).toFixed(2)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </motion.div>
  )
}
