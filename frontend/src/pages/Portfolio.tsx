import { useState } from 'react'
import { useQuery } from 'react-query'
import { Tab } from '@headlessui/react'
import { portfolioApi } from '@/services/api'
import LoadingSpinner from '@/components/LoadingSpinner'
import MetricCard from '@/components/MetricCard'
import PieChart from '@/components/charts/PieChart'
import EfficientFrontierChart from '@/components/charts/EfficientFrontierChart'
import CorrelationHeatmap from '@/components/charts/CorrelationHeatmap'

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(' ')
}

const TABS = [
  'Risk Parity',
  'HRP',
  'Black-Litterman',
  'Efficient Frontier',
  'Correlation',
  'Portfolio VaR',
]

export default function Portfolio() {
  const [tickers, setTickers] = useState(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
  const [tickerInput, setTickerInput] = useState(tickers.join(', '))
  const [selectedTab, setSelectedTab] = useState(0)
  const [riskFreeRate, setRiskFreeRate] = useState(0.05)
  const [targetReturn, setTargetReturn] = useState(0.12)
  
  const endDate = new Date().toISOString().split('T')[0]
  const startDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]

  // Risk Parity
  const { data: riskParityData, isLoading: rpLoading } = useQuery(
    ['riskParity', tickers, startDate, endDate],
    () => portfolioApi.riskParity(tickers, startDate, endDate, riskFreeRate),
    { enabled: tickers.length > 1 && selectedTab === 0 }
  )

  // HRP
  const { data: hrpData, isLoading: hrpLoading } = useQuery(
    ['hrp', tickers, startDate, endDate],
    () => portfolioApi.hrp(tickers, startDate, endDate),
    { enabled: tickers.length > 1 && selectedTab === 1 }
  )

  // Black-Litterman (with default views)
  const blViews: Record<string, number> = tickers.reduce((acc, t, i) => ({
    ...acc,
    [t]: i === 0 ? 0.10 : 0.05  // Example: 10% expected return for first, 5% for others
  }), {} as Record<string, number>)

  const { data: blData, isLoading: blLoading } = useQuery(
    ['blackLitterman', tickers, startDate, endDate],
    () => portfolioApi.blackLitterman(
      tickers,
      blViews,
      startDate,
      endDate,
      riskFreeRate
    ),
    { enabled: tickers.length > 1 && selectedTab === 2 }
  )

  // Efficient Frontier
  const { data: efData, isLoading: efLoading } = useQuery(
    ['efficientFrontier', tickers, startDate, endDate],
    () => portfolioApi.efficientFrontier(tickers, startDate, endDate, riskFreeRate),
    { enabled: tickers.length > 1 && selectedTab === 3 }
  )

  // Correlation
  const { data: corrData, isLoading: corrLoading } = useQuery(
    ['correlation', tickers, startDate, endDate],
    () => portfolioApi.correlationAnalysis(tickers, startDate, endDate),
    { enabled: tickers.length > 1 && selectedTab === 4 }
  )

  // Portfolio VaR
  const weights = tickers.reduce((acc, t) => ({
    ...acc,
    [t]: 1 / tickers.length  // Equal weight
  }), {} as Record<string, number>)

  const { data: varData, isLoading: varLoading } = useQuery(
    ['portfolioVaR', tickers, startDate, endDate],
    () => portfolioApi.portfolioVaR(tickers, weights, startDate, endDate),
    { enabled: tickers.length > 1 && selectedTab === 5 }
  )

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const newTickers = tickerInput
      .split(',')
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t.length > 0)
    setTickers(newTickers)
  }

  const riskParity = riskParityData?.data || {}
  const hrp = hrpData?.data || {}
  const bl = blData?.data || {}
  const ef = efData?.data || {}
  const corr = corrData?.data || {}
  const pVar = varData?.data || {}

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Portfolio Analysis</h1>
          <p className="text-gray-500">
            Analyze {tickers.length} assets: {tickers.join(', ')}
          </p>
        </div>
        
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <input
            type="text"
            value={tickerInput}
            onChange={(e) => setTickerInput(e.target.value)}
            placeholder="AAPL, MSFT, GOOGL..."
            className="input w-64"
          />
          <button type="submit" className="btn-primary">
            Analyze
          </button>
        </form>
      </div>

      {/* Parameters */}
      <div className="card">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Risk-Free Rate</label>
            <input
              type="range"
              min="0"
              max="10"
              step="0.5"
              value={riskFreeRate * 100}
              onChange={(e) => setRiskFreeRate(parseFloat(e.target.value) / 100)}
              className="w-full"
            />
            <span className="text-sm text-gray-500">{(riskFreeRate * 100).toFixed(1)}%</span>
          </div>
          <div>
            <label className="label">Target Return (Black-Litterman)</label>
            <input
              type="range"
              min="5"
              max="30"
              step="1"
              value={targetReturn * 100}
              onChange={(e) => setTargetReturn(parseFloat(e.target.value) / 100)}
              className="w-full"
            />
            <span className="text-sm text-gray-500">{(targetReturn * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <Tab.Group selectedIndex={selectedTab} onChange={setSelectedTab}>
        <Tab.List className="flex space-x-1 rounded-xl bg-gray-100 p-1">
          {TABS.map((tab) => (
            <Tab
              key={tab}
              className={({ selected }) =>
                classNames(
                  'w-full rounded-lg py-2.5 text-sm font-medium leading-5',
                  'ring-white ring-opacity-60 ring-offset-2 focus:outline-none focus:ring-2',
                  selected
                    ? 'bg-white shadow text-primary-700'
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
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Optimal Weights</h3>
                    {riskParity.weights && (
                      <PieChart
                        data={Object.entries(riskParity.weights).map(([name, value]) => ({
                          name,
                          value: (value as number) * 100,
                        }))}
                      />
                    )}
                  </div>
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Risk Contribution</h3>
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

                <div className="card">
                  <h3 className="text-lg font-semibold mb-4">HRP Weights</h3>
                  {hrp.weights && (
                    <PieChart
                      data={Object.entries(hrp.weights).map(([name, value]) => ({
                        name,
                        value: (value as number) * 100,
                      }))}
                    />
                  )}
                </div>

                <div className="card">
                  <h3 className="text-lg font-semibold mb-4">Cluster Order</h3>
                  <div className="flex flex-wrap gap-2">
                    {(hrp.cluster_order || []).map((ticker: string, i: number) => (
                      <span
                        key={ticker}
                        className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm"
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
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Optimal Weights</h3>
                    {bl.weights && (
                      <PieChart
                        data={Object.entries(bl.weights).map(([name, value]) => ({
                          name,
                          value: (value as number) * 100,
                        }))}
                      />
                    )}
                  </div>
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Prior vs Posterior</h3>
                    {bl.prior_returns && bl.posterior_returns && (
                      <div className="space-y-2">
                        {tickers.map((ticker) => (
                          <div key={ticker} className="flex items-center gap-4">
                            <span className="w-16 font-medium">{ticker}</span>
                            <div className="flex-1 flex gap-2 items-center">
                              <span className="text-sm text-gray-500">Prior:</span>
                              <span className="text-sm">
                                {((bl.prior_returns[ticker] || 0) * 100).toFixed(1)}%
                              </span>
                              <span className="text-sm text-gray-500">→</span>
                              <span className="text-sm font-medium text-primary-600">
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
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Efficient Frontier</h3>
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
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Max Sharpe Weights</h3>
                    {ef.max_sharpe_weights && (
                      <PieChart
                        data={Object.entries(ef.max_sharpe_weights).map(([name, value]) => ({
                          name,
                          value: (value as number) * 100,
                        }))}
                      />
                    )}
                  </div>
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Min Volatility Weights</h3>
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
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Correlation Matrix</h3>
                    <CorrelationHeatmap
                      matrix={corr.correlation_matrix}
                      labels={tickers}
                    />
                  </div>
                )}

                {corr.high_correlations && corr.high_correlations.length > 0 && (
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">High Correlations (&gt;0.7)</h3>
                    <div className="space-y-2">
                      {corr.high_correlations.map((pair: any, i: number) => (
                        <div key={i} className="flex items-center justify-between p-2 bg-yellow-50 rounded">
                          <span>{pair.asset1} ↔ {pair.asset2}</span>
                          <span className="font-bold text-yellow-700">{pair.correlation.toFixed(3)}</span>
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
                  <div className="card text-center">
                    <p className="text-sm text-gray-500">Parametric VaR (95%)</p>
                    <p className="text-3xl font-bold text-red-600">
                      {((pVar.var_parametric || 0) * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="card text-center">
                    <p className="text-sm text-gray-500">Historical VaR (95%)</p>
                    <p className="text-3xl font-bold text-red-600">
                      {((pVar.var_historical || 0) * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="card text-center">
                    <p className="text-sm text-gray-500">CVaR (Expected Shortfall)</p>
                    <p className="text-3xl font-bold text-red-600">
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

                <div className="card">
                  <h3 className="text-lg font-semibold mb-4">Component VaR</h3>
                  {pVar.component_var && (
                    <div className="space-y-2">
                      {Object.entries(pVar.component_var)
                        .sort((a, b) => Math.abs(b[1] as number) - Math.abs(a[1] as number))
                        .map(([ticker, var_]) => (
                          <div key={ticker} className="flex items-center">
                            <span className="w-16 font-medium">{ticker}</span>
                            <div className="flex-1 bg-gray-200 rounded-full h-3 mx-2">
                              <div
                                className="bg-red-500 h-3 rounded-full"
                                style={{ width: `${Math.abs(var_ as number) * 100 * 10}%` }}
                              />
                            </div>
                            <span className="text-sm text-red-600 w-20 text-right">
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
    </div>
  )
}
