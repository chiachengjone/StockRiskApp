import { useState } from 'react'
import { useQuery } from 'react-query'
import { Tab } from '@headlessui/react'
import { riskApi, mlApi, dataApi } from '@/services/api'
import LoadingSpinner from '@/components/LoadingSpinner'
import MetricCard from '@/components/MetricCard'
import PriceChart from '@/components/charts/PriceChart'
import VaRChart from '@/components/charts/VaRChart'
import MonteCarloChart from '@/components/charts/MonteCarloChart'

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(' ')
}

const TABS = [
  'Overview',
  'VaR Analysis',
  'GARCH Model',
  'Monte Carlo',
  'Stress Testing',
  'ML Predictions',
]

export default function SingleStock() {
  const [ticker, setTicker] = useState('AAPL')
  const [confidence, setConfidence] = useState(0.95)
  const [selectedTab, setSelectedTab] = useState(0)
  
  const endDate = new Date().toISOString().split('T')[0]
  const startDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]

  // Queries
  const { data: historicalData, isLoading: histLoading } = useQuery(
    ['historical', ticker, startDate, endDate],
    () => dataApi.getHistorical(ticker, startDate, endDate),
    { enabled: !!ticker }
  )

  const { data: infoData } = useQuery(
    ['info', ticker],
    () => dataApi.getInfo(ticker),
    { enabled: !!ticker }
  )

  const { data: metricsData, isLoading: metricsLoading } = useQuery(
    ['metrics', ticker, startDate, endDate],
    () => riskApi.getMetrics(ticker, startDate, endDate),
    { enabled: !!ticker }
  )

  const { data: varData, isLoading: varLoading } = useQuery(
    ['var', ticker, startDate, endDate, confidence],
    () => riskApi.getVaR(ticker, startDate, endDate, confidence),
    { enabled: !!ticker && selectedTab === 1 }
  )

  const { data: garchData, isLoading: garchLoading } = useQuery(
    ['garch', ticker, startDate, endDate],
    () => riskApi.getGARCH(ticker, startDate, endDate),
    { enabled: !!ticker && selectedTab === 2 }
  )

  const { data: mcData, isLoading: mcLoading } = useQuery(
    ['monteCarlo', ticker, startDate, endDate],
    () => riskApi.getMonteCarlo(ticker, startDate, endDate),
    { enabled: !!ticker && selectedTab === 3 }
  )

  const { data: stressData, isLoading: stressLoading } = useQuery(
    ['stressScenarios'],
    () => riskApi.getStressScenarios(),
    { enabled: selectedTab === 4 }
  )

  const { data: mlData, isLoading: mlLoading } = useQuery(
    ['mlPredict', ticker, startDate, endDate],
    () => mlApi.predict(ticker, startDate, endDate),
    { enabled: !!ticker && selectedTab === 5 }
  )

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const formData = new FormData(e.target as HTMLFormElement)
    setTicker((formData.get('ticker') as string).toUpperCase())
  }

  const metrics = metricsData?.data?.metrics || {}
  const info = infoData?.data || {}
  const historical = historicalData?.data || {}
  const varResult = varData?.data || {}
  const garchResult = garchData?.data || {}
  const mcResult = mcData?.data || {}
  const mlResult = mlData?.data || {}

  return (
    <div className="space-y-6">
      {/* Header with Search */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Single Stock Analysis</h1>
          <p className="text-gray-500">Comprehensive risk analysis for {ticker}</p>
        </div>
        
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <input
            type="text"
            name="ticker"
            defaultValue={ticker}
            placeholder="Ticker symbol..."
            className="input w-32"
          />
          <button type="submit" className="btn-primary">
            Analyze
          </button>
        </form>
      </div>

      {/* Stock Header Card */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold">{ticker}</h2>
            <p className="text-gray-500">{info.name}</p>
            <p className="text-sm text-gray-400">{info.sector} • {info.industry}</p>
          </div>
          <div className="text-right">
            <p className="text-3xl font-bold">${info.current_price?.toFixed(2) || '—'}</p>
            <p className="text-sm text-gray-500">Beta: {info.beta?.toFixed(2) || '—'}</p>
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
          {/* Overview Tab */}
          <Tab.Panel>
            {metricsLoading || histLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <div className="space-y-6">
                {/* Key Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    label="Annualized Return"
                    value={`${((metrics.annualized_return || 0) * 100).toFixed(2)}%`}
                    positive={(metrics.annualized_return || 0) >= 0}
                  />
                  <MetricCard
                    label="Volatility"
                    value={`${((metrics.annualized_volatility || 0) * 100).toFixed(2)}%`}
                  />
                  <MetricCard
                    label="Sharpe Ratio"
                    value={(metrics.sharpe_ratio || 0).toFixed(2)}
                    positive={(metrics.sharpe_ratio || 0) >= 1}
                  />
                  <MetricCard
                    label="Sortino Ratio"
                    value={(metrics.sortino_ratio || 0).toFixed(2)}
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
                    value={(metrics.calmar_ratio || 0).toFixed(2)}
                  />
                  <MetricCard
                    label="Skewness"
                    value={(metrics.skewness || 0).toFixed(2)}
                  />
                  <MetricCard
                    label="Kurtosis"
                    value={(metrics.kurtosis || 0).toFixed(2)}
                  />
                </div>

                {/* Price Chart */}
                <div className="card">
                  <h3 className="text-lg font-semibold mb-4">Price History</h3>
                  {historical.dates && historical.close && (
                    <PriceChart
                      dates={historical.dates}
                      prices={historical.close}
                      ticker={ticker}
                    />
                  )}
                </div>
              </div>
            )}
          </Tab.Panel>

          {/* VaR Analysis Tab */}
          <Tab.Panel>
            {varLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <div className="space-y-6">
                {/* Confidence Slider */}
                <div className="card">
                  <label className="label">Confidence Level: {(confidence * 100).toFixed(0)}%</label>
                  <input
                    type="range"
                    min="90"
                    max="99"
                    value={confidence * 100}
                    onChange={(e) => setConfidence(parseInt(e.target.value) / 100)}
                    className="w-full"
                  />
                </div>

                {/* VaR Results */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="card text-center">
                    <p className="text-sm text-gray-500">Parametric VaR</p>
                    <p className="text-3xl font-bold text-red-600">
                      {((varResult.var_parametric || 0) * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="card text-center">
                    <p className="text-sm text-gray-500">Historical VaR</p>
                    <p className="text-3xl font-bold text-red-600">
                      {((varResult.var_historical || 0) * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="card text-center">
                    <p className="text-sm text-gray-500">CVaR (Expected Shortfall)</p>
                    <p className="text-3xl font-bold text-red-600">
                      {((varResult.cvar || 0) * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>

                <div className="card">
                  <p className="text-gray-600">{varResult.var_interpretation}</p>
                </div>

                {/* VaR Chart */}
                {historical.returns && (
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Returns Distribution</h3>
                    <VaRChart
                      returns={historical.returns.filter((r: number | null) => r !== null)}
                      varValue={varResult.var_historical}
                      cvarValue={varResult.cvar}
                    />
                  </div>
                )}
              </div>
            )}
          </Tab.Panel>

          {/* GARCH Tab */}
          <Tab.Panel>
            {garchLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    label="Model Type"
                    value={garchResult.model_type || 'GARCH(1,1)'}
                  />
                  <MetricCard
                    label="Persistence (α + β)"
                    value={(garchResult.persistence || 0).toFixed(4)}
                  />
                  <MetricCard
                    label="Current Volatility"
                    value={`${((garchResult.current_volatility || 0) * 100).toFixed(2)}%`}
                  />
                  <MetricCard
                    label="Forecast Volatility"
                    value={`${((garchResult.forecast_volatility || 0) * 100).toFixed(2)}%`}
                  />
                </div>

                <div className="card">
                  <h3 className="text-lg font-semibold mb-4">GARCH Parameters</h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <p className="text-sm text-gray-500">Omega (ω)</p>
                      <p className="text-lg font-mono">{(garchResult.omega || 0).toExponential(4)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Alpha (α)</p>
                      <p className="text-lg font-mono">{(garchResult.alpha || 0).toFixed(4)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Beta (β)</p>
                      <p className="text-lg font-mono">{(garchResult.beta || 0).toFixed(4)}</p>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <h3 className="text-lg font-semibold mb-4">GARCH VaR Estimate</h3>
                  <p className="text-2xl font-bold text-red-600">
                    {((garchResult.conditional_var || 0) * 100).toFixed(2)}%
                  </p>
                  <p className="text-sm text-gray-500 mt-2">
                    Annualized Volatility: {((garchResult.annualized_volatility || 0) * 100).toFixed(2)}%
                  </p>
                </div>
              </div>
            )}
          </Tab.Panel>

          {/* Monte Carlo Tab */}
          <Tab.Panel>
            {mcLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    label="Simulations"
                    value={mcResult.n_simulations?.toLocaleString() || '10,000'}
                  />
                  <MetricCard
                    label="Horizon"
                    value={`${mcResult.horizon_days || 10} days`}
                  />
                  <MetricCard
                    label="Mean Return"
                    value={`${((mcResult.mean_return || 0) * 100).toFixed(2)}%`}
                    positive={(mcResult.mean_return || 0) >= 0}
                  />
                  <MetricCard
                    label="Std Dev"
                    value={`${((mcResult.std_return || 0) * 100).toFixed(2)}%`}
                  />
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    label="VaR 95%"
                    value={`${((mcResult.var_95 || 0) * 100).toFixed(2)}%`}
                    positive={false}
                  />
                  <MetricCard
                    label="VaR 99%"
                    value={`${((mcResult.var_99 || 0) * 100).toFixed(2)}%`}
                    positive={false}
                  />
                  <MetricCard
                    label="Worst Case"
                    value={`${((mcResult.worst_case || 0) * 100).toFixed(2)}%`}
                    positive={false}
                  />
                  <MetricCard
                    label="Best Case"
                    value={`${((mcResult.best_case || 0) * 100).toFixed(2)}%`}
                    positive={true}
                  />
                </div>

                <div className="card">
                  <p className="text-gray-600">
                    Probability of 10%+ loss: <span className="font-bold text-red-600">
                      {((mcResult.prob_loss_10pct || 0) * 100).toFixed(2)}%
                    </span>
                  </p>
                </div>

                {mcResult.percentiles && (
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Percentile Distribution</h3>
                    <MonteCarloChart percentiles={mcResult.percentiles} />
                  </div>
                )}
              </div>
            )}
          </Tab.Panel>

          {/* Stress Testing Tab */}
          <Tab.Panel>
            {stressLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <div className="space-y-6">
                <div className="card">
                  <h3 className="text-lg font-semibold mb-4">Available Stress Scenarios</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {(stressData?.data?.scenarios || []).map((scenario: any) => (
                      <div
                        key={scenario.name}
                        className="p-4 border rounded-lg hover:bg-gray-50 cursor-pointer"
                      >
                        <p className="font-medium">{scenario.name}</p>
                        <p className="text-sm text-red-600">
                          Market Shock: {(scenario.market_shock * 100).toFixed(0)}%
                        </p>
                        <p className="text-sm text-gray-500">
                          Vol Multiplier: {scenario.vol_multiplier}x
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </Tab.Panel>

          {/* ML Predictions Tab */}
          <Tab.Panel>
            {mlLoading ? (
              <div className="flex justify-center py-12">
                <LoadingSpinner size="lg" />
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    label="Model Type"
                    value={mlResult.model_type || 'XGBoost'}
                  />
                  <MetricCard
                    label="Predicted VaR"
                    value={`${((mlResult.predicted_var || 0) * 100).toFixed(2)}%`}
                    positive={false}
                  />
                  <MetricCard
                    label="Training Score"
                    value={(mlResult.training_score || 0).toFixed(3)}
                  />
                  <MetricCard
                    label="Test Score"
                    value={(mlResult.test_score || 0).toFixed(3)}
                  />
                </div>

                {mlResult.feature_importance && (
                  <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Feature Importance</h3>
                    <div className="space-y-2">
                      {Object.entries(mlResult.feature_importance)
                        .sort((a, b) => (b[1] as number) - (a[1] as number))
                        .slice(0, 10)
                        .map(([feature, importance]) => (
                          <div key={feature} className="flex items-center">
                            <span className="w-32 text-sm text-gray-600">{feature}</span>
                            <div className="flex-1 bg-gray-200 rounded-full h-2 mx-2">
                              <div
                                className="bg-primary-600 h-2 rounded-full"
                                style={{ width: `${(importance as number) * 100}%` }}
                              />
                            </div>
                            <span className="text-sm text-gray-500">
                              {((importance as number) * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </div>
  )
}
