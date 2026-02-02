import { useState } from 'react'
import { useQuery } from 'react-query'
import { dataApi, riskApi } from '@/services/api'
import MetricCard from '@/components/MetricCard'
import LoadingSpinner from '@/components/LoadingSpinner'

export default function Dashboard() {
  const [ticker, setTicker] = useState('AAPL')
  
  // Get current date and 1 year ago
  const endDate = new Date().toISOString().split('T')[0]
  const startDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]

  const { data: quoteData, isLoading: quoteLoading } = useQuery(
    ['quote', ticker],
    () => dataApi.getQuote(ticker),
    { enabled: !!ticker }
  )

  const { data: metricsData, isLoading: metricsLoading } = useQuery(
    ['metrics', ticker, startDate, endDate],
    () => riskApi.getMetrics(ticker, startDate, endDate),
    { enabled: !!ticker }
  )

  const { data: varData, isLoading: varLoading } = useQuery(
    ['var', ticker, startDate, endDate],
    () => riskApi.getVaR(ticker, startDate, endDate),
    { enabled: !!ticker }
  )

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const formData = new FormData(e.target as HTMLFormElement)
    setTicker(formData.get('ticker') as string)
  }

  const isLoading = quoteLoading || metricsLoading || varLoading
  const metrics = metricsData?.data?.metrics || {}
  const quote = quoteData?.data || {}
  const varResult = varData?.data || {}

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-500">Quick overview of stock risk metrics</p>
        </div>
        
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <input
            type="text"
            name="ticker"
            defaultValue={ticker}
            placeholder="Enter ticker..."
            className="input w-32"
          />
          <button type="submit" className="btn-primary">
            Analyze
          </button>
        </form>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : (
        <>
          {/* Stock Info */}
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold">{ticker}</h2>
                <p className="text-gray-500">{quote.name || 'Loading...'}</p>
              </div>
              <div className="text-right">
                <p className="text-3xl font-bold">
                  ${quote.price?.toFixed(2) || '—'}
                </p>
                <p className={`text-sm ${(quote.change_pct || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {quote.change_pct >= 0 ? '+' : ''}{(quote.change_pct || 0).toFixed(2)}%
                </p>
              </div>
            </div>
          </div>

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
              label="Max Drawdown"
              value={`${((metrics.max_drawdown || 0) * 100).toFixed(2)}%`}
              positive={false}
            />
          </div>

          {/* VaR Section */}
          <div className="card">
            <h3 className="text-lg font-semibold mb-4">Value at Risk (95% Confidence)</h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-500">Parametric VaR</p>
                <p className="text-2xl font-bold text-red-600">
                  {((varResult.var_parametric || 0) * 100).toFixed(2)}%
                </p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-500">Historical VaR</p>
                <p className="text-2xl font-bold text-red-600">
                  {((varResult.var_historical || 0) * 100).toFixed(2)}%
                </p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-500">CVaR (Expected Shortfall)</p>
                <p className="text-2xl font-bold text-red-600">
                  {((varResult.cvar || 0) * 100).toFixed(2)}%
                </p>
              </div>
            </div>
            <p className="mt-4 text-sm text-gray-500">
              {varResult.var_interpretation || 'VaR interpretation will appear here.'}
            </p>
          </div>

          {/* Quick Links */}
          <div className="grid grid-cols-3 gap-4">
            <a href="/single-stock" className="card hover:shadow-lg transition-shadow text-center">
              <h4 className="font-semibold">Single Stock Analysis</h4>
              <p className="text-sm text-gray-500">Deep dive into individual stock risk</p>
            </a>
            <a href="/portfolio" className="card hover:shadow-lg transition-shadow text-center">
              <h4 className="font-semibold">Portfolio Optimization</h4>
              <p className="text-sm text-gray-500">Risk Parity, Black-Litterman, HRP</p>
            </a>
            <a href="/technical-analysis" className="card hover:shadow-lg transition-shadow text-center">
              <h4 className="font-semibold">Technical Analysis</h4>
              <p className="text-sm text-gray-500">RSI, MACD, Bollinger Bands</p>
            </a>
          </div>
        </>
      )}
    </div>
  )
}
