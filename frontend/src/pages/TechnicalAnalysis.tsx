import { useState } from 'react'
import { useQuery } from 'react-query'
import { taApi, dataApi } from '@/services/api'
import LoadingSpinner from '@/components/LoadingSpinner'
import MetricCard from '@/components/MetricCard'
import CandlestickChart from '@/components/charts/CandlestickChart'
import TAIndicatorChart from '@/components/charts/TAIndicatorChart'

const SIGNAL_COLORS: Record<string, string> = {
  'Strong Buy': 'bg-green-600 text-white',
  'Buy': 'bg-green-400 text-white',
  'Neutral': 'bg-gray-400 text-white',
  'Sell': 'bg-red-400 text-white',
  'Strong Sell': 'bg-red-600 text-white',
}

export default function TechnicalAnalysis() {
  const [ticker, setTicker] = useState('AAPL')
  const [period, setPeriod] = useState('1y')
  
  // Calculate dates based on period
  const endDate = new Date().toISOString().split('T')[0]
  const periodDays: Record<string, number> = {
    '1m': 30,
    '3m': 90,
    '6m': 180,
    '1y': 365,
    '2y': 730,
  }
  const startDate = new Date(
    Date.now() - (periodDays[period] || 365) * 24 * 60 * 60 * 1000
  ).toISOString().split('T')[0]

  // Fetch all TA indicators
  const { data: taData, isLoading: taLoading } = useQuery(
    ['ta-full', ticker, startDate, endDate],
    () => taApi.getFull(ticker, startDate, endDate),
    { enabled: !!ticker }
  )

  // Fetch signals
  const { data: signalsData, isLoading: signalsLoading } = useQuery(
    ['ta-signals', ticker],
    () => taApi.getSignals(ticker),
    { enabled: !!ticker }
  )

  // Fetch historical data for candlestick
  const { data: historicalData, isLoading: histLoading } = useQuery(
    ['historical', ticker, startDate, endDate],
    () => dataApi.getHistorical(ticker, startDate, endDate),
    { enabled: !!ticker }
  )

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const formData = new FormData(e.target as HTMLFormElement)
    setTicker((formData.get('ticker') as string).toUpperCase())
  }

  const isLoading = taLoading || signalsLoading || histLoading
  const ta = taData?.data || {}
  const signals = signalsData?.data || {}
  const historical = historicalData?.data || {}

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Technical Analysis</h1>
          <p className="text-gray-500">Indicators and signals for {ticker}</p>
        </div>
        
        <div className="flex items-center gap-4">
          {/* Period selector */}
          <div className="flex gap-1">
            {['1m', '3m', '6m', '1y', '2y'].map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`px-3 py-1 text-sm rounded ${
                  period === p
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {p.toUpperCase()}
              </button>
            ))}
          </div>
          
          <form onSubmit={handleSubmit} className="flex items-center gap-2">
            <input
              type="text"
              name="ticker"
              defaultValue={ticker}
              placeholder="Ticker..."
              className="input w-28"
            />
            <button type="submit" className="btn-primary">
              Analyze
            </button>
          </form>
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : (
        <>
          {/* Overall Signal */}
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold">Overall Signal</h2>
                <p className="text-gray-500">Based on multiple technical indicators</p>
              </div>
              <div className={`px-6 py-3 rounded-lg font-bold text-lg ${
                SIGNAL_COLORS[signals.overall_signal] || 'bg-gray-200'
              }`}>
                {signals.overall_signal || 'N/A'}
              </div>
            </div>
          </div>

          {/* Signal Summary */}
          <div className="grid grid-cols-3 gap-4">
            <MetricCard
              label="Buy Signals"
              value={signals.buy_count || 0}
              positive={true}
            />
            <MetricCard
              label="Neutral Signals"
              value={signals.neutral_count || 0}
            />
            <MetricCard
              label="Sell Signals"
              value={signals.sell_count || 0}
              positive={false}
            />
          </div>

          {/* Price Chart with Bollinger Bands */}
          {historical.dates && historical.close && (
            <div className="card">
              <h3 className="text-lg font-semibold mb-4">Price with Bollinger Bands</h3>
              <CandlestickChart
                dates={historical.dates}
                open={historical.open}
                high={historical.high}
                low={historical.low}
                close={historical.close}
                bollingerUpper={ta.bollinger?.upper}
                bollingerMiddle={ta.bollinger?.middle}
                bollingerLower={ta.bollinger?.lower}
              />
            </div>
          )}

          {/* RSI */}
          {ta.rsi && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">RSI (14)</h3>
                <div className="flex items-center gap-4">
                  <span className="text-2xl font-bold">
                    {ta.rsi.value?.toFixed(1)}
                  </span>
                  <span className={`px-3 py-1 rounded text-sm ${
                    ta.rsi.value > 70 ? 'bg-red-100 text-red-700' :
                    ta.rsi.value < 30 ? 'bg-green-100 text-green-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {ta.rsi.value > 70 ? 'Overbought' :
                     ta.rsi.value < 30 ? 'Oversold' : 'Neutral'}
                  </span>
                </div>
              </div>
              {ta.rsi.history && (
                <TAIndicatorChart
                  dates={historical.dates?.slice(-ta.rsi.history.length)}
                  values={ta.rsi.history}
                  name="RSI"
                  color="#8B5CF6"
                  overbought={70}
                  oversold={30}
                />
              )}
            </div>
          )}

          {/* MACD */}
          {ta.macd && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">MACD</h3>
                <div className="flex items-center gap-4">
                  <div>
                    <span className="text-sm text-gray-500">MACD: </span>
                    <span className="font-bold">{ta.macd.macd?.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500">Signal: </span>
                    <span className="font-bold">{ta.macd.signal?.toFixed(2)}</span>
                  </div>
                  <span className={`px-3 py-1 rounded text-sm ${
                    ta.macd.histogram > 0 ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                  }`}>
                    {ta.macd.histogram > 0 ? 'Bullish' : 'Bearish'}
                  </span>
                </div>
              </div>
              {ta.macd.macd_history && (
                <TAIndicatorChart
                  dates={historical.dates?.slice(-ta.macd.macd_history.length)}
                  values={ta.macd.macd_history}
                  values2={ta.macd.signal_history}
                  name="MACD"
                  name2="Signal"
                  color="#3B82F6"
                  color2="#EF4444"
                  histogram={ta.macd.histogram_history}
                />
              )}
            </div>
          )}

          {/* Moving Averages */}
          <div className="card">
            <h3 className="text-lg font-semibold mb-4">Moving Averages</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {ta.sma_20 && (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-500">SMA 20</p>
                  <p className="text-xl font-bold">${ta.sma_20.value?.toFixed(2)}</p>
                  <p className={`text-sm ${
                    historical.close?.[historical.close.length - 1] > ta.sma_20.value
                      ? 'text-green-600' : 'text-red-600'
                  }`}>
                    Price {historical.close?.[historical.close.length - 1] > ta.sma_20.value ? 'Above' : 'Below'}
                  </p>
                </div>
              )}
              {ta.sma_50 && (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-500">SMA 50</p>
                  <p className="text-xl font-bold">${ta.sma_50.value?.toFixed(2)}</p>
                  <p className={`text-sm ${
                    historical.close?.[historical.close.length - 1] > ta.sma_50.value
                      ? 'text-green-600' : 'text-red-600'
                  }`}>
                    Price {historical.close?.[historical.close.length - 1] > ta.sma_50.value ? 'Above' : 'Below'}
                  </p>
                </div>
              )}
              {ta.ema_12 && (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-500">EMA 12</p>
                  <p className="text-xl font-bold">${ta.ema_12.value?.toFixed(2)}</p>
                </div>
              )}
              {ta.ema_26 && (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-500">EMA 26</p>
                  <p className="text-xl font-bold">${ta.ema_26.value?.toFixed(2)}</p>
                </div>
              )}
            </div>
          </div>

          {/* Additional Indicators */}
          <div className="grid grid-cols-2 gap-6">
            {/* ADX */}
            {ta.adx && (
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">ADX (Trend Strength)</h3>
                <div className="flex items-center gap-4">
                  <span className="text-3xl font-bold">{ta.adx.value?.toFixed(1)}</span>
                  <span className={`px-3 py-1 rounded text-sm ${
                    ta.adx.value > 25 ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
                  }`}>
                    {ta.adx.value > 40 ? 'Strong Trend' :
                     ta.adx.value > 25 ? 'Trending' : 'Weak/No Trend'}
                  </span>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">+DI: </span>
                    <span className="font-medium text-green-600">{ta.adx.plus_di?.toFixed(1)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">-DI: </span>
                    <span className="font-medium text-red-600">{ta.adx.minus_di?.toFixed(1)}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Stochastic */}
            {ta.stochastic && (
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Stochastic Oscillator</h3>
                <div className="flex items-center gap-4">
                  <div>
                    <span className="text-sm text-gray-500">%K: </span>
                    <span className="text-2xl font-bold">{ta.stochastic.k?.toFixed(1)}</span>
                  </div>
                  <div>
                    <span className="text-sm text-gray-500">%D: </span>
                    <span className="text-2xl font-bold">{ta.stochastic.d?.toFixed(1)}</span>
                  </div>
                  <span className={`px-3 py-1 rounded text-sm ${
                    ta.stochastic.k > 80 ? 'bg-red-100 text-red-700' :
                    ta.stochastic.k < 20 ? 'bg-green-100 text-green-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {ta.stochastic.k > 80 ? 'Overbought' :
                     ta.stochastic.k < 20 ? 'Oversold' : 'Neutral'}
                  </span>
                </div>
              </div>
            )}

            {/* ATR */}
            {ta.atr && (
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">ATR (Volatility)</h3>
                <div className="flex items-center gap-4">
                  <span className="text-3xl font-bold">${ta.atr.value?.toFixed(2)}</span>
                  <span className="text-sm text-gray-500">
                    ({((ta.atr.value / historical.close?.[historical.close.length - 1]) * 100)?.toFixed(2)}% of price)
                  </span>
                </div>
              </div>
            )}

            {/* Bollinger Bands */}
            {ta.bollinger && (
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Bollinger Bands</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Upper Band:</span>
                    <span className="font-medium">${ta.bollinger.upper?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Middle (SMA 20):</span>
                    <span className="font-medium">${ta.bollinger.middle?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Lower Band:</span>
                    <span className="font-medium">${ta.bollinger.lower?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between pt-2 border-t">
                    <span className="text-gray-500">Band Width:</span>
                    <span className="font-medium">{ta.bollinger.bandwidth?.toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">%B:</span>
                    <span className="font-medium">{ta.bollinger.percent_b?.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Individual Signals */}
          {signals.signals && (
            <div className="card">
              <h3 className="text-lg font-semibold mb-4">Individual Indicator Signals</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(signals.signals).map(([indicator, signal]) => (
                  <div
                    key={indicator}
                    className={`p-3 rounded-lg text-center ${
                      signal === 'Buy' || signal === 'Strong Buy'
                        ? 'bg-green-50 border border-green-200'
                        : signal === 'Sell' || signal === 'Strong Sell'
                        ? 'bg-red-50 border border-red-200'
                        : 'bg-gray-50 border border-gray-200'
                    }`}
                  >
                    <p className="text-sm font-medium text-gray-600">{indicator}</p>
                    <p className={`font-bold ${
                      signal === 'Buy' || signal === 'Strong Buy'
                        ? 'text-green-600'
                        : signal === 'Sell' || signal === 'Strong Sell'
                        ? 'text-red-600'
                        : 'text-gray-600'
                    }`}>
                      {signal as string}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
