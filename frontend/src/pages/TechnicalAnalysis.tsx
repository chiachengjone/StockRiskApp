import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import clsx from 'clsx'
import { taApi, dataApi } from '@/services/api'
import { useUIState } from '@/store/portfolioStore'
import LoadingSpinner from '@/components/LoadingSpinner'
import MetricCard from '@/components/MetricCard'
import CandlestickChart from '@/components/charts/CandlestickChart'
import TAIndicatorChart from '@/components/charts/TAIndicatorChart'

const SIGNAL_COLORS: Record<string, { light: string; dark: string }> = {
  'Strong Buy': { light: 'bg-green-600 text-white', dark: 'bg-green-600 text-white' },
  'Buy': { light: 'bg-green-400 text-white', dark: 'bg-green-500 text-white' },
  'Neutral': { light: 'bg-gray-400 text-white', dark: 'bg-gray-500 text-white' },
  'Sell': { light: 'bg-red-400 text-white', dark: 'bg-red-500 text-white' },
  'Strong Sell': { light: 'bg-red-600 text-white', dark: 'bg-red-600 text-white' },
}

export default function TechnicalAnalysis() {
  const { darkMode } = useUIState()
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
  const { data: taData, isLoading: taLoading } = useQuery({
    queryKey: ['ta-full', ticker, startDate, endDate],
    queryFn: () => taApi.getFull(ticker, startDate, endDate),
    enabled: !!ticker
  })

  // Fetch signals
  const { data: signalsData, isLoading: signalsLoading } = useQuery({
    queryKey: ['ta-signals', ticker],
    queryFn: () => taApi.getSignals(ticker),
    enabled: !!ticker
  })

  // Fetch historical data for candlestick
  const { data: historicalData, isLoading: histLoading } = useQuery({
    queryKey: ['historical', ticker, startDate, endDate],
    queryFn: () => dataApi.getHistorical(ticker, startDate, endDate),
    enabled: !!ticker
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const formData = new FormData(e.target as HTMLFormElement)
    setTicker((formData.get('ticker') as string).toUpperCase())
  }

  const isLoading = taLoading || signalsLoading || histLoading
  type AnyData = Record<string, any>
  const ta = (taData as AnyData)?.data || {} as AnyData
  const signals = (signalsData as AnyData)?.data || {} as AnyData
  const historical = (historicalData as AnyData)?.data || {} as AnyData

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
          )}>Technical Analysis</h1>
          <p className={clsx(darkMode ? 'text-gray-400' : 'text-gray-500')}>
            Indicators and signals for {ticker}
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          {/* Period selector */}
          <div className="flex gap-1">
            {['1m', '3m', '6m', '1y', '2y'].map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={clsx(
                  'px-3 py-1 text-sm rounded transition-all',
                  period === p
                    ? 'bg-emerald-600 text-white'
                    : darkMode 
                      ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                )}
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
              className={clsx(
                'input w-28',
                darkMode && 'bg-gray-800 border-gray-700 text-white placeholder-gray-400'
              )}
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
          <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
            <div className="flex items-center justify-between">
              <div>
                <h2 className={clsx(
                  'text-lg font-semibold',
                  darkMode ? 'text-white' : 'text-gray-900'
                )}>Overall Signal</h2>
                <p className={clsx(darkMode ? 'text-gray-400' : 'text-gray-500')}>
                  Based on multiple technical indicators
                </p>
              </div>
              <div className={clsx(
                'px-6 py-3 rounded-lg font-bold text-lg',
                SIGNAL_COLORS[signals.overall_signal]?.[darkMode ? 'dark' : 'light'] || 'bg-gray-200'
              )}>
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
            <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
              <h3 className={clsx(
                'text-lg font-semibold mb-4',
                darkMode ? 'text-white' : 'text-gray-900'
              )}>Price with Bollinger Bands</h3>
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
            <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
              <div className="flex items-center justify-between mb-4">
                <h3 className={clsx(
                  'text-lg font-semibold',
                  darkMode ? 'text-white' : 'text-gray-900'
                )}>RSI (14)</h3>
                <div className="flex items-center gap-4">
                  <span className={clsx(
                    'text-2xl font-bold',
                    darkMode ? 'text-white' : 'text-gray-900'
                  )}>
                    {ta.rsi.value?.toFixed(1)}
                  </span>
                  <span className={clsx(
                    'px-3 py-1 rounded text-sm',
                    ta.rsi.value > 70 
                      ? (darkMode ? 'bg-red-900/50 text-red-300' : 'bg-red-100 text-red-700')
                      : ta.rsi.value < 30 
                        ? (darkMode ? 'bg-green-900/50 text-green-300' : 'bg-green-100 text-green-700')
                        : (darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700')
                  )}>
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
                  color="#10b981"
                  overbought={70}
                  oversold={30}
                />
              )}
            </div>
          )}

          {/* MACD */}
          {ta.macd && (
            <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
              <div className="flex items-center justify-between mb-4">
                <h3 className={clsx(
                  'text-lg font-semibold',
                  darkMode ? 'text-white' : 'text-gray-900'
                )}>MACD</h3>
                <div className="flex items-center gap-4">
                  <div>
                    <span className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>MACD: </span>
                    <span className={clsx('font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                      {ta.macd.macd?.toFixed(2)}
                    </span>
                  </div>
                  <div>
                    <span className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>Signal: </span>
                    <span className={clsx('font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                      {ta.macd.signal?.toFixed(2)}
                    </span>
                  </div>
                  <span className={clsx(
                    'px-3 py-1 rounded text-sm',
                    ta.macd.histogram > 0 
                      ? (darkMode ? 'bg-green-900/50 text-green-300' : 'bg-green-100 text-green-700')
                      : (darkMode ? 'bg-red-900/50 text-red-300' : 'bg-red-100 text-red-700')
                  )}>
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
                  color="#10b981"
                  color2="#EF4444"
                  histogram={ta.macd.histogram_history}
                />
              )}
            </div>
          )}

          {/* Moving Averages */}
          <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
            <h3 className={clsx(
              'text-lg font-semibold mb-4',
              darkMode ? 'text-white' : 'text-gray-900'
            )}>Moving Averages</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {ta.sma_20 && (
                <div className={clsx(
                  'p-4 rounded-lg',
                  darkMode ? 'bg-gray-700' : 'bg-gray-50'
                )}>
                  <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>SMA 20</p>
                  <p className={clsx(
                    'text-xl font-bold',
                    darkMode ? 'text-white' : 'text-gray-900'
                  )}>${ta.sma_20.value?.toFixed(2)}</p>
                  <p className={clsx('text-sm',
                    historical.close?.[historical.close.length - 1] > ta.sma_20.value
                      ? 'text-green-500' : 'text-red-500'
                  )}>
                    Price {historical.close?.[historical.close.length - 1] > ta.sma_20.value ? 'Above' : 'Below'}
                  </p>
                </div>
              )}
              {ta.sma_50 && (
                <div className={clsx(
                  'p-4 rounded-lg',
                  darkMode ? 'bg-gray-700' : 'bg-gray-50'
                )}>
                  <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>SMA 50</p>
                  <p className={clsx(
                    'text-xl font-bold',
                    darkMode ? 'text-white' : 'text-gray-900'
                  )}>${ta.sma_50.value?.toFixed(2)}</p>
                  <p className={clsx('text-sm',
                    historical.close?.[historical.close.length - 1] > ta.sma_50.value
                      ? 'text-green-500' : 'text-red-500'
                  )}>
                    Price {historical.close?.[historical.close.length - 1] > ta.sma_50.value ? 'Above' : 'Below'}
                  </p>
                </div>
              )}
              {ta.ema_12 && (
                <div className={clsx(
                  'p-4 rounded-lg',
                  darkMode ? 'bg-gray-700' : 'bg-gray-50'
                )}>
                  <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>EMA 12</p>
                  <p className={clsx(
                    'text-xl font-bold',
                    darkMode ? 'text-white' : 'text-gray-900'
                  )}>${ta.ema_12.value?.toFixed(2)}</p>
                </div>
              )}
              {ta.ema_26 && (
                <div className={clsx(
                  'p-4 rounded-lg',
                  darkMode ? 'bg-gray-700' : 'bg-gray-50'
                )}>
                  <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>EMA 26</p>
                  <p className={clsx(
                    'text-xl font-bold',
                    darkMode ? 'text-white' : 'text-gray-900'
                  )}>${ta.ema_26.value?.toFixed(2)}</p>
                </div>
              )}
            </div>
          </div>

          {/* Additional Indicators */}
          <div className="grid grid-cols-2 gap-6">
            {/* ADX */}
            {ta.adx && (
              <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                <h3 className={clsx(
                  'text-lg font-semibold mb-4',
                  darkMode ? 'text-white' : 'text-gray-900'
                )}>ADX (Trend Strength)</h3>
                <div className="flex items-center gap-4">
                  <span className={clsx(
                    'text-3xl font-bold',
                    darkMode ? 'text-white' : 'text-gray-900'
                  )}>{ta.adx.value?.toFixed(1)}</span>
                  <span className={clsx(
                    'px-3 py-1 rounded text-sm',
                    ta.adx.value > 25 
                      ? (darkMode ? 'bg-green-900/50 text-green-300' : 'bg-green-100 text-green-700')
                      : (darkMode ? 'bg-yellow-900/50 text-yellow-300' : 'bg-yellow-100 text-yellow-700')
                  )}>
                    {ta.adx.value > 40 ? 'Strong Trend' :
                     ta.adx.value > 25 ? 'Trending' : 'Weak/No Trend'}
                  </span>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className={clsx(darkMode ? 'text-gray-400' : 'text-gray-500')}>+DI: </span>
                    <span className="font-medium text-green-500">{ta.adx.plus_di?.toFixed(1)}</span>
                  </div>
                  <div>
                    <span className={clsx(darkMode ? 'text-gray-400' : 'text-gray-500')}>-DI: </span>
                    <span className="font-medium text-red-500">{ta.adx.minus_di?.toFixed(1)}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Stochastic */}
            {ta.stochastic && (
              <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                <h3 className={clsx(
                  'text-lg font-semibold mb-4',
                  darkMode ? 'text-white' : 'text-gray-900'
                )}>Stochastic Oscillator</h3>
                <div className="flex items-center gap-4">
                  <div>
                    <span className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>%K: </span>
                    <span className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                      {ta.stochastic.k?.toFixed(1)}
                    </span>
                  </div>
                  <div>
                    <span className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>%D: </span>
                    <span className={clsx('text-2xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
                      {ta.stochastic.d?.toFixed(1)}
                    </span>
                  </div>
                  <span className={clsx(
                    'px-3 py-1 rounded text-sm',
                    ta.stochastic.k > 80 
                      ? (darkMode ? 'bg-red-900/50 text-red-300' : 'bg-red-100 text-red-700')
                      : ta.stochastic.k < 20 
                        ? (darkMode ? 'bg-green-900/50 text-green-300' : 'bg-green-100 text-green-700')
                        : (darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700')
                  )}>
                    {ta.stochastic.k > 80 ? 'Overbought' :
                     ta.stochastic.k < 20 ? 'Oversold' : 'Neutral'}
                  </span>
                </div>
              </div>
            )}

            {/* ATR */}
            {ta.atr && (
              <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                <h3 className={clsx(
                  'text-lg font-semibold mb-4',
                  darkMode ? 'text-white' : 'text-gray-900'
                )}>ATR (Volatility)</h3>
                <div className="flex items-center gap-4">
                  <span className={clsx(
                    'text-3xl font-bold',
                    darkMode ? 'text-white' : 'text-gray-900'
                  )}>${ta.atr.value?.toFixed(2)}</span>
                  <span className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>
                    ({((ta.atr.value / historical.close?.[historical.close.length - 1]) * 100)?.toFixed(2)}% of price)
                  </span>
                </div>
              </div>
            )}

            {/* Bollinger Bands */}
            {ta.bollinger && (
              <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
                <h3 className={clsx(
                  'text-lg font-semibold mb-4',
                  darkMode ? 'text-white' : 'text-gray-900'
                )}>Bollinger Bands</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className={clsx(darkMode ? 'text-gray-400' : 'text-gray-500')}>Upper Band:</span>
                    <span className={clsx('font-medium', darkMode ? 'text-white' : 'text-gray-900')}>
                      ${ta.bollinger.upper?.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className={clsx(darkMode ? 'text-gray-400' : 'text-gray-500')}>Middle (SMA 20):</span>
                    <span className={clsx('font-medium', darkMode ? 'text-white' : 'text-gray-900')}>
                      ${ta.bollinger.middle?.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className={clsx(darkMode ? 'text-gray-400' : 'text-gray-500')}>Lower Band:</span>
                    <span className={clsx('font-medium', darkMode ? 'text-white' : 'text-gray-900')}>
                      ${ta.bollinger.lower?.toFixed(2)}
                    </span>
                  </div>
                  <div className={clsx(
                    'flex justify-between pt-2 border-t',
                    darkMode ? 'border-gray-700' : 'border-gray-200'
                  )}>
                    <span className={clsx(darkMode ? 'text-gray-400' : 'text-gray-500')}>Band Width:</span>
                    <span className={clsx('font-medium', darkMode ? 'text-white' : 'text-gray-900')}>
                      {ta.bollinger.bandwidth?.toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className={clsx(darkMode ? 'text-gray-400' : 'text-gray-500')}>%B:</span>
                    <span className={clsx('font-medium', darkMode ? 'text-white' : 'text-gray-900')}>
                      {ta.bollinger.percent_b?.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Individual Signals */}
          {signals.signals && (
            <div className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}>
              <h3 className={clsx(
                'text-lg font-semibold mb-4',
                darkMode ? 'text-white' : 'text-gray-900'
              )}>Individual Indicator Signals</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(signals.signals).map(([indicator, signal]) => (
                  <div
                    key={indicator}
                    className={clsx(
                      'p-3 rounded-lg text-center border',
                      signal === 'Buy' || signal === 'Strong Buy'
                        ? (darkMode ? 'bg-green-900/30 border-green-700' : 'bg-green-50 border-green-200')
                        : signal === 'Sell' || signal === 'Strong Sell'
                        ? (darkMode ? 'bg-red-900/30 border-red-700' : 'bg-red-50 border-red-200')
                        : (darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200')
                    )}
                  >
                    <p className={clsx(
                      'text-sm font-medium',
                      darkMode ? 'text-gray-300' : 'text-gray-600'
                    )}>{indicator}</p>
                    <p className={clsx(
                      'font-bold',
                      signal === 'Buy' || signal === 'Strong Buy'
                        ? 'text-green-500'
                        : signal === 'Sell' || signal === 'Strong Sell'
                        ? 'text-red-500'
                        : (darkMode ? 'text-gray-300' : 'text-gray-600')
                    )}>
                      {signal as string}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </motion.div>
  )
}
