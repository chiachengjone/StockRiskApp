/**
 * Reports Page
 * ============
 * Generate and download PDF reports
 */

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  DocumentTextIcon, 
  ArrowDownTrayIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline'
import clsx from 'clsx'
import toast from 'react-hot-toast'

import { useCurrentTicker, useDateRange, useUIState, usePortfolios } from '../store/portfolioStore'
import { useReportAvailability, useDownloadStockReport, useDownloadPortfolioReport } from '../services/hooks'
import LoadingSpinner from '../components/LoadingSpinner'

export default function Reports() {
  const { darkMode } = useUIState()
  const currentTicker = useCurrentTicker()
  const dateRange = useDateRange()
  const portfolios = usePortfolios()
  
  const [selectedPortfolio, setSelectedPortfolio] = useState<string>('')
  const [reportType, setReportType] = useState<'single' | 'portfolio'>('single')

  const { data: availability, isLoading: availLoading } = useReportAvailability()
  const downloadStockReport = useDownloadStockReport()
  const downloadPortfolioReport = useDownloadPortfolioReport()

  const handleDownloadSingleStock = () => {
    toast.promise(
      downloadStockReport.mutateAsync({
        ticker: currentTicker,
        startDate: dateRange.startDate,
        endDate: dateRange.endDate,
      }),
      {
        loading: `Generating ${currentTicker} report...`,
        success: `${currentTicker} report downloaded!`,
        error: 'Failed to generate report',
      }
    )
  }

  const handleDownloadPortfolio = () => {
    const portfolio = portfolios.find(p => p.id === selectedPortfolio)
    if (!portfolio) {
      toast.error('Please select a portfolio')
      return
    }

    // Extract tickers and weights from holdings
    const tickers = portfolio.holdings.map(h => h.ticker)
    const weights = portfolio.holdings.reduce((acc, h) => ({
      ...acc,
      [h.ticker]: h.weight
    }), {} as Record<string, number>)

    toast.promise(
      downloadPortfolioReport.mutateAsync({
        name: portfolio.name,
        tickers,
        weights,
        startDate: dateRange.startDate,
        endDate: dateRange.endDate,
      }),
      {
        loading: `Generating ${portfolio.name} report...`,
        success: `${portfolio.name} report downloaded!`,
        error: 'Failed to generate report',
      }
    )
  }

  const isAvailable = availability?.pdf_available || false

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className={clsx('text-3xl font-bold', darkMode ? 'text-white' : 'text-gray-900')}>
          Reports
        </h1>
        <p className={clsx('mt-1', darkMode ? 'text-gray-400' : 'text-gray-500')}>
          Generate comprehensive PDF reports for stocks and portfolios
        </p>
      </div>

      {/* Availability Status */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}
      >
        <div className="flex items-center gap-4">
          {availLoading ? (
            <LoadingSpinner size="sm" />
          ) : isAvailable ? (
            <>
              <CheckCircleIcon className="h-8 w-8 text-emerald-500" />
              <div>
                <h3 className={clsx('font-semibold', darkMode ? 'text-white' : 'text-gray-900')}>
                  PDF Reports Available
                </h3>
                <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>
                  ReportLab is installed. You can generate PDF reports.
                </p>
              </div>
            </>
          ) : (
            <>
              <ExclamationCircleIcon className="h-8 w-8 text-amber-500" />
              <div>
                <h3 className={clsx('font-semibold', darkMode ? 'text-white' : 'text-gray-900')}>
                  PDF Reports Unavailable
                </h3>
                <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>
                  Install ReportLab: <code className="bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded">pip install reportlab</code>
                </p>
              </div>
            </>
          )}
        </div>
      </motion.div>

      {/* Report Type Selection */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Single Stock Report */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className={clsx(
            'card cursor-pointer transition-all',
            reportType === 'single' && 'ring-2 ring-emerald-500',
            darkMode && 'bg-gray-800 border-gray-700'
          )}
          onClick={() => setReportType('single')}
        >
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0">
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                <DocumentTextIcon className="h-6 w-6 text-white" />
              </div>
            </div>
            <div className="flex-1">
              <h3 className={clsx('text-lg font-semibold', darkMode ? 'text-white' : 'text-gray-900')}>
                Single Stock Report
              </h3>
              <p className={clsx('text-sm mt-1', darkMode ? 'text-gray-400' : 'text-gray-500')}>
                Comprehensive risk analysis for {currentTicker}
              </p>
              <ul className={clsx('text-sm mt-3 space-y-1', darkMode ? 'text-gray-400' : 'text-gray-600')}>
                <li>• VaR & CVaR Analysis</li>
                <li>• GARCH Volatility Model</li>
                <li>• Monte Carlo Simulation</li>
                <li>• Factor Exposures</li>
                <li>• ML Predictions</li>
              </ul>
            </div>
          </div>
          
          {reportType === 'single' && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>
                    Ticker: <span className="font-mono font-bold">{currentTicker}</span>
                  </p>
                  <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>
                    Period: {dateRange.startDate} to {dateRange.endDate}
                  </p>
                </div>
                <button
                  onClick={handleDownloadSingleStock}
                  disabled={!isAvailable || downloadStockReport.isPending}
                  className={clsx(
                    'btn-primary flex items-center gap-2',
                    (!isAvailable || downloadStockReport.isPending) && 'opacity-50 cursor-not-allowed'
                  )}
                >
                  {downloadStockReport.isPending ? (
                    <LoadingSpinner size="sm" />
                  ) : (
                    <ArrowDownTrayIcon className="h-5 w-5" />
                  )}
                  Download PDF
                </button>
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Portfolio Report */}
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className={clsx(
            'card cursor-pointer transition-all',
            reportType === 'portfolio' && 'ring-2 ring-emerald-500',
            darkMode && 'bg-gray-800 border-gray-700'
          )}
          onClick={() => setReportType('portfolio')}
        >
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0">
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
                <DocumentTextIcon className="h-6 w-6 text-white" />
              </div>
            </div>
            <div className="flex-1">
              <h3 className={clsx('text-lg font-semibold', darkMode ? 'text-white' : 'text-gray-900')}>
                Portfolio Report
              </h3>
              <p className={clsx('text-sm mt-1', darkMode ? 'text-gray-400' : 'text-gray-500')}>
                Multi-asset risk and allocation analysis
              </p>
              <ul className={clsx('text-sm mt-3 space-y-1', darkMode ? 'text-gray-400' : 'text-gray-600')}>
                <li>• Portfolio VaR</li>
                <li>• Correlation Analysis</li>
                <li>• Risk Contribution</li>
                <li>• Efficient Frontier</li>
                <li>• Diversification Score</li>
              </ul>
            </div>
          </div>
          
          {reportType === 'portfolio' && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700"
            >
              {portfolios.length > 0 ? (
                <div className="space-y-4">
                  <div>
                    <label className="label">Select Portfolio</label>
                    <select
                      value={selectedPortfolio}
                      onChange={(e) => setSelectedPortfolio(e.target.value)}
                      className={clsx('input', darkMode && 'bg-gray-700 border-gray-600 text-white')}
                    >
                      <option value="">Choose a portfolio...</option>
                      {portfolios.map(p => (
                        <option key={p.id} value={p.id}>{p.name} ({p.holdings.length} assets)</option>
                      ))}
                    </select>
                  </div>
                  <div className="flex justify-end">
                    <button
                      onClick={handleDownloadPortfolio}
                      disabled={!isAvailable || !selectedPortfolio || downloadPortfolioReport.isPending}
                      className={clsx(
                        'btn-primary flex items-center gap-2',
                        (!isAvailable || !selectedPortfolio || downloadPortfolioReport.isPending) && 'opacity-50 cursor-not-allowed'
                      )}
                    >
                      {downloadPortfolioReport.isPending ? (
                        <LoadingSpinner size="sm" />
                      ) : (
                        <ArrowDownTrayIcon className="h-5 w-5" />
                      )}
                      Download PDF
                    </button>
                  </div>
                </div>
              ) : (
                <p className={clsx('text-sm text-center py-4', darkMode ? 'text-gray-500' : 'text-gray-400')}>
                  No portfolios saved. Create one in the Portfolio tab.
                </p>
              )}
            </motion.div>
          )}
        </motion.div>
      </div>

      {/* Report Preview Info */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className={clsx('card', darkMode && 'bg-gray-800 border-gray-700')}
      >
        <h3 className={clsx('text-lg font-semibold mb-4', darkMode ? 'text-white' : 'text-gray-900')}>
          Report Contents
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div>
            <h4 className={clsx('font-medium mb-2', darkMode ? 'text-gray-300' : 'text-gray-700')}>
              Executive Summary
            </h4>
            <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>
              Key metrics, risk ratings, and investment recommendations at a glance.
            </p>
          </div>
          <div>
            <h4 className={clsx('font-medium mb-2', darkMode ? 'text-gray-300' : 'text-gray-700')}>
              Risk Analysis
            </h4>
            <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>
              VaR, CVaR, GARCH volatility, EVT tail risk, and Monte Carlo simulations.
            </p>
          </div>
          <div>
            <h4 className={clsx('font-medium mb-2', darkMode ? 'text-gray-300' : 'text-gray-700')}>
              Charts & Visualizations
            </h4>
            <p className={clsx('text-sm', darkMode ? 'text-gray-400' : 'text-gray-500')}>
              Price history, return distributions, correlation matrices, and more.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
