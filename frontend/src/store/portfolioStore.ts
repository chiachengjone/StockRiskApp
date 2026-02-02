/**
 * Portfolio State Management with Zustand
 * ========================================
 * Global state for portfolio, settings, and UI with persistence
 */

import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

export interface StockHolding {
  ticker: string
  weight: number
  shares?: number
  currentPrice?: number
  value?: number
}

export interface Portfolio {
  id: string
  name: string
  holdings: StockHolding[]
  createdAt: string
  updatedAt: string
}

export interface DateRange {
  startDate: string
  endDate: string
}

export interface RiskSettings {
  confidenceLevel: number
  varHorizon: number
  monteCarloSims: number
  garchP: number
  garchQ: number
  evtThreshold: number
}

export interface UIState {
  activeTab: string
  sidebarOpen: boolean
  darkMode: boolean
}

interface PortfolioState {
  // Current analysis context
  currentTicker: string
  currentPortfolio: Portfolio | null
  dateRange: DateRange
  
  // Saved portfolios
  portfolios: Portfolio[]
  
  // Risk settings
  riskSettings: RiskSettings
  
  // UI state
  ui: UIState
  
  // Actions - Ticker
  setCurrentTicker: (ticker: string) => void
  
  // Actions - Portfolio
  setCurrentPortfolio: (portfolio: Portfolio | null) => void
  addPortfolio: (portfolio: Portfolio) => void
  updatePortfolio: (id: string, portfolio: Partial<Portfolio>) => void
  deletePortfolio: (id: string) => void
  
  // Actions - Date Range
  setDateRange: (range: DateRange) => void
  
  // Actions - Risk Settings
  setRiskSettings: (settings: Partial<RiskSettings>) => void
  
  // Actions - UI
  setActiveTab: (tab: string) => void
  toggleSidebar: () => void
  toggleDarkMode: () => void
}

// Default date range: 1 year ago to today
const getDefaultDateRange = (): DateRange => {
  const end = new Date()
  const start = new Date()
  start.setFullYear(start.getFullYear() - 1)
  
  return {
    startDate: start.toISOString().split('T')[0],
    endDate: end.toISOString().split('T')[0]
  }
}

export const usePortfolioStore = create<PortfolioState>()(
  persist(
    (set) => ({
      // Initial state
      currentTicker: 'AAPL',
      currentPortfolio: null,
      dateRange: getDefaultDateRange(),
      portfolios: [],
      
      riskSettings: {
        confidenceLevel: 0.95,
        varHorizon: 10,
        monteCarloSims: 10000,
        garchP: 1,
        garchQ: 1,
        evtThreshold: 0.95
      },
      
      ui: {
        activeTab: 'overview',
        sidebarOpen: true,
        darkMode: true
      },
      
      // Ticker actions
      setCurrentTicker: (ticker) => set({ currentTicker: ticker.toUpperCase() }),
      
      // Portfolio actions
      setCurrentPortfolio: (portfolio) => set({ currentPortfolio: portfolio }),
      
      addPortfolio: (portfolio) => set((state) => ({
        portfolios: [...state.portfolios, portfolio]
      })),
      
      updatePortfolio: (id, updates) => set((state) => ({
        portfolios: state.portfolios.map(p => 
          p.id === id ? { ...p, ...updates, updatedAt: new Date().toISOString() } : p
        ),
        currentPortfolio: state.currentPortfolio?.id === id 
          ? { ...state.currentPortfolio, ...updates, updatedAt: new Date().toISOString() }
          : state.currentPortfolio
      })),
      
      deletePortfolio: (id) => set((state) => ({
        portfolios: state.portfolios.filter(p => p.id !== id),
        currentPortfolio: state.currentPortfolio?.id === id ? null : state.currentPortfolio
      })),
      
      // Date range actions
      setDateRange: (range) => set({ dateRange: range }),
      
      // Risk settings actions
      setRiskSettings: (settings) => set((state) => ({
        riskSettings: { ...state.riskSettings, ...settings }
      })),
      
      // UI actions
      setActiveTab: (tab) => set((state) => ({
        ui: { ...state.ui, activeTab: tab }
      })),
      
      toggleSidebar: () => set((state) => ({
        ui: { ...state.ui, sidebarOpen: !state.ui.sidebarOpen }
      })),
      
      toggleDarkMode: () => set((state) => ({
        ui: { ...state.ui, darkMode: !state.ui.darkMode }
      }))
    }),
    {
      name: 'stock-risk-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        currentTicker: state.currentTicker,
        portfolios: state.portfolios,
        dateRange: state.dateRange,
        riskSettings: state.riskSettings,
        ui: state.ui
      })
    }
  )
)

// Selector hooks for performance optimization
export const useCurrentTicker = () => usePortfolioStore((state) => state.currentTicker)
export const useDateRange = () => usePortfolioStore((state) => state.dateRange)
export const useRiskSettings = () => usePortfolioStore((state) => state.riskSettings)
export const useUIState = () => usePortfolioStore((state) => state.ui)
export const usePortfolios = () => usePortfolioStore((state) => state.portfolios)
