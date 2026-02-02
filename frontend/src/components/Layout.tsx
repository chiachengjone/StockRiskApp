import { Outlet, NavLink } from 'react-router-dom'
import {
  ChartBarIcon,
  ChartPieIcon,
  PresentationChartLineIcon,
  CubeIcon,
  SunIcon,
  MoonIcon,
  DocumentTextIcon,
} from '@heroicons/react/24/outline'
import { useUIState, usePortfolioStore } from '../store/portfolioStore'
import clsx from 'clsx'

const navigation = [
  { name: 'Dashboard', href: '/', icon: ChartBarIcon },
  { name: 'Single Stock', href: '/single-stock', icon: PresentationChartLineIcon },
  { name: 'Portfolio', href: '/portfolio', icon: ChartPieIcon },
  { name: 'Technical Analysis', href: '/technical-analysis', icon: CubeIcon },
  { name: 'Reports', href: '/reports', icon: DocumentTextIcon },
]

export default function Layout() {
  const { darkMode } = useUIState()
  const toggleDarkMode = usePortfolioStore((state) => state.toggleDarkMode)

  return (
    <div className={clsx('min-h-screen transition-colors duration-200', darkMode ? 'dark bg-gray-900' : 'bg-gray-50')}>
      {/* Header */}
      <header className={clsx(
        'shadow-lg border-b transition-colors duration-200',
        darkMode 
          ? 'bg-gray-800 border-gray-700' 
          : 'bg-white border-gray-200'
      )}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="flex items-center justify-center h-10 w-10 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600">
                <ChartBarIcon className="h-6 w-6 text-white" />
              </div>
              <span className={clsx(
                'ml-3 text-xl font-bold',
                darkMode ? 'text-white' : 'text-gray-900'
              )}>
                Stock Risk Modelling
              </span>
              <span className="ml-2 text-xs bg-gradient-to-r from-emerald-500 to-teal-600 text-white px-2 py-1 rounded-full font-medium">
                v4.3
              </span>
            </div>
            <nav className="flex items-center space-x-1">
              {navigation.map((item) => (
                <NavLink
                  key={item.name}
                  to={item.href}
                  className={({ isActive }) =>
                    clsx(
                      'flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200',
                      isActive
                        ? darkMode 
                          ? 'bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/50'
                          : 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200'
                        : darkMode
                          ? 'text-gray-400 hover:bg-gray-700 hover:text-gray-200'
                          : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                    )
                  }
                >
                  <item.icon className="h-5 w-5 mr-1.5" />
                  {item.name}
                </NavLink>
              ))}
              <div className="ml-2 pl-2 border-l border-gray-600">
                <button
                  onClick={toggleDarkMode}
                  className={clsx(
                    'p-2 rounded-lg transition-colors',
                    darkMode 
                      ? 'text-yellow-400 hover:bg-gray-700' 
                      : 'text-gray-600 hover:bg-gray-100'
                  )}
                  title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                  {darkMode ? <SunIcon className="h-5 w-5" /> : <MoonIcon className="h-5 w-5" />}
                </button>
              </div>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className={clsx(
        'max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 min-h-[calc(100vh-8rem)]',
        darkMode ? 'text-gray-100' : 'text-gray-900'
      )}>
        <Outlet />
      </main>

      {/* Footer */}
      <footer className={clsx(
        'border-t mt-auto transition-colors duration-200',
        darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
      )}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <p className={clsx(
            'text-center text-sm',
            darkMode ? 'text-gray-400' : 'text-gray-500'
          )}>
            Stock Risk Modelling App v4.3 — FastAPI + React + TypeScript
          </p>
        </div>
      </footer>
    </div>
  )
}
