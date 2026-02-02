import clsx from 'clsx'
import { useUIState } from '../store/portfolioStore'

interface MetricCardProps {
  label: string
  value: string | number
  positive?: boolean
  subtitle?: string
  icon?: React.ComponentType<{ className?: string }>
}

export default function MetricCard({ label, value, positive, subtitle, icon: Icon }: MetricCardProps) {
  const { darkMode } = useUIState()
  
  return (
    <div className={clsx(
      'metric-card transition-all duration-200 hover:shadow-lg',
      darkMode && 'bg-gray-800 border-gray-700'
    )}>
      <div className="flex items-start justify-between">
        <div>
          <p className={clsx('metric-label', darkMode && 'text-gray-400')}>{label}</p>
          <p className={clsx(
            'metric-value mt-1',
            positive === true ? 'text-emerald-500' : 
            positive === false ? 'text-red-500' : 
            darkMode ? 'text-white' : 'text-gray-900'
          )}>
            {value}
          </p>
          {subtitle && (
            <p className={clsx('text-xs mt-1', darkMode ? 'text-gray-500' : 'text-gray-400')}>
              {subtitle}
            </p>
          )}
        </div>
        {Icon && (
          <Icon className={clsx('h-5 w-5', darkMode ? 'text-gray-600' : 'text-gray-300')} />
        )}
      </div>
    </div>
  )
}
