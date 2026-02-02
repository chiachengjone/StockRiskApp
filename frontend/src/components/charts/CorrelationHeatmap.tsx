import { useUIState } from '../../store/portfolioStore'
import clsx from 'clsx'

interface CorrelationHeatmapProps {
  matrix: number[][]
  labels: string[]
}

export default function CorrelationHeatmap({ matrix, labels }: CorrelationHeatmapProps) {
  const { darkMode } = useUIState()
  
  const getColor = (value: number) => {
    // Value ranges from -1 to 1
    // -1: red, 0: neutral, 1: emerald
    if (value >= 0) {
      const intensity = Math.round(value * 255)
      if (darkMode) {
        // Dark mode: blend with darker background
        return `rgb(${16 + (1 - value) * 30}, ${185 * value + 60}, ${129 * value + 60})`
      }
      return `rgb(${255 - intensity}, ${255 - intensity * 0.3}, ${255 - intensity * 0.5})`
    } else {
      const intensity = Math.abs(value)
      if (darkMode) {
        return `rgb(${239 * intensity + 60}, ${68 * intensity + 60}, ${68 * intensity + 60})`
      }
      const i = Math.round(intensity * 255)
      return `rgb(255, ${255 - i}, ${255 - i})`
    }
  }

  const getTextColor = (value: number) => {
    return Math.abs(value) > 0.4 ? 'white' : (darkMode ? '#d1d5db' : '#374151')
  }

  if (!matrix || !labels || matrix.length === 0) {
    return <div className={clsx('text-center py-8', darkMode ? 'text-gray-400' : 'text-gray-500')}>No data available</div>
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full">
        <thead>
          <tr>
            <th className="w-20"></th>
            {labels.map((label) => (
              <th
                key={label}
                className={clsx(
                  'px-3 py-2 text-xs font-medium text-center',
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                )}
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <td className={clsx(
                'px-3 py-2 text-xs font-medium text-right',
                darkMode ? 'text-gray-300' : 'text-gray-600'
              )}>
                {labels[i]}
              </td>
              {row.map((value, j) => (
                <td
                  key={`${i}-${j}`}
                  className="px-3 py-2 text-center text-xs font-medium"
                  style={{
                    backgroundColor: getColor(value),
                    color: getTextColor(value),
                  }}
                >
                  {value.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      
      {/* Legend */}
      <div className="flex items-center justify-center mt-4 gap-2">
        <span className={clsx('text-xs', darkMode ? 'text-gray-400' : 'text-gray-500')}>-1 (Negative)</span>
        <div className="flex h-4 rounded overflow-hidden">
          {[...Array(11)].map((_, i) => {
            const value = (i - 5) / 5 // -1 to 1
            return (
              <div
                key={i}
                className="w-6 h-full"
                style={{ backgroundColor: getColor(value) }}
              />
            )
          })}
        </div>
        <span className={clsx('text-xs', darkMode ? 'text-gray-400' : 'text-gray-500')}>+1 (Positive)</span>
      </div>
    </div>
  )
}
