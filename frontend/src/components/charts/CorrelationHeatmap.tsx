interface CorrelationHeatmapProps {
  matrix: number[][]
  labels: string[]
}

export default function CorrelationHeatmap({ matrix, labels }: CorrelationHeatmapProps) {
  const getColor = (value: number) => {
    // Value ranges from -1 to 1
    // -1: dark red, 0: white, 1: dark blue
    if (value >= 0) {
      const intensity = Math.round(value * 255)
      return `rgb(${255 - intensity}, ${255 - intensity}, 255)`
    } else {
      const intensity = Math.round(Math.abs(value) * 255)
      return `rgb(255, ${255 - intensity}, ${255 - intensity})`
    }
  }

  const getTextColor = (value: number) => {
    return Math.abs(value) > 0.5 ? 'white' : 'black'
  }

  if (!matrix || !labels || matrix.length === 0) {
    return <div className="text-center text-gray-500 py-8">No data available</div>
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
                className="px-3 py-2 text-xs font-medium text-gray-600 text-center"
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <td className="px-3 py-2 text-xs font-medium text-gray-600 text-right">
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
        <span className="text-xs text-gray-500">-1 (Negative)</span>
        <div className="flex h-4">
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
        <span className="text-xs text-gray-500">+1 (Positive)</span>
      </div>
    </div>
  )
}
