import { useMemo } from 'react'
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from 'recharts'
import { useUIState } from '../../store/portfolioStore'

interface EfficientFrontierChartProps {
  returns: number[]
  volatilities: number[]
  maxSharpePoint: { x: number; y: number }
  minVolPoint: { x: number; y: number }
}

export default function EfficientFrontierChart({
  returns,
  volatilities,
  maxSharpePoint,
  minVolPoint,
}: EfficientFrontierChartProps) {
  const { darkMode } = useUIState()
  
  const data = useMemo(() => 
    returns.map((r, i) => ({
      risk: volatilities[i] * 100,
      return: r * 100,
    })),
  [returns, volatilities])

  if (!returns || !volatilities || returns.length === 0) {
    return <div className={`text-center py-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>No data available</div>
  }

  const axisColor = darkMode ? '#9ca3af' : '#6b7280'
  const gridColor = darkMode ? '#374151' : '#e5e7eb'

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
        <XAxis
          type="number"
          dataKey="risk"
          name="Risk"
          unit="%"
          domain={['dataMin - 1', 'dataMax + 1']}
          label={{ value: 'Risk (Volatility %)', position: 'bottom', offset: 0, fill: axisColor }}
          tick={{ fill: axisColor }}
          stroke={gridColor}
        />
        <YAxis
          type="number"
          dataKey="return"
          name="Return"
          unit="%"
          domain={['dataMin - 1', 'dataMax + 1']}
          label={{ value: 'Return %', angle: -90, position: 'insideLeft', fill: axisColor }}
          tick={{ fill: axisColor }}
          stroke={gridColor}
        />
        <Tooltip
          formatter={(value: number, name: string) => [`${value.toFixed(2)}%`, name]}
          cursor={{ strokeDasharray: '3 3' }}
          contentStyle={{
            backgroundColor: darkMode ? '#1f2937' : '#ffffff',
            borderColor: darkMode ? '#374151' : '#e5e7eb',
            color: darkMode ? '#f3f4f6' : '#111827',
          }}
        />
        <Scatter
          name="Efficient Frontier"
          data={data}
          fill="#10b981"
          line={{ stroke: '#10b981', strokeWidth: 2 }}
        />
        {/* Max Sharpe Point */}
        <ReferenceDot
          x={maxSharpePoint.x * 100}
          y={maxSharpePoint.y * 100}
          r={8}
          fill="#10B981"
          stroke="#fff"
          strokeWidth={2}
        />
        {/* Min Vol Point */}
        <ReferenceDot
          x={minVolPoint.x * 100}
          y={minVolPoint.y * 100}
          r={8}
          fill="#F59E0B"
          stroke="#fff"
          strokeWidth={2}
        />
      </ScatterChart>
    </ResponsiveContainer>
  )
}
