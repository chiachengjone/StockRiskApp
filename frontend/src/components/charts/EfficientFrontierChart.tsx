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
  const data = useMemo(() => 
    returns.map((r, i) => ({
      risk: volatilities[i] * 100,
      return: r * 100,
    })),
  [returns, volatilities])

  if (!returns || !volatilities || returns.length === 0) {
    return <div className="text-center text-gray-500 py-8">No data available</div>
  }

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="risk"
          name="Risk"
          unit="%"
          domain={['dataMin - 1', 'dataMax + 1']}
          label={{ value: 'Risk (Volatility %)', position: 'bottom', offset: 0 }}
        />
        <YAxis
          type="number"
          dataKey="return"
          name="Return"
          unit="%"
          domain={['dataMin - 1', 'dataMax + 1']}
          label={{ value: 'Return %', angle: -90, position: 'insideLeft' }}
        />
        <Tooltip
          formatter={(value: number, name: string) => [`${value.toFixed(2)}%`, name]}
          cursor={{ strokeDasharray: '3 3' }}
        />
        <Scatter
          name="Efficient Frontier"
          data={data}
          fill="#3B82F6"
          line={{ stroke: '#3B82F6', strokeWidth: 2 }}
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
