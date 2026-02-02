import { useMemo } from 'react'
import { PieChart as RechartsPie, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'

interface PieChartProps {
  data: Array<{ name: string; value: number }>
}

const COLORS = [
  '#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6',
  '#EC4899', '#06B6D4', '#84CC16', '#F97316', '#6366F1',
]

export default function PieChart({ data }: PieChartProps) {
  const formattedData = useMemo(() => 
    data.filter((d) => d.value > 0.1).map((d, i) => ({
      ...d,
      color: COLORS[i % COLORS.length],
    })),
  [data])

  if (!data || data.length === 0) {
    return <div className="text-center text-gray-500 py-8">No data available</div>
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RechartsPie>
        <Pie
          data={formattedData}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
          outerRadius={100}
          fill="#8884d8"
          dataKey="value"
        >
          {formattedData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip 
          formatter={(value: number) => `${value.toFixed(2)}%`}
        />
        <Legend />
      </RechartsPie>
    </ResponsiveContainer>
  )
}
