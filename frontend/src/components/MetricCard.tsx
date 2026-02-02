interface MetricCardProps {
  label: string
  value: string | number
  positive?: boolean
  subtitle?: string
}

export default function MetricCard({ label, value, positive, subtitle }: MetricCardProps) {
  return (
    <div className="metric-card">
      <p className="metric-label">{label}</p>
      <p className={`metric-value ${
        positive === true ? 'text-green-600' : 
        positive === false ? 'text-red-600' : 
        'text-gray-900'
      }`}>
        {value}
      </p>
      {subtitle && (
        <p className="text-xs text-gray-400 mt-1">{subtitle}</p>
      )}
    </div>
  )
}
