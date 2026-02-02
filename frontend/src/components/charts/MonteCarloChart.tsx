import { useMemo } from 'react'
import Chart from 'react-apexcharts'
import { ApexOptions } from 'apexcharts'

interface MonteCarloChartProps {
  percentiles: Record<string, number>
}

export default function MonteCarloChart({ percentiles }: MonteCarloChartProps) {
  const data = useMemo(() => {
    if (!percentiles) return []
    
    return Object.entries(percentiles)
      .map(([pct, value]) => ({
        x: parseInt(pct),
        y: (value as number) * 100,
      }))
      .sort((a, b) => a.x - b.x)
  }, [percentiles])

  const options: ApexOptions = useMemo(() => ({
    chart: {
      type: 'area',
      height: 300,
      toolbar: { show: false },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      curve: 'smooth',
      width: 2,
    },
    fill: {
      type: 'gradient',
      gradient: {
        shadeIntensity: 1,
        opacityFrom: 0.5,
        opacityTo: 0.1,
        stops: [0, 100],
      },
    },
    xaxis: {
      type: 'numeric',
      title: {
        text: 'Percentile',
      },
      labels: {
        formatter: (val) => `${val}%`,
      },
    },
    yaxis: {
      title: {
        text: 'Return (%)',
      },
      labels: {
        formatter: (val) => `${val.toFixed(1)}%`,
      },
    },
    colors: ['#8B5CF6'],
    annotations: {
      yaxis: [
        {
          y: 0,
          borderColor: '#6B7280',
          borderWidth: 1,
          strokeDashArray: 5,
        },
      ],
      points: [
        {
          x: 5,
          y: data.find((d) => d.x === 5)?.y || 0,
          marker: {
            size: 6,
            fillColor: '#EF4444',
            strokeColor: '#fff',
          },
          label: {
            text: 'VaR 95%',
            style: {
              background: '#EF4444',
              color: '#fff',
            },
          },
        },
        {
          x: 50,
          y: data.find((d) => d.x === 50)?.y || 0,
          marker: {
            size: 6,
            fillColor: '#3B82F6',
            strokeColor: '#fff',
          },
          label: {
            text: 'Median',
            style: {
              background: '#3B82F6',
              color: '#fff',
            },
          },
        },
      ],
    },
    title: {
      text: 'Monte Carlo Percentile Distribution',
      align: 'left',
      style: {
        fontSize: '14px',
        fontWeight: 600,
      },
    },
  }), [data])

  const series = useMemo(() => [{
    name: 'Return',
    data: data.map((d) => [d.x, d.y]),
  }], [data])

  if (!percentiles || Object.keys(percentiles).length === 0) {
    return <div className="text-center text-gray-500 py-8">No data available</div>
  }

  return (
    <Chart
      options={options}
      series={series}
      type="area"
      height={300}
    />
  )
}
