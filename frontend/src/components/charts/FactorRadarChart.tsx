/**
 * Factor Radar Chart
 * ==================
 * Visualize factor exposures on a radar chart
 */

import { useMemo } from 'react'
import Chart from 'react-apexcharts'
import { ApexOptions } from 'apexcharts'
import { useUIState } from '../../store/portfolioStore'

interface FactorRadarChartProps {
  factors: {
    name: string
    value: number
  }[]
  title?: string
}

export default function FactorRadarChart({ factors, title = 'Factor Exposures' }: FactorRadarChartProps) {
  const { darkMode } = useUIState()
  
  const options: ApexOptions = useMemo(() => ({
    chart: {
      type: 'radar',
      height: 350,
      background: 'transparent',
      toolbar: {
        show: false,
      },
    },
    theme: {
      mode: darkMode ? 'dark' : 'light',
    },
    title: {
      text: title,
      align: 'left',
      style: {
        fontSize: '14px',
        fontWeight: 600,
        color: darkMode ? '#f3f4f6' : '#111827',
      },
    },
    xaxis: {
      categories: factors.map(f => f.name),
      labels: {
        style: {
          colors: darkMode ? '#9ca3af' : '#6b7280',
          fontSize: '12px',
        },
      },
    },
    yaxis: {
      show: false,
    },
    stroke: {
      width: 2,
    },
    fill: {
      opacity: 0.3,
    },
    markers: {
      size: 4,
      hover: {
        size: 6,
      },
    },
    colors: ['#10b981'],
    tooltip: {
      theme: darkMode ? 'dark' : 'light',
      y: {
        formatter: (val) => val.toFixed(3),
      },
    },
    plotOptions: {
      radar: {
        polygons: {
          strokeColors: darkMode ? '#374151' : '#e5e7eb',
          fill: {
            colors: darkMode ? ['#1f2937', '#111827'] : ['#f9fafb', '#ffffff'],
          },
        },
      },
    },
  }), [factors, title, darkMode])

  const series = useMemo(() => [{
    name: 'Exposure',
    data: factors.map(f => f.value),
  }], [factors])

  return (
    <Chart
      options={options}
      series={series}
      type="radar"
      height={350}
    />
  )
}
