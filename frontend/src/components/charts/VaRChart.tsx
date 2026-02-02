import { useMemo } from 'react'
import Chart from 'react-apexcharts'
import { ApexOptions } from 'apexcharts'
import { useUIState } from '../../store/portfolioStore'

interface VaRChartProps {
  returns: number[]
  varValue: number
  cvarValue: number
}

export default function VaRChart({ returns, varValue, cvarValue }: VaRChartProps) {
  const { darkMode } = useUIState()
  
  // Create histogram bins
  const bins = useMemo(() => {
    if (!returns || returns.length === 0) return { categories: [], counts: [] }
    
    const min = Math.min(...returns)
    const max = Math.max(...returns)
    const binCount = 50
    const binWidth = (max - min) / binCount
    
    const binCounts = new Array(binCount).fill(0)
    const categories: string[] = []
    
    for (let i = 0; i < binCount; i++) {
      const binStart = min + i * binWidth
      categories.push(`${(binStart * 100).toFixed(1)}%`)
    }
    
    returns.forEach((r) => {
      const binIndex = Math.min(Math.floor((r - min) / binWidth), binCount - 1)
      binCounts[binIndex]++
    })
    
    return { categories, counts: binCounts }
  }, [returns])

  const options: ApexOptions = useMemo(() => ({
    chart: {
      type: 'bar',
      height: 300,
      background: 'transparent',
      toolbar: { show: false },
    },
    theme: {
      mode: darkMode ? 'dark' : 'light',
    },
    plotOptions: {
      bar: {
        borderRadius: 0,
        columnWidth: '100%',
      },
    },
    dataLabels: {
      enabled: false,
    },
    grid: {
      borderColor: darkMode ? '#374151' : '#e5e7eb',
    },
    xaxis: {
      categories: bins.categories,
      labels: {
        show: true,
        rotate: -45,
        rotateAlways: true,
        hideOverlappingLabels: true,
        showDuplicates: false,
        maxHeight: 60,
        style: {
          fontSize: '10px',
          colors: darkMode ? '#9ca3af' : '#6b7280',
        },
      },
      tickAmount: 10,
    },
    yaxis: {
      title: {
        text: 'Frequency',
        style: {
          color: darkMode ? '#9ca3af' : '#6b7280',
        },
      },
      labels: {
        style: {
          colors: darkMode ? '#9ca3af' : '#6b7280',
        },
      },
    },
    colors: ['#6366f1'],
    tooltip: {
      theme: darkMode ? 'dark' : 'light',
    },
    annotations: {
      xaxis: [
        {
          x: `${(varValue * 100).toFixed(1)}%`,
          borderColor: '#EF4444',
          borderWidth: 2,
          label: {
            text: `VaR: ${(varValue * 100).toFixed(2)}%`,
            style: {
              color: '#fff',
              background: '#EF4444',
            },
          },
        },
        {
          x: `${(cvarValue * 100).toFixed(1)}%`,
          borderColor: '#F97316',
          borderWidth: 2,
          label: {
            text: `CVaR: ${(cvarValue * 100).toFixed(2)}%`,
            style: {
              color: '#fff',
              background: '#F97316',
            },
          },
        },
      ],
    },
    title: {
      text: 'Returns Distribution',
      align: 'left',
      style: {
        fontSize: '14px',
        fontWeight: 600,
        color: darkMode ? '#f3f4f6' : '#111827',
      },
    },
  }), [bins.categories, varValue, cvarValue, darkMode])

  const series = useMemo(() => [{
    name: 'Count',
    data: bins.counts,
  }], [bins.counts])

  if (!returns || returns.length === 0) {
    return <div className="text-center text-gray-500 py-8">No data available</div>
  }

  return (
    <Chart
      options={options}
      series={series}
      type="bar"
      height={300}
    />
  )
}
