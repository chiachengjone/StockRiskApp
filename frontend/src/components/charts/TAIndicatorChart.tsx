import { useMemo } from 'react'
import Chart from 'react-apexcharts'
import { ApexOptions } from 'apexcharts'
import { useUIState } from '../../store/portfolioStore'

interface TAIndicatorChartProps {
  dates: string[]
  values: number[]
  values2?: number[]
  name: string
  name2?: string
  color: string
  color2?: string
  overbought?: number
  oversold?: number
  histogram?: number[]
}

export default function TAIndicatorChart({
  dates,
  values,
  values2,
  name,
  name2,
  color,
  color2,
  overbought,
  oversold,
  histogram,
}: TAIndicatorChartProps) {
  const { darkMode } = useUIState()
  
  const series: ApexAxisChartSeries = useMemo(() => {
    const result: ApexAxisChartSeries = [
      {
        name,
        type: 'line',
        data: values,
      },
    ]

    if (values2 && name2) {
      result.push({
        name: name2,
        type: 'line',
        data: values2,
      })
    }

    if (histogram) {
      result.push({
        name: 'Histogram',
        type: 'bar',
        data: histogram,
      })
    }

    return result
  }, [values, values2, name, name2, histogram])

  const options: ApexOptions = useMemo(() => {
    const annotations: ApexAnnotations = {
      yaxis: [],
    }

    if (overbought !== undefined) {
      annotations.yaxis!.push({
        y: overbought,
        borderColor: '#EF4444',
        borderWidth: 1,
        strokeDashArray: 5,
        label: {
          text: 'Overbought',
          style: {
            color: '#EF4444',
            background: 'transparent',
          },
        },
      })
    }

    if (oversold !== undefined) {
      annotations.yaxis!.push({
        y: oversold,
        borderColor: '#10B981',
        borderWidth: 1,
        strokeDashArray: 5,
        label: {
          text: 'Oversold',
          style: {
            color: '#10B981',
            background: 'transparent',
          },
        },
      })
    }

    return {
      chart: {
        type: 'line',
        height: 200,
        toolbar: { show: false },
        background: 'transparent',
        animations: {
          enabled: false,
        },
      },
      theme: {
        mode: darkMode ? 'dark' : 'light',
      },
      stroke: {
        width: histogram ? [2, 2, 0] : [2, 2],
        curve: 'smooth',
      },
      grid: {
        borderColor: darkMode ? '#374151' : '#e5e7eb',
      },
      xaxis: {
        type: 'datetime',
        categories: dates,
        labels: {
          show: false,
          style: {
            colors: darkMode ? '#9ca3af' : '#6b7280',
          },
        },
        axisTicks: {
          show: false,
        },
      },
      yaxis: {
        labels: {
          formatter: (val) => val?.toFixed(1) || '',
          style: {
            colors: darkMode ? '#9ca3af' : '#6b7280',
          },
        },
      },
      tooltip: {
        theme: darkMode ? 'dark' : 'light',
      },
      colors: histogram 
        ? [color, color2 || '#6B7280', '#9CA3AF'] 
        : [color, color2 || '#6B7280'],
      annotations,
      legend: {
        show: true,
        position: 'top',
        labels: {
          colors: darkMode ? '#d1d5db' : '#374151',
        },
      },
      plotOptions: {
        bar: {
          colors: {
            ranges: [
              {
                from: -1000,
                to: 0,
                color: '#EF4444',
              },
              {
                from: 0,
                to: 1000,
                color: '#10B981',
              },
            ],
          },
        },
      },
    }
  }, [dates, color, color2, overbought, oversold, histogram, darkMode])

  if (!dates || !values || values.length === 0) {
    return <div className={`text-center py-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>No data available</div>
  }

  return (
    <Chart
      options={options}
      series={series}
      type="line"
      height={200}
    />
  )
}
