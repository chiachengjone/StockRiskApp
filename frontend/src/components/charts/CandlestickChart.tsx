import { useMemo } from 'react'
import Chart from 'react-apexcharts'
import { ApexOptions } from 'apexcharts'
import { useUIState } from '../../store/portfolioStore'

interface CandlestickChartProps {
  dates: string[]
  open: number[]
  high: number[]
  low: number[]
  close: number[]
  bollingerUpper?: number[]
  bollingerMiddle?: number[]
  bollingerLower?: number[]
}

export default function CandlestickChart({
  dates,
  open,
  high,
  low,
  close,
  bollingerUpper,
  bollingerMiddle,
  bollingerLower,
}: CandlestickChartProps) {
  const { darkMode } = useUIState()
  
  const candlestickData = useMemo(() => 
    dates.map((date, i) => ({
      x: new Date(date).getTime(),
      y: [open[i], high[i], low[i], close[i]],
    })),
  [dates, open, high, low, close])

  const series: ApexAxisChartSeries = useMemo(() => {
    const result: ApexAxisChartSeries = [
      {
        name: 'Price',
        type: 'candlestick',
        data: candlestickData,
      },
    ]

    if (bollingerUpper && bollingerMiddle && bollingerLower) {
      result.push(
        {
          name: 'Upper Band',
          type: 'line',
          data: dates.map((date, i) => ({
            x: new Date(date).getTime(),
            y: bollingerUpper[i],
          })),
        },
        {
          name: 'SMA 20',
          type: 'line',
          data: dates.map((date, i) => ({
            x: new Date(date).getTime(),
            y: bollingerMiddle[i],
          })),
        },
        {
          name: 'Lower Band',
          type: 'line',
          data: dates.map((date, i) => ({
            x: new Date(date).getTime(),
            y: bollingerLower[i],
          })),
        }
      )
    }

    return result
  }, [candlestickData, dates, bollingerUpper, bollingerMiddle, bollingerLower])

  const options: ApexOptions = useMemo(() => ({
    chart: {
      type: 'candlestick',
      height: 400,
      background: 'transparent',
      toolbar: {
        show: true,
        tools: {
          download: true,
          selection: true,
          zoom: true,
          zoomin: true,
          zoomout: true,
          pan: true,
          reset: true,
        },
      },
    },
    theme: {
      mode: darkMode ? 'dark' : 'light',
    },
    grid: {
      borderColor: darkMode ? '#374151' : '#e5e7eb',
    },
    xaxis: {
      type: 'datetime',
      labels: {
        datetimeFormatter: {
          year: 'yyyy',
          month: "MMM 'yy",
          day: 'dd MMM',
        },
        style: {
          colors: darkMode ? '#9ca3af' : '#6b7280',
        },
      },
    },
    yaxis: {
      tooltip: {
        enabled: true,
      },
      labels: {
        formatter: (val) => `$${val.toFixed(2)}`,
        style: {
          colors: darkMode ? '#9ca3af' : '#6b7280',
        },
      },
    },
    plotOptions: {
      candlestick: {
        colors: {
          upward: '#10B981',
          downward: '#EF4444',
        },
      },
    },
    stroke: {
      width: [1, 1, 1, 1],
      curve: 'smooth',
    },
    colors: [darkMode ? '#6b7280' : '#374151', '#10b981', '#3B82F6', '#10b981'],
    legend: {
      show: true,
      position: 'top',
      labels: {
        colors: darkMode ? '#d1d5db' : '#374151',
      },
    },
    tooltip: {
      shared: true,
      theme: darkMode ? 'dark' : 'light',
    },
  }), [darkMode])

  if (!dates || dates.length === 0) {
    return <div className={`text-center py-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>No data available</div>
  }

  return (
    <Chart
      options={options}
      series={series}
      type="candlestick"
      height={400}
    />
  )
}
