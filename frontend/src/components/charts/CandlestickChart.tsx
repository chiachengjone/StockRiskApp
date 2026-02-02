import { useMemo } from 'react'
import Chart from 'react-apexcharts'
import { ApexOptions } from 'apexcharts'

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
    xaxis: {
      type: 'datetime',
      labels: {
        datetimeFormatter: {
          year: 'yyyy',
          month: "MMM 'yy",
          day: 'dd MMM',
        },
      },
    },
    yaxis: {
      tooltip: {
        enabled: true,
      },
      labels: {
        formatter: (val) => `$${val.toFixed(2)}`,
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
    colors: ['#000', '#8B5CF6', '#3B82F6', '#8B5CF6'],
    legend: {
      show: true,
      position: 'top',
    },
    tooltip: {
      shared: true,
    },
  }), [])

  if (!dates || dates.length === 0) {
    return <div className="text-center text-gray-500 py-8">No data available</div>
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
