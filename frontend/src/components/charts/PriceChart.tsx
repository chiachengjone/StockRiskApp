import { useMemo } from 'react'
import Chart from 'react-apexcharts'
import { ApexOptions } from 'apexcharts'
import { useUIState } from '../../store/portfolioStore'

interface PriceChartProps {
  dates: string[]
  prices: number[]
  ticker: string
}

export default function PriceChart({ dates, prices, ticker }: PriceChartProps) {
  const { darkMode } = useUIState()
  
  const options: ApexOptions = useMemo(() => ({
    chart: {
      type: 'area',
      height: 350,
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
      animations: {
        enabled: true,
        easing: 'easeinout',
        speed: 500,
      },
    },
    theme: {
      mode: darkMode ? 'dark' : 'light',
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
        opacityFrom: 0.4,
        opacityTo: 0.1,
        stops: [0, 90, 100],
      },
    },
    grid: {
      borderColor: darkMode ? '#374151' : '#e5e7eb',
      strokeDashArray: 4,
    },
    xaxis: {
      type: 'datetime',
      categories: dates,
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
      axisBorder: {
        color: darkMode ? '#374151' : '#e5e7eb',
      },
      axisTicks: {
        color: darkMode ? '#374151' : '#e5e7eb',
      },
    },
    yaxis: {
      title: {
        text: 'Price ($)',
        style: {
          color: darkMode ? '#9ca3af' : '#6b7280',
        },
      },
      labels: {
        formatter: (val) => `$${val.toFixed(2)}`,
        style: {
          colors: darkMode ? '#9ca3af' : '#6b7280',
        },
      },
    },
    tooltip: {
      theme: darkMode ? 'dark' : 'light',
      x: {
        format: 'MMM dd, yyyy',
      },
      y: {
        formatter: (val) => `$${val.toFixed(2)}`,
      },
    },
    colors: ['#10b981'],
    title: {
      text: `${ticker} Price History`,
      align: 'left',
      style: {
        fontSize: '14px',
        fontWeight: 600,
        color: darkMode ? '#f3f4f6' : '#111827',
      },
    },
  }), [dates, ticker, darkMode])

  const series = useMemo(() => [{
    name: 'Price',
    data: prices,
  }], [prices])

  return (
    <Chart
      options={options}
      series={series}
      type="area"
      height={350}
    />
  )
}
