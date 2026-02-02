import { useMemo } from 'react'
import Chart from 'react-apexcharts'
import { ApexOptions } from 'apexcharts'

interface PriceChartProps {
  dates: string[]
  prices: number[]
  ticker: string
}

export default function PriceChart({ dates, prices, ticker }: PriceChartProps) {
  const options: ApexOptions = useMemo(() => ({
    chart: {
      type: 'area',
      height: 350,
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
    xaxis: {
      type: 'datetime',
      categories: dates,
      labels: {
        datetimeFormatter: {
          year: 'yyyy',
          month: "MMM 'yy",
          day: 'dd MMM',
        },
      },
    },
    yaxis: {
      title: {
        text: 'Price ($)',
      },
      labels: {
        formatter: (val) => `$${val.toFixed(2)}`,
      },
    },
    tooltip: {
      x: {
        format: 'MMM dd, yyyy',
      },
      y: {
        formatter: (val) => `$${val.toFixed(2)}`,
      },
    },
    colors: ['#3B82F6'],
    title: {
      text: `${ticker} Price History`,
      align: 'left',
      style: {
        fontSize: '14px',
        fontWeight: 600,
      },
    },
  }), [dates, ticker])

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
