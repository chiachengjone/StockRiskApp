/**
 * Volatility Surface Chart
 * ========================
 * 3D-like visualization of option volatility surface
 */

import { useMemo } from 'react'
import Chart from 'react-apexcharts'
import { ApexOptions } from 'apexcharts'
import { useUIState } from '../../store/portfolioStore'

interface VolatilitySurfaceProps {
  strikes: number[]
  expiries: string[]
  volatilities: number[][] // 2D array [expiry][strike]
}

export default function VolatilitySurfaceChart({ strikes, expiries, volatilities }: VolatilitySurfaceProps) {
  const { darkMode } = useUIState()
  
  // Convert to heatmap format
  const series = useMemo(() => {
    return expiries.map((expiry, i) => ({
      name: expiry,
      data: strikes.map((strike, j) => ({
        x: `$${strike}`,
        y: volatilities[i]?.[j] ? (volatilities[i][j] * 100).toFixed(1) : 0,
      })),
    }))
  }, [strikes, expiries, volatilities])

  const options: ApexOptions = useMemo(() => ({
    chart: {
      type: 'heatmap',
      height: 400,
      background: 'transparent',
      toolbar: {
        show: true,
      },
    },
    theme: {
      mode: darkMode ? 'dark' : 'light',
    },
    dataLabels: {
      enabled: true,
      style: {
        colors: [darkMode ? '#ffffff' : '#111827'],
        fontSize: '10px',
      },
      formatter: (val) => `${val}%`,
    },
    colors: ['#10b981'],
    title: {
      text: 'Implied Volatility Surface',
      align: 'left',
      style: {
        fontSize: '14px',
        fontWeight: 600,
        color: darkMode ? '#f3f4f6' : '#111827',
      },
    },
    xaxis: {
      title: {
        text: 'Strike Price',
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
    yaxis: {
      title: {
        text: 'Expiry',
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
    tooltip: {
      theme: darkMode ? 'dark' : 'light',
    },
    plotOptions: {
      heatmap: {
        shadeIntensity: 0.5,
        colorScale: {
          ranges: [
            { from: 0, to: 20, color: '#10b981', name: 'Low' },
            { from: 20, to: 30, color: '#f59e0b', name: 'Medium' },
            { from: 30, to: 50, color: '#ef4444', name: 'High' },
            { from: 50, to: 100, color: '#7c3aed', name: 'Very High' },
          ],
        },
      },
    },
  }), [darkMode])

  if (!volatilities.length || !strikes.length) {
    return (
      <div className="text-center py-8 text-gray-500">
        No volatility surface data available
      </div>
    )
  }

  return (
    <Chart
      options={options}
      series={series}
      type="heatmap"
      height={400}
    />
  )
}
