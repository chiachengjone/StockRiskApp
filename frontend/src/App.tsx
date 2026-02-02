import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import SingleStock from './pages/SingleStock'
import Portfolio from './pages/Portfolio'
import TechnicalAnalysis from './pages/TechnicalAnalysis'
import Reports from './pages/Reports'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="single-stock" element={<SingleStock />} />
        <Route path="portfolio" element={<Portfolio />} />
        <Route path="technical-analysis" element={<TechnicalAnalysis />} />
        <Route path="reports" element={<Reports />} />
      </Route>
    </Routes>
  )
}

export default App
