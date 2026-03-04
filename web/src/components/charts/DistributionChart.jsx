import { useMemo } from 'react'
import { PlotlyChart } from './PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'

// distribution_data.json has {xs, kde_synthetic, kde_reference} per feature
export function DistributionChart({ distData, feature, height = 280 }) {
  const data = useMemo(() => {
    if (!distData || !feature) return []
    const fd = distData[feature]
    if (!fd) return []
    const traces = []
    if (fd.kde_synthetic) {
      traces.push({
        x: fd.xs, y: fd.kde_synthetic,
        type: 'scatter', mode: 'lines',
        fill: 'tozeroy', fillcolor: 'rgba(99,102,241,0.15)',
        line: { color: '#6366f1', width: 2 },
        name: 'Synthetic',
      })
    }
    if (fd.kde_reference) {
      traces.push({
        x: fd.xs, y: fd.kde_reference,
        type: 'scatter', mode: 'lines',
        fill: 'tozeroy', fillcolor: 'rgba(34,211,238,0.1)',
        line: { color: '#22d3ee', width: 2, dash: 'dash' },
        name: 'Reference',
      })
    }
    return traces
  }, [distData, feature])

  const layout = useMemo(() => mergeLayout({
    height,
    xaxis: { title: { text: feature ? feature.replace(/_/g, ' ') : '' } },
    yaxis: { title: { text: 'Density' } },
  }), [feature, height])

  return <PlotlyChart data={data} layout={layout} style={{ height }} />
}
