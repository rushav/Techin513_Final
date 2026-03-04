import { useMemo } from 'react'
import { PlotlyChart } from './PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'
import { MODEL_COLORS } from '../../utils/colors'

export function BaselineChart({ baselineData, metric = 'r2', height = 280 }) {
  const data = useMemo(() => {
    if (!baselineData) return []
    const models = Object.keys(baselineData)
    const colors = { rf: MODEL_COLORS.rf, ridge: MODEL_COLORS.ridge, mean_baseline: MODEL_COLORS.mean }
    return [{
      x: models.map(m => m.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())),
      y: models.map(m => baselineData[m][metric] ?? 0),
      type: 'bar',
      marker: {
        color: models.map(m => colors[m] || '#6366f1'),
        opacity: 0.85,
      },
      hovertemplate: '<b>%{x}</b><br>' + metric.toUpperCase() + ': %{y:.4f}<extra></extra>',
    }]
  }, [baselineData, metric])

  const layout = useMemo(() => mergeLayout({
    height,
    xaxis: { title: { text: 'Model' } },
    yaxis: { title: { text: metric.toUpperCase() } },
    margin: { l: 55, r: 20, t: 30, b: 60 },
  }), [metric, height])

  return <PlotlyChart data={data} layout={layout} style={{ height }} />
}
