import { useMemo } from 'react'
import { PlotlyChart } from './PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'

const LABELS = ['Good', 'Moderate', 'Poor']

export function ConfusionMatrix({ matrix, height = 300 }) {
  const data = useMemo(() => {
    if (!matrix) return []
    // Normalize by row for percentage display
    const norm = matrix.map(row => {
      const s = row.reduce((a, b) => a + b, 0)
      return row.map(v => s > 0 ? v / s : 0)
    })
    const text = matrix.map(row => row.map(v => String(v)))
    return [{
      z: norm,
      x: LABELS,
      y: LABELS,
      type: 'heatmap',
      colorscale: [[0, '#12121a'], [0.5, '#4338ca'], [1, '#6366f1']],
      showscale: false,
      text,
      texttemplate: '<b>%{text}</b>',
      textfont: { color: '#f1f5f9', size: 14 },
      hovertemplate: 'True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>',
    }]
  }, [matrix])

  const layout = useMemo(() => mergeLayout({
    height,
    xaxis: { title: { text: 'Predicted' }, side: 'bottom' },
    yaxis: { title: { text: 'True' }, autorange: 'reversed' },
    margin: { l: 70, r: 20, t: 30, b: 60 },
  }), [height])

  return <PlotlyChart data={data} layout={layout} style={{ height }} />
}
