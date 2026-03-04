import { useMemo } from 'react'
import { PlotlyChart } from './PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'
import { CLASS_COLORS } from '../../utils/colors'

const CLASS_LABELS = ['Good', 'Moderate', 'Poor']

export function ROCChart({ rocData, height = 300 }) {
  const data = useMemo(() => {
    if (!rocData) return []
    const traces = rocData.map((c, i) => ({
      x: c.fpr, y: c.tpr,
      type: 'scatter', mode: 'lines',
      name: `${CLASS_LABELS[i]} (AUC=${c.auc.toFixed(2)})`,
      line: { color: CLASS_COLORS[i], width: 2 },
    }))
    // Diagonal
    traces.push({
      x: [0, 1], y: [0, 1],
      type: 'scatter', mode: 'lines',
      name: 'Random',
      line: { color: '#475569', width: 1, dash: 'dash' },
      showlegend: false,
    })
    return traces
  }, [rocData])

  const layout = useMemo(() => mergeLayout({
    height,
    xaxis: { title: { text: 'False Positive Rate' }, range: [0, 1] },
    yaxis: { title: { text: 'True Positive Rate' }, range: [0, 1] },
  }), [height])

  return <PlotlyChart data={data} layout={layout} style={{ height }} />
}
