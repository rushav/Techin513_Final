import { useMemo } from 'react'
import { PlotlyChart } from './PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'

export function ROCChart({ rocData, height = 300 }) {
  const data = useMemo(() => {
    const curves = rocData?.curves ?? []
    if (!curves.length) return []
    const traces = curves.map(c => ({
      x: c.fpr, y: c.tpr,
      type: 'scatter', mode: 'lines',
      name: `${c.class} (AUC=${c.auc?.toFixed(2)})`,
      line: { color: c.color, width: 2 },
    }))
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
