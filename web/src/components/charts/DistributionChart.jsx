import { useMemo } from 'react'
import { PlotlyChart } from './PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'
import { QUALITY_COLORS, QUALITY_LABELS } from '../../utils/colors'

const QUALITY_ORDER = ['good', 'moderate', 'poor']

export function DistributionChart({ distData, feature, height = 280 }) {
  const data = useMemo(() => {
    if (!distData || !feature) return []
    const fd = distData[feature]
    if (!fd) return []
    return QUALITY_ORDER.map(q => {
      const d = fd[q]
      if (!d) return null
      return {
        x: d.x, y: d.y,
        type: 'scatter', mode: 'lines',
        fill: 'tozeroy',
        fillcolor: `${QUALITY_COLORS[q]}22`,
        line: { color: QUALITY_COLORS[q], width: 2 },
        name: QUALITY_LABELS[q],
        hovertemplate: `${QUALITY_LABELS[q]}<br>%{x:.2f}<extra></extra>`,
      }
    }).filter(Boolean)
  }, [distData, feature])

  const layout = useMemo(() => mergeLayout({
    height,
    xaxis: { title: { text: feature ? feature.replace(/_/g, ' ') : '' } },
    yaxis: { title: { text: 'Density' } },
    barmode: 'overlay',
  }), [feature, height])

  return <PlotlyChart data={data} layout={layout} style={{ height }} />
}
