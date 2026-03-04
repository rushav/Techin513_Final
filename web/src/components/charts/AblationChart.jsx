import { useMemo } from 'react'
import { PlotlyChart } from './PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'

export function AblationChart({ ablationData, height = 300 }) {
  const { labels, deltas, colors } = useMemo(() => {
    if (!ablationData) return { labels: [], deltas: [], colors: [] }
    const sorted = [...ablationData].sort((a, b) => a.delta_r2 - b.delta_r2)
    const labs = sorted.map(d => d.removed_component.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()))
    const dels = sorted.map(d => d.delta_r2)
    const cols = dels.map(v => v < -0.1 ? '#f87171' : v < -0.01 ? '#fb923c' : '#4ade80')
    return { labels: labs, deltas: dels, colors: cols }
  }, [ablationData])

  const data = useMemo(() => [{
    x: deltas,
    y: labels,
    type: 'bar',
    orientation: 'h',
    marker: { color: colors },
    hovertemplate: '<b>%{y}</b><br>ΔR²: %{x:.4f}<extra></extra>',
  }], [deltas, labels, colors])

  const layout = useMemo(() => mergeLayout({
    height,
    xaxis: { title: { text: 'ΔR² (vs. full model)' }, zeroline: true, zerolinecolor: '#475569', zerolinewidth: 1 },
    yaxis: { automargin: true },
    margin: { l: 160, r: 20, t: 30, b: 50 },
  }), [height])

  return <PlotlyChart data={data} layout={layout} style={{ height }} />
}
