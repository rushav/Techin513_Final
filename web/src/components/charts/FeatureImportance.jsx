import { useMemo } from 'react'
import { PlotlyChart } from './PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'

export function FeatureImportance({ data: featData, topN = 12, height = 320 }) {
  const { features, importances } = useMemo(() => {
    if (!featData) return { features: [], importances: [] }
    const sorted = [...featData].sort((a, b) => b.importance - a.importance).slice(0, topN)
    return {
      features: sorted.map(d => d.feature).reverse(),
      importances: sorted.map(d => d.importance).reverse(),
    }
  }, [featData, topN])

  const data = useMemo(() => [{
    x: importances,
    y: features,
    type: 'bar',
    orientation: 'h',
    marker: {
      color: importances.map((v, i) => {
        const t = i / (importances.length - 1)
        return `rgba(${Math.round(99 + (34-99)*t)},${Math.round(102 + (211-102)*t)},${Math.round(241 + (238-241)*t)},0.85)`
      }),
    },
    hovertemplate: '<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
  }], [importances, features])

  const layout = useMemo(() => mergeLayout({
    height,
    xaxis: { title: { text: 'Mean Decrease in Impurity' } },
    yaxis: { automargin: true },
    margin: { l: 130, r: 20, t: 30, b: 50 },
  }), [height])

  return <PlotlyChart data={data} layout={layout} style={{ height }} />
}
