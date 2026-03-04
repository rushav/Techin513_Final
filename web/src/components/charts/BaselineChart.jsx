import { useMemo } from 'react'
import { PlotlyChart } from './PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'
import { MODEL_COLORS } from '../../utils/colors'

// baseline_comparison.json: { models: [...], metrics: { mean_r2: [rf, ridge, mean] }, per_label: {...} }
export function BaselineChart({ baselineData, height = 280 }) {
  const data = useMemo(() => {
    if (!baselineData) return []
    const models = baselineData.models ?? []
    const r2s = baselineData.metrics?.mean_r2 ?? []
    const palette = [MODEL_COLORS.rf, MODEL_COLORS.ridge, MODEL_COLORS.mean]
    return [{
      x: models,
      y: r2s,
      type: 'bar',
      marker: { color: palette.slice(0, models.length), opacity: 0.85 },
      hovertemplate: '<b>%{x}</b><br>Mean R²: %{y:.4f}<extra></extra>',
    }]
  }, [baselineData])

  const layout = useMemo(() => mergeLayout({
    height,
    xaxis: { title: { text: 'Model' } },
    yaxis: { title: { text: 'Mean R² (test)' } },
    margin: { l: 55, r: 20, t: 30, b: 60 },
  }), [height])

  return <PlotlyChart data={data} layout={layout} style={{ height }} />
}
