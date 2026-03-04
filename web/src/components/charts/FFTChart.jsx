import { useMemo } from 'react'
import { PlotlyChart } from './PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'
import { SIGNAL_COLORS } from '../../utils/colors'

export function FFTChart({ fftData, signal = 'temperature', height = 280 }) {
  const data = useMemo(() => {
    if (!fftData) return []
    const d = fftData[signal]
    if (!d) return []
    return [{
      x: d.freqs,
      y: d.power,
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      fillcolor: `${SIGNAL_COLORS[signal]}22`,
      line: { color: SIGNAL_COLORS[signal], width: 1.5 },
      name: signal,
    }]
  }, [fftData, signal])

  const layout = useMemo(() => mergeLayout({
    height,
    xaxis: { title: { text: 'Frequency (cycles / hour)' }, type: 'log' },
    yaxis: { title: { text: 'Power Spectral Density' } },
  }), [height])

  return <PlotlyChart data={data} layout={layout} style={{ height }} />
}
