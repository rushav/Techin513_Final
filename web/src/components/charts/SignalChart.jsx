import { useMemo } from 'react'
import { PlotlyChart } from './PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'
import { SIGNAL_COLORS } from '../../utils/colors'

export function SignalChart({ raw, filtered, hours, signal = 'temperature', showRaw = true, height = 280 }) {
  const data = useMemo(() => {
    const traces = []
    if (showRaw && raw) {
      traces.push({
        x: hours, y: raw,
        type: 'scatter', mode: 'lines',
        name: 'Raw',
        line: { color: SIGNAL_COLORS[signal], width: 1, dash: 'dot' },
        opacity: 0.5,
      })
    }
    if (filtered) {
      traces.push({
        x: hours, y: filtered,
        type: 'scatter', mode: 'lines',
        name: 'Filtered',
        line: { color: SIGNAL_COLORS[signal], width: 2 },
      })
    }
    return traces
  }, [raw, filtered, hours, signal, showRaw])

  const layout = useMemo(() => mergeLayout({
    height,
    xaxis: { title: { text: 'Hour of Night' } },
    yaxis: { title: { text: signal === 'temperature' ? 'Temperature (°C)' : signal === 'humidity' ? 'Humidity (%)' : signal === 'noise' ? 'Noise (dB)' : 'Lux' } },
    showlegend: showRaw,
  }), [signal, showRaw, height])

  return <PlotlyChart data={data} layout={layout} style={{ height }} />
}
