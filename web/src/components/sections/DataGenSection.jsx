import { useState } from 'react'
import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { PlotlyChart } from '../charts/PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'
import { QUALITY_COLORS, QUALITY_LABELS } from '../../utils/colors'

const QUALITY_ORDER = ['good', 'moderate', 'poor']

export function DataGenSection() {
  const { data: stats } = useData('dataset_stats.json')
  const { data: distData } = useData('distribution_data.json')
  const [activeQ, setActiveQ] = useState('good')

  const qualDist = distData?.session_quality
  const qualTraces = qualDist
    ? QUALITY_ORDER.map(q => ({
        x: qualDist[q]?.x ?? [],
        y: qualDist[q]?.y ?? [],
        type: 'scatter', mode: 'lines',
        fill: 'tozeroy',
        fillcolor: `${QUALITY_COLORS[q]}22`,
        line: { color: QUALITY_COLORS[q], width: 2 },
        name: QUALITY_LABELS[q],
      }))
    : []

  return (
    <section id="data-gen" className="py-16 scroll-mt-6">
      <SectionHeader
        title="Data Generation"
        subtitle="We generate 500 synthetic sleep sessions using a physiologically-motivated model. Each session simulates 8 hours of bedroom sensor data at 1 Hz, sampling temperature, light, humidity, and noise according to session-specific profiles drawn from predefined quality classes."
      />

      <div className="grid lg:grid-cols-3 gap-4 mb-6">
        {[
          { label: 'Total Sessions', value: stats?.n_sessions ?? '—', variant: 'primary' },
          { label: 'Duration', value: '8 hours', variant: 'secondary' },
          { label: 'Signals', value: '4 channels', variant: 'success' },
        ].map(({ label, value, variant }) => (
          <Card key={label} className="text-center">
            <div className="text-2xl font-bold font-mono mb-1" style={{ color: 'var(--tw-color-primary, #6366f1)' }}>
              {value}
            </div>
            <div className="text-xs text-slate-400 uppercase tracking-wider">{label}</div>
          </Card>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Quality Score Distribution by Class</h3>
          <PlotlyChart
            data={qualTraces}
            layout={mergeLayout({
              height: 260,
              xaxis: { title: { text: 'Sleep Quality Score' } },
              yaxis: { title: { text: 'Density' } },
            })}
            style={{ height: 260 }}
          />
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Session Profile Parameters</h3>
          <div className="space-y-3 text-sm text-slate-400">
            {[
              { param: 'Base Temperature', range: '16–26 °C', note: 'Gaussian by quality class' },
              { param: 'Light Level', range: '0–50 lux', note: 'Good=low, Poor=high' },
              { param: 'Humidity', range: '30–70 %', note: 'Optimal near 45–55%' },
              { param: 'Noise Events', range: '0–30 / night', note: 'Poisson process, λ by class' },
              { param: 'Pink Noise', range: '1/f spectrum', note: 'Added to all channels' },
              { param: 'Circadian Drift', range: '±0.5 °C', note: 'Sinusoidal over 8 h' },
            ].map(({ param, range, note }) => (
              <div key={param} className="flex items-start gap-3">
                <div className="w-40 shrink-0 font-medium text-slate-300">{param}</div>
                <div className="font-mono text-xs bg-black/40 px-2 py-0.5 rounded text-secondary shrink-0">{range}</div>
                <div className="text-slate-500 text-xs">{note}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </section>
  )
}
