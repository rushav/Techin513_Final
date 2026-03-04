import { useState, useMemo } from 'react'
import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { PlotlyChart } from '../charts/PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'
import { SIGNAL_COLORS, QUALITY_COLORS } from '../../utils/colors'

// parameter_explorer.json: { grid: [{ quality, temp_setting, t_hours, temperature, light, humidity, noise, ... }] }
// temp_setting values: cold, optimal, warm
// quality values: poor, moderate, good

const QUALITY_ORDER = ['good', 'moderate', 'poor']
const TEMP_ORDER = ['cold', 'optimal', 'warm']
const TEMP_LABELS = { cold: 'Cold (~18°C)', optimal: 'Optimal (~20°C)', warm: 'Warm (~24°C)' }
const SIGNALS = ['temperature', 'light', 'humidity', 'noise']
const SIGNAL_UNITS = { temperature: '°C', light: 'lux', humidity: '%', noise: 'dB' }

export function ExplorerSection() {
  const { data: explorerData, loading } = useData('parameter_explorer.json')
  const [quality, setQuality] = useState('good')
  const [temp, setTemp] = useState('optimal')
  const [signal, setSignal] = useState('temperature')

  const session = useMemo(() => {
    if (!explorerData?.grid) return null
    return explorerData.grid.find(s => s.quality === quality && s.temp_setting === temp) ?? null
  }, [explorerData, quality, temp])

  const hours = session?.t_hours ?? []
  const sigData = session?.[signal] ?? []

  const traces = sigData.length ? [{
    x: hours, y: sigData,
    type: 'scatter', mode: 'lines',
    name: signal,
    line: { color: SIGNAL_COLORS[signal], width: 2 },
  }] : []

  return (
    <section id="explorer" className="py-16 scroll-mt-6">
      <SectionHeader
        title="Parameter Explorer"
        subtitle="Explore how sleep quality class and room temperature profile affect sensor signals. Select a quality class and temperature setting to view a representative 8-hour session."
      />

      <div className="grid lg:grid-cols-3 gap-6 mb-6">
        <Card animate={false}>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Sleep Quality</h3>
          <div className="flex flex-col gap-2">
            {QUALITY_ORDER.map(q => (
              <button key={q} onClick={() => setQuality(q)}
                className="px-4 py-3 rounded-lg text-sm font-medium transition-all border text-left capitalize"
                style={quality === q
                  ? { borderColor: QUALITY_COLORS[q], color: QUALITY_COLORS[q], background: `${QUALITY_COLORS[q]}15` }
                  : { borderColor: '#1e1e2e', color: '#64748b' }}
              >
                {q}
              </button>
            ))}
          </div>
        </Card>

        <Card animate={false}>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Temperature Setting</h3>
          <div className="flex flex-col gap-2">
            {TEMP_ORDER.map(t => (
              <button key={t} onClick={() => setTemp(t)}
                className={`px-4 py-3 rounded-lg text-sm font-medium transition-all border text-left ${
                  temp === t ? 'border-secondary text-secondary bg-secondary/10' : 'border-border text-slate-500 hover:text-slate-300'
                }`}
              >
                {TEMP_LABELS[t]}
              </button>
            ))}
          </div>
        </Card>

        <Card animate={false}>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Signal Channel</h3>
          <div className="flex flex-col gap-2">
            {SIGNALS.map(s => (
              <button key={s} onClick={() => setSignal(s)}
                className={`px-4 py-3 rounded-lg text-sm font-medium transition-all border text-left capitalize ${
                  signal === s ? 'border-primary text-primary bg-primary/10' : 'border-border text-slate-500 hover:text-slate-300'
                }`}
              >
                {s} <span className="text-xs opacity-60">({SIGNAL_UNITS[s]})</span>
              </button>
            ))}
          </div>
        </Card>
      </div>

      <Card>
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <h3 className="text-sm font-semibold text-slate-300">
            Session — <span className="capitalize">{quality}</span> · {TEMP_LABELS[temp]}
          </h3>
          {session && (
            <div className="flex gap-2 flex-wrap text-xs font-mono text-slate-500">
              <Badge variant={quality === 'good' ? 'success' : quality === 'moderate' ? 'warning' : 'danger'}>
                score: {session.sleep_score?.toFixed(1)}
              </Badge>
              <Badge variant="neutral">eff: {(session.sleep_efficiency * 100)?.toFixed(1)}%</Badge>
              <Badge variant="neutral">wakes: {session.awakenings}</Badge>
            </div>
          )}
        </div>
        {loading ? (
          <div className="h-64 flex items-center justify-center text-slate-500 text-sm">Loading…</div>
        ) : traces.length ? (
          <PlotlyChart
            data={traces}
            layout={mergeLayout({
              height: 300,
              xaxis: { title: { text: 'Hour of Night' } },
              yaxis: { title: { text: `${signal} (${SIGNAL_UNITS[signal]})` } },
            })}
            style={{ height: 300 }}
          />
        ) : (
          <div className="h-64 flex items-center justify-center text-slate-500 text-sm">No data for this selection.</div>
        )}
      </Card>
    </section>
  )
}
