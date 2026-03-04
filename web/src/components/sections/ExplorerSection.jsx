import { useState, useMemo } from 'react'
import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { SignalChart } from '../charts/SignalChart'
import { QUALITY_COLORS } from '../../utils/colors'

const QUALITY_ORDER = ['good', 'moderate', 'poor']
const TEMP_ORDER = ['cool', 'neutral', 'warm']
const TEMP_LABELS = { cool: 'Cool (~18°C)', neutral: 'Neutral (~20°C)', warm: 'Warm (~24°C)' }

export function ExplorerSection() {
  const { data: explorerData, loading } = useData('parameter_explorer.json')
  const [quality, setQuality] = useState('good')
  const [temp, setTemp] = useState('neutral')
  const [signal, setSignal] = useState('temperature')

  const session = useMemo(() => {
    if (!explorerData) return null
    return explorerData?.[quality]?.[temp] ?? null
  }, [explorerData, quality, temp])

  const hours = session?.hours ?? []
  const rawSig = session?.[`${signal}_raw`] ?? session?.[signal] ?? null
  const filtSig = session?.[signal] ?? null

  return (
    <section id="explorer" className="py-16 scroll-mt-6">
      <SectionHeader
        title="Parameter Explorer"
        subtitle="Interactively explore how different sleep quality classes and room temperature profiles affect the sensor signals. Select a quality class and temperature setting below to view a representative session."
      />

      <div className="grid lg:grid-cols-3 gap-6 mb-6">
        <Card animate={false}>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Sleep Quality</h3>
          <div className="flex flex-col gap-2">
            {QUALITY_ORDER.map(q => (
              <button
                key={q}
                onClick={() => setQuality(q)}
                className={`px-4 py-3 rounded-lg text-sm font-medium transition-all border text-left ${
                  quality === q
                    ? 'border-current'
                    : 'border-border text-slate-500 hover:text-slate-300 hover:border-slate-500'
                }`}
                style={quality === q ? { borderColor: QUALITY_COLORS[q], color: QUALITY_COLORS[q], background: `${QUALITY_COLORS[q]}15` } : {}}
              >
                <span className="capitalize">{q}</span>
                {quality === q && <span className="ml-2 text-xs opacity-70">selected</span>}
              </button>
            ))}
          </div>
        </Card>

        <Card animate={false}>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Temperature Setting</h3>
          <div className="flex flex-col gap-2">
            {TEMP_ORDER.map(t => (
              <button
                key={t}
                onClick={() => setTemp(t)}
                className={`px-4 py-3 rounded-lg text-sm font-medium transition-all border text-left ${
                  temp === t
                    ? 'border-secondary text-secondary bg-secondary/10'
                    : 'border-border text-slate-500 hover:text-slate-300 hover:border-slate-500'
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
            {['temperature', 'light', 'humidity', 'noise'].map(s => (
              <button
                key={s}
                onClick={() => setSignal(s)}
                className={`px-4 py-3 rounded-lg text-sm font-medium transition-all border text-left capitalize ${
                  signal === s
                    ? 'border-primary text-primary bg-primary/10'
                    : 'border-border text-slate-500 hover:text-slate-300 hover:border-slate-500'
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </Card>
      </div>

      <Card>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-slate-300">
            Session Preview — <span className="capitalize">{quality}</span> quality, {TEMP_LABELS[temp]}
          </h3>
          {session && (
            <div className="flex gap-2 text-xs font-mono text-slate-500">
              <span>ID: {session.session_id}</span>
              {session.quality_score !== undefined && (
                <Badge variant={quality === 'good' ? 'success' : quality === 'moderate' ? 'warning' : 'danger'}>
                  score: {session.quality_score?.toFixed(2)}
                </Badge>
              )}
            </div>
          )}
        </div>
        {loading ? (
          <div className="h-64 flex items-center justify-center text-slate-500 text-sm">Loading session data…</div>
        ) : session ? (
          <SignalChart
            raw={rawSig}
            filtered={filtSig}
            hours={hours}
            signal={signal}
            showRaw={false}
            height={300}
          />
        ) : (
          <div className="h-64 flex items-center justify-center text-slate-500 text-sm">No session found.</div>
        )}
      </Card>
    </section>
  )
}
