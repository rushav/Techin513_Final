import { useState } from 'react'
import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { FFTChart } from '../charts/FFTChart'
import { PlotlyChart } from '../charts/PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'
import { SIGNAL_COLORS } from '../../utils/colors'

// raw_signals.json:  { t_hours, sessions: { good: { temperature, light, humidity, noise }, ... } }
// filtered_signals.json: { t_hours, signals: { good: { raw_temperature, filtered_temperature }, ... } }

const SIGNALS = ['temperature', 'light', 'humidity', 'noise']
const SIGNAL_LABELS = { temperature: 'Temperature', light: 'Light', humidity: 'Humidity', noise: 'Noise' }
const QUALITY = 'good'

export function SignalProcessingSection() {
  const { data: rawData } = useData('raw_signals.json')
  const { data: filtData } = useData('filtered_signals.json')
  const { data: fftData } = useData('fft_data.json')
  const [activeSignal, setActiveSignal] = useState('temperature')

  const hours = rawData?.t_hours ?? []
  const rawSig = rawData?.sessions?.[QUALITY]?.[activeSignal] ?? []
  const filtSig = filtData?.signals?.[QUALITY]?.['filtered_' + activeSignal]
    ?? filtData?.signals?.[QUALITY]?.[activeSignal]
    ?? []

  const yLabel = activeSignal === 'temperature' ? 'Temperature (°C)'
    : activeSignal === 'humidity' ? 'Humidity (%)'
    : activeSignal === 'noise' ? 'Noise (dB)'
    : 'Lux'

  const chartTraces = [
    rawSig.length ? {
      x: hours, y: rawSig, type: 'scatter', mode: 'lines',
      name: 'Raw', line: { color: SIGNAL_COLORS[activeSignal], width: 1, dash: 'dot' }, opacity: 0.5,
    } : null,
    filtSig.length ? {
      x: hours, y: filtSig, type: 'scatter', mode: 'lines',
      name: 'Filtered', line: { color: SIGNAL_COLORS[activeSignal], width: 2 },
    } : null,
  ].filter(Boolean)

  return (
    <section id="signals" className="py-16 scroll-mt-6">
      <SectionHeader
        title="Signal Processing"
        subtitle="Raw sensor signals contain Poisson-distributed noise events and pink (1/f) background noise. We apply a Butterworth low-pass filter at 0.1 cycles/hour to recover the underlying environmental trend, then compute the PSD via FFT to verify the 1/f spectral signature."
      />

      <div className="flex flex-wrap gap-2 mb-6">
        {SIGNALS.map(s => (
          <button key={s} onClick={() => setActiveSignal(s)}
            className={`tab-btn ${activeSignal === s ? 'active' : ''}`}>
            {SIGNAL_LABELS[s]}
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Raw vs. Filtered Signal (Good-quality session)</h3>
          <PlotlyChart
            data={chartTraces}
            layout={mergeLayout({
              height: 280,
              xaxis: { title: { text: 'Hour of Night' } },
              yaxis: { title: { text: yLabel } },
              showlegend: true,
            })}
            style={{ height: 280 }}
          />
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Power Spectral Density</h3>
          <FFTChart fftData={fftData} signal={activeSignal} height={280} />
        </Card>
      </div>

      <div className="callout">
        <strong className="text-slate-200">Filter Design:</strong> We use a 4th-order Butterworth low-pass filter
        with cutoff at 0.02 cpm (1.2 cycles/hour, ≈50-minute period). This preserves circadian drift and
        HVAC cycles while attenuating high-frequency noise. The PSD confirms residuals follow 1/f pink noise statistics.
      </div>
    </section>
  )
}
