import { useState } from 'react'
import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { SignalChart } from '../charts/SignalChart'
import { FFTChart } from '../charts/FFTChart'

const SIGNALS = ['temperature', 'light', 'humidity', 'noise']
const SIGNAL_LABELS = { temperature: 'Temperature', light: 'Light', humidity: 'Humidity', noise: 'Noise' }

export function SignalProcessingSection() {
  const { data: rawData } = useData('raw_signals.json')
  const { data: filtData } = useData('filtered_signals.json')
  const { data: fftData } = useData('fft_data.json')
  const [activeSignal, setActiveSignal] = useState('temperature')

  const hours = rawData?.hours ?? []

  return (
    <section id="signals" className="py-16 scroll-mt-6">
      <SectionHeader
        title="Signal Processing"
        subtitle="Raw sensor signals contain Poisson-distributed noise events and pink (1/f) background noise. We apply a Butterworth low-pass filter at 0.1 cycles/hour to recover the underlying environmental trend, then compute the power spectral density via FFT to verify the 1/f spectral signature."
      />

      <div className="flex flex-wrap gap-2 mb-6">
        {SIGNALS.map(s => (
          <button
            key={s}
            onClick={() => setActiveSignal(s)}
            className={`tab-btn ${activeSignal === s ? 'active' : ''}`}
          >
            {SIGNAL_LABELS[s]}
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Raw vs. Filtered Signal</h3>
          <SignalChart
            raw={rawData?.[activeSignal]}
            filtered={filtData?.[activeSignal]}
            hours={hours}
            signal={activeSignal}
            showRaw={true}
            height={280}
          />
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Power Spectral Density</h3>
          <FFTChart fftData={fftData} signal={activeSignal} height={280} />
        </Card>
      </div>

      <div className="callout">
        <strong className="text-slate-200">Filter Design:</strong> We use a 4th-order Butterworth low-pass filter
        with a cutoff at 0.1 cycles/hour (≈ 10-hour period). The −3 dB point preserves circadian-scale
        variation while attenuating Poisson spikes. The 1/f PSD confirms that residuals follow pink noise statistics.
      </div>
    </section>
  )
}
