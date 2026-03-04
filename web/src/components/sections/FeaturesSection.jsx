import { useState } from 'react'
import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { FeatureImportance } from '../charts/FeatureImportance'
import { DistributionChart } from '../charts/DistributionChart'

const FEATURE_OPTIONS = [
  'temp_mean', 'temp_std', 'light_mean', 'light_std',
  'humidity_mean', 'humidity_std', 'noise_mean', 'noise_std',
  'temp_trend', 'light_trend',
]

export function FeaturesSection() {
  const { data: featData } = useData('feature_importance.json')
  const { data: distData } = useData('distribution_data.json')
  const [activeFeat, setActiveFeat] = useState('temp_mean')

  return (
    <section id="features" className="py-16 scroll-mt-6">
      <SectionHeader
        title="Feature Extraction"
        subtitle="We extract 20 statistical and spectral features from each filtered signal channel: mean, standard deviation, min/max, skewness, kurtosis, linear trend slope, dominant frequency, and spectral entropy. These form the feature matrix for ML training."
      />

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Feature Importance (Random Forest)</h3>
          <FeatureImportance data={featData} topN={12} height={320} />
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Distribution by Quality Class</h3>
          <div className="flex flex-wrap gap-1.5 mb-3">
            {FEATURE_OPTIONS.map(f => (
              <button
                key={f}
                onClick={() => setActiveFeat(f)}
                className={`pill text-xs transition-all ${
                  activeFeat === f
                    ? 'border-primary text-primary bg-primary/10'
                    : 'border-border text-slate-500 hover:border-slate-500 hover:text-slate-300'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
          <DistributionChart distData={distData} feature={activeFeat} height={260} />
        </Card>
      </div>

      <div className="callout">
        <strong className="text-slate-200">Key insight:</strong> Temperature mean and standard deviation
        are the top two predictors, followed by light mean. This reflects the physiological importance
        of thermoregulation during sleep — a bedroom temperature between 18–22°C is associated with
        deeper, higher-quality sleep.
      </div>
    </section>
  )
}
