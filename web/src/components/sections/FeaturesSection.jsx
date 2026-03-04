import { useState } from 'react'
import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { FeatureImportance } from '../charts/FeatureImportance'
import { DistributionChart } from '../charts/DistributionChart'

// distribution_data.json has keys: sleep_efficiency, sleep_duration_h, awakenings, sleep_score, temperature_mean
const DIST_FEATURES = [
  { key: 'sleep_efficiency',  label: 'Sleep Efficiency' },
  { key: 'sleep_duration_h',  label: 'Sleep Duration (h)' },
  { key: 'awakenings',        label: 'Awakenings' },
  { key: 'sleep_score',       label: 'Sleep Score' },
  { key: 'temperature_mean',  label: 'Temperature Mean' },
]

export function FeaturesSection() {
  const { data: featData } = useData('feature_importance.json')
  const { data: distData } = useData('distribution_data.json')
  const [activeFeat, setActiveFeat] = useState('sleep_score')

  return (
    <section id="features" className="py-16 scroll-mt-6">
      <SectionHeader
        title="Feature Extraction"
        subtitle="We extract statistical and spectral features from each filtered signal channel: mean, standard deviation, min/max, skewness, kurtosis, linear trend slope, dominant frequency, and spectral entropy. These 20+ features form the ML input matrix."
      />

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Feature Importance (Random Forest)</h3>
          <FeatureImportance data={featData} topN={12} height={320} />
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Label Distribution — Synthetic vs. Reference</h3>
          <div className="flex flex-wrap gap-1.5 mb-3">
            {DIST_FEATURES.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setActiveFeat(key)}
                className={`pill text-xs transition-all ${
                  activeFeat === key
                    ? 'border-primary text-primary bg-primary/10'
                    : 'border-border text-slate-500 hover:border-slate-500 hover:text-slate-300'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <DistributionChart distData={distData} feature={activeFeat} height={260} />
          <p className="text-xs text-slate-500 mt-2">
            Synthetic KDE (solid) vs. reference distribution (dashed). Close alignment validates that
            our synthetic generator reproduces realistic sleep label distributions.
          </p>
        </Card>
      </div>

      <div className="callout">
        <strong className="text-slate-200">Key insight:</strong> Temperature-derived features dominate
        importance, consistent with thermoregulation's critical role in sleep quality. Light level
        features rank second, reflecting disruption from nocturnal light exposure.
      </div>
    </section>
  )
}
