import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { BaselineChart } from '../charts/BaselineChart'

// baseline_comparison.json: { models, metrics: { mean_r2: [...] }, per_label: { label: { rf, ridge, mean } } }
export function ResultsSection() {
  const { data: baselineData } = useData('baseline_comparison.json')
  const { data: mlSummary } = useData('ml_summary.json')

  const models = baselineData?.models ?? []
  const r2s = baselineData?.metrics?.mean_r2 ?? []
  const colors = baselineData?.colors ?? ['#6366f1', '#22d3ee', '#94a3b8']
  const perLabel = baselineData?.per_label ?? {}

  return (
    <section id="results" className="py-16 scroll-mt-6">
      <SectionHeader
        title="Results"
        subtitle="We compare our Random Forest regressor against Ridge regression and a mean baseline across all sleep quality labels. RF achieves substantially higher mean R², demonstrating that nonlinear feature interactions are important for this prediction task."
      />

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Model Comparison — Mean R²</h3>
          <BaselineChart baselineData={baselineData} height={280} />
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">R² by Label & Model</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-border">
                  <th className="text-left py-2 pr-3">Label</th>
                  {models.map((m, i) => (
                    <th key={m} className="text-right py-2 px-2" style={{ color: colors[i] }}>{m.replace(' Baseline', '')}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(perLabel).map(([label, vals]) => (
                  <tr key={label} className="border-b border-border/50 hover:bg-white/5">
                    <td className="py-2 pr-3 font-mono text-slate-300">{label.replace(/_/g, ' ')}</td>
                    <td className="py-2 px-2 text-right font-mono text-primary">{vals.rf?.toFixed(3) ?? '—'}</td>
                    <td className="py-2 px-2 text-right font-mono text-secondary">{vals.ridge?.toFixed(3) ?? '—'}</td>
                    <td className="py-2 px-2 text-right font-mono text-slate-500">{vals.mean?.toFixed(3) ?? '—'}</td>
                  </tr>
                ))}
                {r2s.length > 0 && (
                  <tr className="border-t border-border font-semibold">
                    <td className="py-2 pr-3 text-slate-200">Mean R²</td>
                    <td className="py-2 px-2 text-right font-mono text-primary">{r2s[0]?.toFixed(3)}</td>
                    <td className="py-2 px-2 text-right font-mono text-secondary">{r2s[1]?.toFixed(3)}</td>
                    <td className="py-2 px-2 text-right font-mono text-slate-500">{r2s[2]?.toFixed(3)}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          {mlSummary && (
            <div className="mt-3 flex gap-2 flex-wrap">
              <Badge variant="primary">RF Δ vs Ridge: +{mlSummary.rf_vs_ridge_delta?.toFixed(3) ?? baselineData?.rf_vs_ridge_delta?.toFixed(3) ?? '—'}</Badge>
              <Badge variant="secondary">RF Δ vs Mean: +{mlSummary.rf_vs_mean_delta?.toFixed(3) ?? baselineData?.rf_vs_mean_delta?.toFixed(3) ?? '—'}</Badge>
            </div>
          )}
        </Card>
      </div>
    </section>
  )
}
