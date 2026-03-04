import { useState } from 'react'
import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { BaselineChart } from '../charts/BaselineChart'

export function ResultsSection() {
  const { data: baselineData } = useData('baseline_comparison.json')
  const { data: mlSummary } = useData('ml_summary.json')
  const [metric, setMetric] = useState('r2')

  return (
    <section id="results" className="py-16 scroll-mt-6">
      <SectionHeader
        title="Results"
        subtitle="We compare our Random Forest regressor against Ridge regression and a mean baseline. The RF model achieves substantially higher R² and lower RMSE, demonstrating that nonlinear feature interactions captured by ensemble methods are important for this prediction task."
      />

      <div className="flex gap-2 mb-6">
        {['r2', 'rmse', 'mae'].map(m => (
          <button
            key={m}
            onClick={() => setMetric(m)}
            className={`tab-btn ${metric === m ? 'active' : ''}`}
          >
            {m.toUpperCase()}
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Model Comparison — {metric.toUpperCase()}</h3>
          <BaselineChart baselineData={baselineData} metric={metric} height={280} />
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Performance Summary</h3>
          <div className="space-y-4">
            {baselineData && Object.entries(baselineData).map(([model, metrics]) => {
              const variant = model === 'rf' ? 'primary' : model === 'ridge' ? 'secondary' : 'neutral'
              const name = model === 'rf' ? 'Random Forest' : model === 'ridge' ? 'Ridge Regression' : 'Mean Baseline'
              return (
                <div key={model} className="p-3 rounded-lg bg-black/20 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-slate-200 text-sm">{name}</span>
                    <Badge variant={variant}>{model.toUpperCase()}</Badge>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                    {['r2', 'rmse', 'mae'].map(m => (
                      <div key={m} className="text-center">
                        <div className="text-slate-500 uppercase mb-0.5">{m}</div>
                        <div className={`font-semibold ${metric === m ? 'text-slate-100' : 'text-slate-400'}`}>
                          {metrics[m]?.toFixed(4) ?? '—'}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </Card>
      </div>
    </section>
  )
}
