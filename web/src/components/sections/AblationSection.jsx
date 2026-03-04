import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { AblationChart } from '../charts/AblationChart'
import { Badge } from '../ui/Badge'

// ablation_results.json: { conditions: [{ ablation, label, mean_r2, delta_r2, is_baseline }] }
export function AblationSection() {
  const { data: ablationData } = useData('ablation_results.json')

  const conditions = ablationData?.conditions ?? []
  const baseline = conditions.find(d => d.is_baseline)
  const rows = [...conditions.filter(d => !d.is_baseline)].sort((a, b) => a.delta_r2 - b.delta_r2)
  const worstComp = rows[0] ?? null

  return (
    <section id="ablation" className="py-16 scroll-mt-6">
      <SectionHeader
        title="Ablation Study"
        subtitle="We systematically remove each signal-processing component and retrain the RF model to quantify each component's contribution to predictive accuracy. ΔR² measures the change in mean test R² relative to the full pipeline."
      />

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Component Contribution (ΔR²)</h3>
          <AblationChart ablationData={ablationData} height={300} />
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Ablation Results Table</h3>
          {baseline && (
            <div className="flex items-center gap-3 mb-3 p-2 rounded-lg bg-primary/10 border border-primary/30">
              <span className="text-xs text-slate-400">Full pipeline R²:</span>
              <span className="font-mono text-sm font-bold text-primary">{baseline.mean_r2?.toFixed(4)}</span>
            </div>
          )}
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-border">
                  <th className="text-left py-2 pr-3">Removed Component</th>
                  <th className="text-right py-2 px-2">Ablated R²</th>
                  <th className="text-right py-2 pl-2">ΔR²</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={i} className="border-b border-border/50 hover:bg-white/5">
                    <td className="py-2 pr-3 text-slate-300">{row.label}</td>
                    <td className="py-2 px-2 text-right font-mono text-slate-400">{row.mean_r2?.toFixed(4)}</td>
                    <td className="py-2 pl-2 text-right font-mono">
                      <span className={row.delta_r2 < -0.05 ? 'text-danger' : row.delta_r2 < 0 ? 'text-warning' : 'text-success'}>
                        {row.delta_r2 >= 0 ? '+' : ''}{row.delta_r2?.toFixed(4)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {worstComp && (
            <div className="mt-4 pt-4 border-t border-border text-xs text-slate-400">
              <span className="text-slate-300 font-medium">Most critical: </span>
              <Badge variant="danger">{worstComp.label}</Badge>
              <span className="ml-2 font-mono text-danger">{worstComp.delta_r2?.toFixed(4)}</span>
            </div>
          )}
        </Card>
      </div>

      <div className="callout">
        <strong className="text-slate-200">Key finding:</strong> The component with the largest negative
        ΔR² is the most critical for predictive accuracy. Positive ΔR² values indicate that removing
        that component actually slightly improved the model — suggesting those components add noise
        that the RF can overfit to.
      </div>
    </section>
  )
}
