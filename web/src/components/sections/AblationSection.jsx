import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { AblationChart } from '../charts/AblationChart'
import { Badge } from '../ui/Badge'

export function AblationSection() {
  const { data: ablationData } = useData('ablation_results.json')

  const worstComp = ablationData
    ? [...ablationData].sort((a, b) => a.delta_r2 - b.delta_r2)[0]
    : null

  return (
    <section id="ablation" className="py-16 scroll-mt-6">
      <SectionHeader
        title="Ablation Study"
        subtitle="We systematically remove each data-generation component and retrain the RF model to quantify each component's contribution to predictive accuracy. ΔR² measures the drop in test R² relative to the full model."
      />

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Component Importance (ΔR²)</h3>
          <AblationChart ablationData={ablationData} height={300} />
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Ablation Results Table</h3>
          {ablationData && (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-500 border-b border-border">
                    <th className="text-left py-2 pr-3">Removed Component</th>
                    <th className="text-right py-2 px-2">Full R²</th>
                    <th className="text-right py-2 px-2">Ablated R²</th>
                    <th className="text-right py-2 pl-2">ΔR²</th>
                  </tr>
                </thead>
                <tbody>
                  {[...ablationData].sort((a, b) => a.delta_r2 - b.delta_r2).map((row, i) => (
                    <tr key={i} className="border-b border-border/50 hover:bg-white/5">
                      <td className="py-2 pr-3 font-mono text-slate-300">
                        {row.removed_component.replace(/_/g, ' ')}
                      </td>
                      <td className="py-2 px-2 text-right font-mono text-slate-400">
                        {row.full_r2?.toFixed(4)}
                      </td>
                      <td className="py-2 px-2 text-right font-mono text-slate-400">
                        {row.ablated_r2?.toFixed(4)}
                      </td>
                      <td className="py-2 pl-2 text-right font-mono">
                        <span className={row.delta_r2 < -0.05 ? 'text-danger' : row.delta_r2 < -0.01 ? 'text-warning' : 'text-success'}>
                          {row.delta_r2?.toFixed(4)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {worstComp && (
            <div className="mt-4 pt-4 border-t border-border">
              <div className="text-xs text-slate-400">
                <span className="text-slate-300 font-medium">Most critical component: </span>
                <Badge variant="danger">
                  {worstComp.removed_component.replace(/_/g, ' ')}
                </Badge>
                <span className="ml-2 font-mono text-danger">ΔR² = {worstComp.delta_r2?.toFixed(4)}</span>
              </div>
            </div>
          )}
        </Card>
      </div>

      <div className="callout">
        <strong className="text-slate-200">Key finding:</strong> Removing Poisson noise events causes
        the largest drop in predictive accuracy, confirming that discrete disturbance events are the
        primary discriminator between sleep quality classes. Circadian drift and pink noise contribute
        less but still improve model generalization.
      </div>
    </section>
  )
}
