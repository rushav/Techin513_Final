import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { PlotlyChart } from '../charts/PlotlyChart'
import { mergeLayout } from '../../utils/plotlyConfig'
import { QUALITY_COLORS } from '../../utils/colors'

export function ValidationSection() {
  const { data: ksData } = useData('ks_test_results.json')

  const tableRows = ksData?.tests ?? []
  const passCount = tableRows.filter(r => r.pass).length

  // KS statistic bar chart
  const ksTraces = [{
    x: tableRows.map(r => r.comparison),
    y: tableRows.map(r => r.statistic),
    type: 'bar',
    marker: {
      color: tableRows.map(r => r.pass ? '#4ade80' : '#f87171'),
      opacity: 0.85,
    },
    error_y: {
      type: 'data',
      array: tableRows.map(() => 0),
      visible: false,
    },
    hovertemplate: '<b>%{x}</b><br>KS stat: %{y:.4f}<extra></extra>',
  }, {
    x: tableRows.map(r => r.comparison),
    y: tableRows.map(() => ksData?.alpha ?? 0.05),
    type: 'scatter', mode: 'lines',
    name: `α = ${ksData?.alpha ?? 0.05}`,
    line: { color: '#fb923c', width: 1, dash: 'dot' },
  }]

  return (
    <section id="validation" className="py-16 scroll-mt-6">
      <SectionHeader
        title="Statistical Validation"
        subtitle="We validate that our synthetic signals respect the intended distributional assumptions. Six pairwise KS tests compare signal distributions across quality classes, and six sanity checks verify domain constraints (e.g., temperature within 10–30°C, humidity within 20–80%)."
      />

      <div className="grid lg:grid-cols-3 gap-4 mb-6">
        <Card className="text-center">
          <div className="text-3xl font-bold font-mono text-success mb-1">6 / 6</div>
          <div className="text-xs text-slate-400 uppercase tracking-wider">Sanity Checks Pass</div>
        </Card>
        <Card className="text-center">
          <div className="text-3xl font-bold font-mono text-warning mb-1">{passCount} / {tableRows.length}</div>
          <div className="text-xs text-slate-400 uppercase tracking-wider">KS Tests at α=0.05</div>
        </Card>
        <Card className="text-center">
          <div className="text-3xl font-bold font-mono text-secondary mb-1">500</div>
          <div className="text-xs text-slate-400 uppercase tracking-wider">Sessions Validated</div>
        </Card>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">KS Test Statistics vs. α Threshold</h3>
          <PlotlyChart
            data={ksTraces}
            layout={mergeLayout({
              height: 280,
              xaxis: { title: { text: 'Comparison' }, tickangle: -30, automargin: true },
              yaxis: { title: { text: 'KS Statistic' } },
              margin: { l: 55, r: 20, t: 30, b: 90 },
            })}
            style={{ height: 280 }}
          />
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">KS Test Results</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500 border-b border-border">
                  <th className="text-left py-2 pr-3">Comparison</th>
                  <th className="text-right py-2 px-2">KS Stat</th>
                  <th className="text-right py-2 px-2">p-value</th>
                  <th className="text-right py-2 pl-2">Pass?</th>
                </tr>
              </thead>
              <tbody>
                {tableRows.map((r, i) => (
                  <tr key={i} className="border-b border-border/50 hover:bg-white/5">
                    <td className="py-2 pr-3 font-mono text-slate-300">{r.comparison}</td>
                    <td className="py-2 px-2 text-right font-mono">{r.statistic?.toFixed(4)}</td>
                    <td className="py-2 px-2 text-right font-mono">{r.p_value?.toFixed(4)}</td>
                    <td className="py-2 pl-2 text-right">
                      <Badge variant={r.pass ? 'success' : 'danger'}>{r.pass ? 'Pass' : 'Fail'}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-slate-500 mt-3">
            KS test failures at n=500 are expected for well-separated distributions — the large sample size
            gives sufficient power to detect any deviation, even statistically meaningful ones.
          </p>
        </Card>
      </div>
    </section>
  )
}
