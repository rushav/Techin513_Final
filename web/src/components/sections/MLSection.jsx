import { useState } from 'react'
import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { ConfusionMatrix } from '../charts/ConfusionMatrix'
import { ROCChart } from '../charts/ROCChart'

// ml_summary.json keys: rf_test_mean_r2, clf_accuracy, clf_macro_f1, per_label: {label: {test_r2, test_rmse, ...}}, ridge_mean_r2
// classification_report.json: { Good: {precision, recall, f1-score}, ... }

export function MLSection() {
  const { data: mlSummary } = useData('ml_summary.json')
  const { data: cm } = useData('confusion_matrix.json')
  const { data: roc } = useData('roc_curves.json')
  const { data: clsReport } = useData('classification_report.json')
  const [activeTab, setActiveTab] = useState('confusion')

  const rfR2 = mlSummary?.rf_test_mean_r2
  const ridgeR2 = mlSummary?.ridge_mean_r2
  const accuracy = mlSummary?.clf_accuracy
  const macroF1 = mlSummary?.clf_macro_f1

  return (
    <section id="ml" className="py-16 scroll-mt-6">
      <SectionHeader
        title="ML Models"
        subtitle="We train a Random Forest regressor (multi-output) to predict sleep quality labels from environmental features, and a Random Forest classifier for quality class prediction. A Ridge regression baseline is included. All models use 80/20 train/test split, seed=42, 5-fold CV."
      />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {[
          { label: 'RF Mean R²', value: rfR2?.toFixed(3), variant: 'primary' },
          { label: 'Ridge Mean R²', value: ridgeR2?.toFixed(3), variant: 'secondary' },
          { label: 'Classifier Accuracy', value: accuracy ? `${(accuracy * 100).toFixed(1)}%` : null, variant: 'success' },
          { label: 'Macro F1', value: macroF1?.toFixed(3), variant: 'warning' },
        ].map(({ label, value, variant }) => (
          <Card key={label} className="text-center py-4">
            <Badge variant={variant}>{value ?? '—'}</Badge>
            <div className="text-xs text-slate-500 mt-2 uppercase tracking-wider">{label}</div>
          </Card>
        ))}
      </div>

      <div className="flex gap-2 mb-4">
        <button className={`tab-btn ${activeTab === 'confusion' ? 'active' : ''}`} onClick={() => setActiveTab('confusion')}>
          Confusion Matrix
        </button>
        <button className={`tab-btn ${activeTab === 'roc' ? 'active' : ''}`} onClick={() => setActiveTab('roc')}>
          ROC Curves
        </button>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          {activeTab === 'confusion' ? (
            <>
              <h3 className="text-sm font-semibold text-slate-300 mb-3">Confusion Matrix (RF Classifier)</h3>
              <ConfusionMatrix matrix={cm?.matrix_raw} height={300} />
            </>
          ) : (
            <>
              <h3 className="text-sm font-semibold text-slate-300 mb-3">ROC Curves (one-vs-rest)</h3>
              <ROCChart rocData={roc} height={300} />
            </>
          )}
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Per-Label R² (RF Regressor)</h3>
          <div className="space-y-2 text-sm mb-4">
            {Object.entries(mlSummary?.per_label ?? {}).map(([key, v]) => (
              <div key={key} className="flex items-center gap-3 p-3 rounded-lg bg-black/20">
                <div className="w-36 font-medium text-slate-300 text-xs">{v.display ?? key}</div>
                <div className="flex-1 bg-border rounded-full h-1.5">
                  <div className="h-1.5 rounded-full bg-primary" style={{ width: `${Math.max(0, (v.test_r2 ?? 0) * 100).toFixed(1)}%` }} />
                </div>
                <div className="font-mono text-xs text-slate-300 w-12 text-right">{v.test_r2?.toFixed(3)}</div>
              </div>
            ))}
          </div>

          {clsReport && (
            <>
              <h3 className="text-sm font-semibold text-slate-300 mb-2 pt-2 border-t border-border">Classification Report</h3>
              <div className="space-y-1.5">
                {['Good', 'Moderate', 'Poor'].map((cls, i) => {
                  const r = clsReport[cls]
                  if (!r) return null
                  const colors = ['#4ade80', '#fb923c', '#f87171']
                  return (
                    <div key={cls} className="flex items-center gap-3 text-xs font-mono text-slate-400">
                      <div className="w-2 h-2 rounded-full shrink-0" style={{ background: colors[i] }} />
                      <div className="w-16 text-slate-300">{cls}</div>
                      <span>P:<strong className="text-slate-200 ml-1">{r.precision?.toFixed(2)}</strong></span>
                      <span>R:<strong className="text-slate-200 ml-1">{r.recall?.toFixed(2)}</strong></span>
                      <span>F1:<strong className="text-slate-200 ml-1">{r['f1-score']?.toFixed(2)}</strong></span>
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </Card>
      </div>
    </section>
  )
}
