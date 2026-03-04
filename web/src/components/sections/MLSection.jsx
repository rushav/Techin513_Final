import { useState } from 'react'
import { useData } from '../../hooks/useData'
import { Card, SectionHeader } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { ConfusionMatrix } from '../charts/ConfusionMatrix'
import { ROCChart } from '../charts/ROCChart'

export function MLSection() {
  const { data: mlSummary } = useData('ml_summary.json')
  const { data: cm } = useData('confusion_matrix.json')
  const { data: roc } = useData('roc_curves.json')
  const [activeTab, setActiveTab] = useState('confusion')

  const rf = mlSummary?.rf
  const ridge = mlSummary?.ridge

  return (
    <section id="ml" className="py-16 scroll-mt-6">
      <SectionHeader
        title="ML Models"
        subtitle="We train a Random Forest regressor to predict sleep quality score (continuous) and a Random Forest classifier to predict quality class (good/moderate/poor). A Ridge regression baseline is included for comparison. All models use an 80/20 train/test split with seed=42."
      />

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {rf && [
          { label: 'RF R²', value: rf.r2?.toFixed(3), variant: 'primary' },
          { label: 'RF RMSE', value: rf.rmse?.toFixed(3), variant: 'secondary' },
          { label: 'Ridge R²', value: ridge?.r2?.toFixed(3), variant: 'neutral' },
          { label: 'RF Accuracy', value: `${(rf.accuracy * 100)?.toFixed(1)}%`, variant: 'success' },
        ].map(({ label, value, variant }) => (
          <Card key={label} className="text-center py-4">
            <div className="text-2xl font-bold font-mono mb-1">
              <Badge variant={variant}>{value ?? '—'}</Badge>
            </div>
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
              <ConfusionMatrix matrix={cm?.matrix} height={300} />
            </>
          ) : (
            <>
              <h3 className="text-sm font-semibold text-slate-300 mb-3">ROC Curves (one-vs-rest)</h3>
              <ROCChart rocData={roc} height={300} />
            </>
          )}
        </Card>

        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Classification Report</h3>
          <div className="space-y-2 text-sm">
            {['good', 'moderate', 'poor'].map((cls, i) => {
              const report = mlSummary?.classification_report?.[cls]
              const colors = ['#4ade80', '#fb923c', '#f87171']
              if (!report) return null
              return (
                <div key={cls} className="flex items-center gap-3 p-3 rounded-lg bg-black/20">
                  <div className="w-2 h-2 rounded-full shrink-0" style={{ background: colors[i] }} />
                  <div className="w-20 font-medium text-slate-300 capitalize">{cls}</div>
                  <div className="flex gap-3 text-xs font-mono text-slate-400">
                    <span>P: <strong className="text-slate-200">{report.precision?.toFixed(2)}</strong></span>
                    <span>R: <strong className="text-slate-200">{report.recall?.toFixed(2)}</strong></span>
                    <span>F1: <strong className="text-slate-200">{report['f1-score']?.toFixed(2)}</strong></span>
                  </div>
                </div>
              )
            })}
          </div>

          <div className="mt-4 pt-4 border-t border-border">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Hyperparameters</h4>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono text-slate-400">
              <span>n_estimators</span><span className="text-secondary">200</span>
              <span>max_depth</span><span className="text-secondary">None</span>
              <span>min_samples_leaf</span><span className="text-secondary">2</span>
              <span>random_state</span><span className="text-secondary">42</span>
            </div>
          </div>
        </Card>
      </div>
    </section>
  )
}
