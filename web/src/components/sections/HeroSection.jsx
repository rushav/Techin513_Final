import { motion } from 'framer-motion'
import { useData } from '../../hooks/useData'
import { StatCounter } from '../ui/StatCounter'
import { Badge } from '../ui/Badge'

// dataset_stats.json: { n_sessions, n_features, n_samples_per_session, session_duration_h, sample_rate_min, ... }
export function HeroSection() {
  const { data } = useData('dataset_stats.json')

  // sample_rate_min is minutes-per-sample; convert to Hz = 1 / (sample_rate_min * 60)
  const sampleRateHz = data?.sample_rate_min ? (1 / (data.sample_rate_min * 60)).toFixed(3) : null

  return (
    <section id="hero" className="min-h-[60vh] flex flex-col justify-center py-20">
      <motion.div
        initial={{ opacity: 0, y: 32 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: 'easeOut' }}
      >
        <div className="flex flex-wrap gap-2 mb-6">
          <Badge variant="primary">TECHIN 513</Badge>
          <Badge variant="secondary">Synthetic Sleep Data</Badge>
          <Badge variant="neutral">seed = 42</Badge>
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-slate-100 leading-tight mb-4">
          Sleep Environment<br/>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">
            Signal Processing
          </span>
        </h1>
        <p className="text-slate-400 text-lg max-w-2xl mb-10 leading-relaxed">
          We generate synthetic bedroom sensor data — temperature, light, humidity, and noise —
          apply signal processing, validate statistical properties, and train ML models to
          predict sleep quality from environmental features.
        </p>

        {data && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <StatCounter value={data.n_sessions} label="Sessions" color="#6366f1" />
            <StatCounter value={data.n_features} label="Features" color="#22d3ee" />
            <StatCounter value={data.n_samples_per_session} label="Samples / Session" color="#4ade80" />
            <StatCounter value={data.session_duration_h} label="Hours / Session" color="#fb923c" decimals={0} />
          </div>
        )}
      </motion.div>
    </section>
  )
}
