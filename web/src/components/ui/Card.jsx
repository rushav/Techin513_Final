import { motion } from 'framer-motion'
import { useInView } from '../../hooks/useInView'

export function Card({ children, className = '', animate = true }) {
  const [ref, inView] = useInView()

  if (!animate) {
    return <div className={`glow-card ${className}`}>{children}</div>
  }

  return (
    <motion.div
      ref={ref}
      className={`glow-card ${className}`}
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      {children}
    </motion.div>
  )
}

export function SectionHeader({ title, subtitle, id }) {
  const [ref, inView] = useInView()
  return (
    <motion.div
      ref={ref}
      id={id}
      className="mb-8"
      initial={{ opacity: 0, y: 16 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.4 }}
    >
      <h2 className="section-heading">{title}</h2>
      {subtitle && <p className="mt-4 text-slate-400 text-sm leading-relaxed max-w-3xl">{subtitle}</p>}
    </motion.div>
  )
}
