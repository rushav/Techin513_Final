import { useEffect, useRef, useState } from 'react'
import { useInView } from '../../hooks/useInView'

export function StatCounter({ value, label, suffix = '', prefix = '', decimals = 0, color = '#6366f1' }) {
  const [ref, inView] = useInView()
  const [display, setDisplay] = useState(0)
  const raf = useRef(null)

  useEffect(() => {
    if (!inView) return
    const start = 0
    const end = parseFloat(value)
    const duration = 1200
    const t0 = performance.now()
    const step = (now) => {
      const progress = Math.min((now - t0) / duration, 1)
      const ease = 1 - Math.pow(1 - progress, 3)
      setDisplay(+(start + (end - start) * ease).toFixed(decimals))
      if (progress < 1) raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [inView, value, decimals])

  return (
    <div ref={ref} className="text-center">
      <div className="text-3xl font-bold font-mono" style={{ color }}>
        {prefix}{display.toLocaleString()}{suffix}
      </div>
      <div className="text-xs text-slate-400 mt-1 uppercase tracking-wider">{label}</div>
    </div>
  )
}
