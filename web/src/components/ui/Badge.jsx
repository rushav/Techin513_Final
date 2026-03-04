const VARIANTS = {
  primary:   { bg: 'rgba(99,102,241,0.15)',  border: '#6366f1', color: '#a5b4fc' },
  secondary: { bg: 'rgba(34,211,238,0.12)',  border: '#22d3ee', color: '#67e8f9' },
  success:   { bg: 'rgba(74,222,128,0.12)',  border: '#4ade80', color: '#86efac' },
  warning:   { bg: 'rgba(251,146,60,0.12)',  border: '#fb923c', color: '#fdba74' },
  danger:    { bg: 'rgba(248,113,113,0.12)', border: '#f87171', color: '#fca5a5' },
  neutral:   { bg: 'rgba(148,163,184,0.1)',  border: '#475569', color: '#94a3b8' },
}

export function Badge({ children, variant = 'primary', className = '' }) {
  const s = VARIANTS[variant] || VARIANTS.primary
  return (
    <span
      className={`metric-badge ${className}`}
      style={{ background: s.bg, borderColor: s.border, color: s.color }}
    >
      {children}
    </span>
  )
}
