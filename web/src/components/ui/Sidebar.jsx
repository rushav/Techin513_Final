import { useState, useEffect } from 'react'

const NAV = [
  { id: 'hero',       label: 'Overview' },
  { id: 'data-gen',   label: 'Data Generation' },
  { id: 'signals',    label: 'Signal Processing' },
  { id: 'validation', label: 'Validation' },
  { id: 'features',   label: 'Feature Extraction' },
  { id: 'ml',         label: 'ML Models' },
  { id: 'results',    label: 'Results' },
  { id: 'ablation',   label: 'Ablation Study' },
  { id: 'explorer',   label: 'Parameter Explorer' },
]

export function Sidebar() {
  const [active, setActive] = useState('hero')

  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach(e => { if (e.isIntersecting) setActive(e.target.id) })
      },
      { rootMargin: '-40% 0px -55% 0px' }
    )
    NAV.forEach(({ id }) => {
      const el = document.getElementById(id)
      if (el) obs.observe(el)
    })
    return () => obs.disconnect()
  }, [])

  return (
    <aside className="hidden lg:flex flex-col w-52 shrink-0 sticky top-0 h-screen py-10 pl-6 pr-4 overflow-y-auto">
      <div className="mb-8">
        <div className="text-xs font-mono text-slate-500 uppercase tracking-widest mb-1">Techin 513</div>
        <div className="text-sm font-semibold text-slate-200 leading-snug">Sleep Environment<br/>Dashboard</div>
      </div>
      <nav className="flex flex-col gap-1">
        {NAV.map(({ id, label }) => (
          <a
            key={id}
            href={`#${id}`}
            onClick={e => { e.preventDefault(); document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' }) }}
            className={`text-sm px-3 py-2 rounded-lg transition-all duration-200 ${
              active === id
                ? 'text-primary font-medium bg-primary/10 border-l-2 border-primary pl-2.5'
                : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
            }`}
          >
            {label}
          </a>
        ))}
      </nav>
      <div className="mt-auto pt-8 text-xs text-slate-600">
        <div>TECHIN 513 Final</div>
        <div className="font-mono mt-0.5">seed = 42</div>
      </div>
    </aside>
  )
}
