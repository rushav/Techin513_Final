export const DARK_LAYOUT = {
  paper_bgcolor: '#0a0a0f',
  plot_bgcolor:  '#12121a',
  font: { color: '#f1f5f9', family: 'Inter, system-ui, sans-serif', size: 12 },
  xaxis: {
    gridcolor:     '#1e1e2e',
    linecolor:     '#1e1e2e',
    tickcolor:     '#94a3b8',
    zerolinecolor: '#1e1e2e',
    tickfont: { color: '#94a3b8', size: 11 },
    title: { font: { color: '#f1f5f9', size: 12 } },
  },
  yaxis: {
    gridcolor:     '#1e1e2e',
    linecolor:     '#1e1e2e',
    tickcolor:     '#94a3b8',
    zerolinecolor: '#1e1e2e',
    tickfont: { color: '#94a3b8', size: 11 },
    title: { font: { color: '#f1f5f9', size: 12 } },
  },
  legend: {
    bgcolor:     'rgba(18,18,26,0.8)',
    bordercolor: '#1e1e2e',
    borderwidth: 1,
    font: { color: '#f1f5f9', size: 11 },
  },
  hoverlabel: {
    bgcolor:     '#12121a',
    bordercolor: '#6366f1',
    font: { color: '#f1f5f9', family: 'Inter' },
  },
  title: {
    font: { color: '#f1f5f9', size: 14, family: 'Inter' },
    pad: { t: 4 },
  },
  margin: { l: 55, r: 20, t: 45, b: 50 },
  colorway: ['#6366f1','#22d3ee','#4ade80','#fb923c','#f87171','#a78bfa'],
}

export const PLOTLY_CONFIG = {
  responsive: true,
  displayModeBar: false,
  displaylogo: false,
  scrollZoom: false,
}

export function mergeLayout(overrides = {}) {
  const merged = JSON.parse(JSON.stringify(DARK_LAYOUT))
  for (const [k, v] of Object.entries(overrides)) {
    if (typeof v === 'object' && v !== null && !Array.isArray(v) && typeof merged[k] === 'object') {
      merged[k] = { ...merged[k], ...v }
    } else {
      merged[k] = v
    }
  }
  return merged
}
