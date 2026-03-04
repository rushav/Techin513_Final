import { useRef, useEffect } from 'react'
import { PLOTLY_CONFIG } from '../../utils/plotlyConfig'

// Thin wrapper around Plotly — avoids the heavy react-plotly.js bundle overhead.
// Plotly is loaded via the plotly.js-dist-min chunk defined in vite.config.js.
let Plotly = null

async function getPlotly() {
  if (!Plotly) Plotly = (await import('plotly.js-dist-min')).default
  return Plotly
}

export function PlotlyChart({ data, layout, config = {}, style = {}, className = '' }) {
  const divRef = useRef(null)
  const plotRef = useRef(null)

  useEffect(() => {
    let cancelled = false
    getPlotly().then(P => {
      if (cancelled || !divRef.current) return
      if (plotRef.current) {
        P.react(divRef.current, data, layout, { ...PLOTLY_CONFIG, ...config })
      } else {
        P.newPlot(divRef.current, data, layout, { ...PLOTLY_CONFIG, ...config })
        plotRef.current = true
      }
    })
    return () => { cancelled = true }
  }, [data, layout, config])

  useEffect(() => {
    return () => {
      getPlotly().then(P => { if (divRef.current) P.purge(divRef.current) })
    }
  }, [])

  return (
    <div
      ref={divRef}
      className={className}
      style={{ width: '100%', minHeight: 280, ...style }}
    />
  )
}
