import { useState, useEffect } from 'react'

const BASE = import.meta.env.BASE_URL  // '/Techin513_Final/' in prod, '/' in dev

const cache = {}

export function useData(filename) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (cache[filename]) {
      setData(cache[filename])
      setLoading(false)
      return
    }
    const url = `${BASE}data/${filename}`
    fetch(url)
      .then(r => { if (!r.ok) throw new Error(`${r.status} ${url}`); return r.json() })
      .then(d => { cache[filename] = d; setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [filename])

  return { data, loading, error }
}
