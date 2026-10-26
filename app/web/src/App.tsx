import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Day, Index } from './api'
import { fetchDay, fetchIndex } from './api'
import { frameIndexAt, hhmm, predMap } from './model'
import { HubMap } from './components/HubMap'
import { WaveChart } from './components/WaveChart'
import { BriefingPanel, ConnectionsPanel, FlightPanel, Legend } from './components/Panels'
import { Tooltip } from './components/Tooltip'
import type { Tip } from './components/Tooltip'

const MODEL_LABEL: Record<string, string> = { gnn: 'GNN', lgbm: 'LightGBM' }
const SPEED = 18 // simulated minutes per second while playing

/** Callback ref + ResizeObserver: works for elements that mount after the first render. */
function useWidth<T extends HTMLElement>() {
  const ro = useRef<ResizeObserver | null>(null)
  const [w, setW] = useState(0)
  const ref = useCallback((el: T | null) => {
    ro.current?.disconnect()
    if (!el) return
    setW(Math.floor(el.getBoundingClientRect().width))
    ro.current = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)))
    ro.current.observe(el)
  }, [])
  return [ref, w] as const
}

function Metrics({ index }: { index: Index }) {
  const m = index.metrics?.models
  if (!m) return null
  const models = ['lgbm', 'gnn'].filter((k) => m[k])
  return (
    <section className="card metrics">
      <header className="card-h">
        <h2>Test-set accuracy</h2>
        <span className="muted">{index.dataset} data · lower CRPS is better · Brier skill vs base rate</span>
      </header>
      <table>
        <thead>
          <tr><th>Model</th>{['1', '3', '6'].map((h) => <th key={h}>CRPS dep {h}h</th>)}<th>cov90 1h</th>{['1', '3', '6'].map((h) => <th key={`b${h}`}>Miss BSS {h}h</th>)}</tr>
        </thead>
        <tbody>
          {models.map((k) => (
            <tr key={k}>
              <td>{MODEL_LABEL[k]}</td>
              {['1', '3', '6'].map((h) => <td key={h}>{m[k].horizons[h].dep.crps.toFixed(2)}</td>)}
              <td>{(m[k].horizons['1'].dep.cov90 * 100).toFixed(0)}%</td>
              {['1', '3', '6'].map((h) => <td key={`b${h}`}>{m[k].connections[h].brier_skill.toFixed(2)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

export default function App() {
  const [index, setIndex] = useState<Index | null>(null)
  const [dayKey, setDayKey] = useState<string>('')
  const [model, setModel] = useState<string>('')
  const [day, setDay] = useState<Day | null>(null)
  const params = useMemo(() => new URLSearchParams(window.location.search), [])
  const [t, setT] = useState(() => {
    const m = /^(\d{1,2}):(\d{2})$/.exec(params.get('t') ?? '')
    return m ? Number(m[1]) * 60 + Number(m[2]) : 6 * 60
  })
  const [playing, setPlaying] = useState(false)
  const [selected, setSelected] = useState<number | null>(null)
  const [highlight, setHighlight] = useState<Set<number>>(new Set())
  const [hindsight, setHindsight] = useState(false)
  const [tip, setTip] = useState<Tip | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [mapRef, mapW] = useWidth<HTMLDivElement>()
  const [waveRef, waveW] = useWidth<HTMLDivElement>()

  useEffect(() => {
    fetchIndex()
      .then((ix) => {
        setIndex(ix)
        const d = params.get('day'), m = params.get('model'), f = params.get('flight')
        setDayKey(d && ix.days.includes(d) ? d : ix.days[0])
        setModel(m && ix.models.includes(m) ? m : ix.models.includes('gnn') ? 'gnn' : ix.models[0])
        if (f) setSelected(Number(f))
      })
      .catch((e) => setErr(String(e)))
  }, [params])

  useEffect(() => {
    if (!dayKey || !model) return
    fetchDay(dayKey, model).then(setDay).catch((e) => setErr(String(e)))
  }, [dayKey, model])

  const tMin = day?.frames[0].t ?? 240
  const tMax = (day?.frames[day.frames.length - 1].t ?? 1410) + 29

  useEffect(() => {
    if (!playing) return
    let raf = 0
    let last = performance.now()
    const step = (now: number) => {
      const dt = (now - last) / 1000
      last = now
      setT((v) => {
        const nv = v + dt * SPEED
        if (nv >= tMax) { setPlaying(false); return tMax }
        return nv
      })
      raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [playing, tMax])

  const flights = useMemo(() => new Map((day?.flights ?? []).map((f) => [f.id, f])), [day])
  const frameIdx = day ? frameIndexAt(day, t) : 0
  const frame = day?.frames[frameIdx]
  const preds = useMemo(() => (frame ? predMap(frame) : new Map()), [frame])

  const select = (id: number) => { setSelected(id); setHighlight(new Set()) }
  const pickConn = (i: number, o: number) => { setSelected(o); setHighlight(new Set([i, o])) }

  if (err) return <main className="app"><p className="err pad">Cannot load data: {err}. Run <code>uv run ripple demo</code> and <code>uv run ripple serve</code>.</p></main>
  if (!index || !day || !frame) return <main className="app"><p className="muted pad">Loading replay…</p></main>

  return (
    <main className="app" onMouseLeave={() => setTip(null)}>
      <header className="top">
        <div className="brand">
          <h1>Ripple</h1>
          <p className="muted">Delay propagation and connection risk at the WAW hub</p>
        </div>
        <div className="controls">
          <label>Day
            <select value={dayKey} onChange={(e) => setDayKey(e.target.value)}>
              {index.days.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </label>
          <div className="seg" role="group" aria-label="Model">
            {index.models.map((m) => (
              <button key={m} className={m === model ? 'on' : ''} onClick={() => setModel(m)}>{MODEL_LABEL[m] ?? m}</button>
            ))}
          </div>
          <label className="check"><input type="checkbox" checked={hindsight} onChange={(e) => setHindsight(e.target.checked)} /> Show outcome</label>
        </div>
      </header>
      {index.dataset === 'synthetic' && (
        <p className="banner">Synthetic replay: a simulated LOT-like network used until real scheduled times are collected. It is not real LOT data.</p>
      )}
      <div className="timebar">
        <button className="btn play" onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause' : 'Play'}>{playing ? '❚❚' : '▶'}</button>
        <input type="range" min={tMin} max={tMax} step={1} value={t} aria-label="Time of day"
          onChange={(e) => { setPlaying(false); setT(Number(e.target.value)) }} />
        <span className="clock">{hhmm(t)} <span className="muted">local · forecast issued {hhmm(frame.t)}</span></span>
      </div>
      <div className="grid">
        <div className="col-main">
          <section className="card">
            <header className="card-h">
              <h2>Network now</h2>
              <span className="muted">aircraft in the air, coloured by departure delay · rings = late departures</span>
            </header>
            <div ref={mapRef} className="map-wrap">
              {mapW > 0 && <HubMap width={mapW} height={Math.round(Math.min(520, mapW * 0.62))} airports={index.airports} day={day} t={t}
                preds={preds} selected={selected} highlight={highlight} onSelect={select} onTip={setTip} />}
            </div>
            <Legend />
          </section>
          <section className="card">
            <header className="card-h">
              <h2>WAW banks</h2>
              <span className="muted">each dot is a flight; filled = observed, hollow = forecast median</span>
            </header>
            <div ref={waveRef} className="wave-wrap">
              {waveW > 0 && <WaveChart width={waveW} day={day} t={t} preds={preds} selected={selected} highlight={highlight} onSelect={select} onTip={setTip} />}
            </div>
          </section>
          <Metrics index={index} />
        </div>
        <div className="col-side">
          <ConnectionsPanel day={day} frame={frame} flights={flights} onPick={pickConn} active={selected} />
          <FlightPanel day={day} flight={selected !== null ? flights.get(selected) ?? null : null} t={t} preds={preds}
            flights={flights} hindsight={hindsight} onSelect={select} />
          <BriefingPanel day={day} frameIdx={frameIdx} t={t} />
        </div>
      </div>
      <Tooltip tip={tip} />
    </main>
  )
}
