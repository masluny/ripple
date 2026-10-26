import { useState } from 'react'
import type { Day, Flight, Frame, Pred } from '../api'
import { fetchBriefing } from '../api'
import { CLASS_LABEL, GROUP_LABEL, currentDelay, delayClass, hhmm, status } from '../model'

const fmt = (v: number) => `${v >= 0 ? '+' : ''}${Math.round(v)}`

export function ConnectionsPanel({ day, frame, flights, onPick, active }: {
  day: Day; frame: Frame; flights: Map<number, Flight>; onPick: (inId: number, outId: number) => void; active: number | null
}) {
  const rows = frame.conns.filter((c) => c[3] >= 0.05).slice(0, 12)
  return (
    <section className="card">
      <header className="card-h">
        <h2>Connections at risk</h2>
        <span className="muted">WAW · next 6 h · P(miss) with MCT 35 min</span>
      </header>
      {rows.length === 0 && <p className="muted pad">No connection above 5% risk at this time.</p>}
      <ul className="conn-list">
        {rows.map(([i, o, slack, p]) => {
          const a = flights.get(i)!, b = flights.get(o)!
          const sev = p >= 0.5 ? 'critical' : p >= 0.25 ? 'serious' : 'warning'
          return (
            <li key={`${i}-${o}`}>
              <button className={`conn ${active === o ? 'on' : ''}`} onClick={() => onPick(i, o)}>
                <span className="conn-flights">
                  <b>{a.no}</b> <span className="muted">{a.dep}</span> → <b>{b.no}</b> <span className="muted">{b.arr}</span>
                </span>
                <span className="conn-meta muted">{slack} min · STD {hhmm(b.sd)}</span>
                <span className="pbar" aria-hidden><span className={`pfill c-${sev}`} style={{ width: `${p * 100}%` }} /></span>
                <span className="pval">{Math.round(p * 100)}%</span>
              </button>
            </li>
          )
        })}
      </ul>
      <p className="foot muted">Connections are every schedule-feasible arrival→departure pair (no passenger data). {day.model.toUpperCase()} marginals, sampled independently.</p>
    </section>
  )
}

function Fan({ q, label, actual, max }: { q: number[]; label: string; actual: number | null; max: number }) {
  const lo = -30
  const W = 260
  const x = (v: number) => 8 + ((Math.max(lo, Math.min(max, v)) - lo) / (max - lo)) * (W - 16)
  return (
    <div className="fan">
      <span className="fan-label">{label}</span>
      <svg width={W} height={30} role="img" aria-label={`${label}: median ${Math.round(q[2])} min, 90% range ${Math.round(q[0])} to ${Math.round(q[4])}`}>
        <line x1={x(0)} x2={x(0)} y1={2} y2={28} className="zero" />
        <line x1={x(q[0])} x2={x(q[4])} y1={15} y2={15} className="fan-90" />
        <rect x={x(q[1])} y={9} width={Math.max(2, x(q[3]) - x(q[1]))} height={12} rx={3} className={`fan-50 c-${delayClass(q[2])}`} />
        <line x1={x(q[2])} x2={x(q[2])} y1={6} y2={24} className="fan-med" />
        {actual !== null && <path d={`M${x(actual)},4 l4,5 l-4,5 l-4,-5z`} className="actual" />}
      </svg>
      <span className="fan-val">{fmt(q[2])} <span className="muted">({Math.round(q[0])}…{Math.round(q[4])})</span></span>
    </div>
  )
}

export function FlightPanel({ day, flight, t, preds, flights, hindsight, onSelect }: {
  day: Day; flight: Flight | null; t: number; preds: Map<number, Pred>; flights: Map<number, Flight>; hindsight: boolean; onSelect: (id: number) => void
}) {
  if (!flight) {
    const watch = [...preds.values()].sort((a, b) => b.dep[2] - a.dep[2]).slice(0, 6)
    return (
      <section className="card">
        <header className="card-h"><h2>Watch list</h2><span className="muted">largest forecast departure delays</span></header>
        <ol className="chain">
          {watch.map((p) => {
            const f = flights.get(p.id)!
            return (
              <li key={p.id}>
                <button onClick={() => onSelect(p.id)}>
                  <span className={`dot c-${delayClass(p.dep[2])}`} aria-hidden />
                  <span>{hhmm(f.sd)} {f.no} {f.dep}→{f.arr}</span>
                  <span>{fmt(p.dep[2])} <span className="muted">min</span></span>
                </button>
              </li>
            )
          })}
        </ol>
        <p className="foot muted">Pick any flight on the map, in the bank chart or in the connection list for its forecast and drivers.</p>
      </section>
    )
  }
  const p = preds.get(flight.id)
  const st = status(flight, t)
  const cd = currentDelay(flight, t, preds)
  const chain = day.flights.filter((f) => f.tail === flight.tail && f.sd >= 0 && f.sd < 1440).sort((a, b) => a.sd - b.sd)
  const max = p ? Math.max(60, p.dep[4], p.arr[4]) + 10 : 60
  const maxDelta = p ? Math.max(5, ...p.deltas.map(Math.abs)) : 5
  return (
    <section className="card">
      <header className="card-h">
        <h2>{flight.no} <span className="muted">{flight.dep} → {flight.arr}</span></h2>
        <span className={`chip c-${delayClass(cd.value)}`}>{st}</span>
      </header>
      <p className="muted pad-x">{flight.tail} · {flight.type} · STD {hhmm(flight.sd)} · STA {hhmm(flight.sa)}</p>
      {p ? (
        <>
          <h3>Forecast at {hhmm(t)}</h3>
          <Fan q={p.dep} label="Departure" max={max} actual={hindsight && flight.ad !== null ? flight.ad - flight.sd : null} />
          <Fan q={p.arr} label="Arrival" max={max} actual={hindsight && flight.aa !== null ? flight.aa - flight.sa : null} />
          <p className="legend-line muted">bar = 50% interval, line = 90%, tick = median{hindsight ? ', ◆ = what happened' : ''}</p>
          <h3>Drivers of the forecast</h3>
          <div className="drivers">
            {day.groups.map((g, i) => (
              <div key={g} className="driver">
                <span>{GROUP_LABEL[g] ?? g}</span>
                <span className="dbar"><span className={`dfill s${i + 1}`} style={{ width: `${(Math.max(0, p.deltas[i]) / maxDelta) * 100}%` }} /></span>
                <span className="dval">{fmt(p.deltas[i])} min</span>
              </div>
            ))}
          </div>
          <p className="cause">{p.text || 'No single dominant driver.'}</p>
          {hindsight && flight.truth && <p className="muted pad-x">Simulator ground truth: <b>{flight.truth}</b></p>}
        </>
      ) : (
        <p className="muted pad">
          {cd.kind.startsWith('actual')
            ? `Observed ${cd.leg === 'arr' ? 'arrival' : 'departure'} delay ${fmt(cd.value!)} min.`
            : 'Not forecast at this time (more than 6.5 h ahead).'}
        </p>
      )}
      <h3>Aircraft {flight.tail} today</h3>
      <ol className="chain">
        {chain.map((f) => {
          const c = currentDelay(f, t, preds)
          return (
            <li key={f.id}>
              <button className={f.id === flight.id ? 'on' : ''} onClick={() => onSelect(f.id)}>
                <span className={`dot c-${delayClass(c.value)}`} aria-hidden />
                <span>{hhmm(f.sd)} {f.no} {f.dep}→{f.arr}</span>
                <span className="muted">{c.value === null ? '' : `${c.kind === 'forecast' ? '~' : ''}${fmt(c.value)}`}</span>
              </button>
            </li>
          )
        })}
      </ol>
      {flight.prev !== null && flights.get(flight.prev) === undefined && <p className="muted pad-x">Previous leg not in view.</p>}
    </section>
  )
}

function Markdown({ text }: { text: string }) {
  const inline = (s: string) =>
    s.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g).map((part, i) =>
      part.startsWith('**') ? <b key={i}>{part.slice(2, -2)}</b>
        : part.startsWith('*') && part.length > 2 ? <i key={i}>{part.slice(1, -1)}</i> : part)
  const out: React.ReactNode[] = []
  let list: string[] = []
  const flush = () => {
    if (list.length) out.push(<ul key={`u${out.length}`}>{list.map((l, i) => <li key={i}>{inline(l)}</li>)}</ul>)
    list = []
  }
  text.split('\n').forEach((line) => {
    const l = line.trim()
    if (/^[-*] /.test(l)) list.push(l.slice(2))
    else {
      flush()
      if (l.startsWith('#')) out.push(<h4 key={out.length}>{inline(l.replace(/^#+\s*/, ''))}</h4>)
      else if (l) out.push(<p key={out.length}>{inline(l)}</p>)
    }
  })
  flush()
  return <div className="md">{out}</div>
}

export function BriefingPanel({ day, frameIdx, t }: { day: Day; frameIdx: number; t: number }) {
  const [state, setState] = useState<{ text: string; source: string; at: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const run = async () => {
    setBusy(true)
    setErr(null)
    try {
      const r = await fetchBriefing(day.day, day.model, frameIdx)
      setState({ ...r, at: hhmm(t) })
    } catch (e) {
      setErr(String(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <section className="card">
      <header className="card-h">
        <h2>Ops briefing</h2>
        <button className="btn" onClick={run} disabled={busy}>{busy ? 'Writing…' : `Brief me at ${hhmm(t)}`}</button>
      </header>
      {err && <p className="pad err">{err}</p>}
      {state && (
        <>
          <Markdown text={state.text} />
          <p className="foot muted">{state.at} · {state.source}. Generated from the forecasts on screen, not checked by a human.</p>
        </>
      )}
      {!state && !err && <p className="muted pad">One paragraph for the duty manager: watch list, connections at risk, options. Uses Claude if an Anthropic key is configured, otherwise a template.</p>}
    </section>
  )
}

export function Legend() {
  return (
    <div className="legend" aria-label="Delay colour legend">
      {(['good', 'warning', 'serious', 'critical', 'unknown'] as const).map((c) => (
        <span key={c}><span className={`dot c-${c}`} aria-hidden /> {CLASS_LABEL[c]}</span>
      ))}
      <span className="muted">hollow = forecast · filled = observed</span>
    </div>
  )
}
