import type { Day, Pred } from '../api'
import { currentDelay, delayClass, hhmm } from '../model'
import type { Tip } from './Tooltip'

interface Props {
  width: number
  day: Day
  t: number
  preds: Map<number, Pred>
  selected: number | null
  highlight: Set<number>
  onSelect: (id: number) => void
  onTip: (tip: Tip | null) => void
}

const T0 = 4 * 60
const T1 = 26 * 60
const BIN = 10
const STEP = 7

/** WAW bank structure: one dot per flight in its scheduled 10-minute slot, arrivals above the axis,
 * departures below, coloured by the observed delay or (for flights still to come) the forecast. */
export function WaveChart({ width, day, t, preds, selected, highlight, onSelect, onTip }: Props) {
  const x = (m: number) => 40 + ((m - T0) / (T1 - T0)) * (width - 52)
  const arr = day.flights.filter((f) => f.arr === 'WAW' && f.sa >= T0 && f.sa < T1)
  const dep = day.flights.filter((f) => f.dep === 'WAW' && f.sd >= T0 && f.sd < T1)
  const stack = (fs: typeof arr, key: 'sa' | 'sd') => {
    const count = new Map<number, number>()
    return fs
      .slice()
      .sort((a, b) => a[key] - b[key])
      .map((f) => {
        const b = Math.floor(f[key] / BIN)
        const k = count.get(b) ?? 0
        count.set(b, k + 1)
        return { f, bx: b * BIN + BIN / 2, k }
      })
  }
  const A = stack(arr, 'sa')
  const D = stack(dep, 'sd')
  const maxA = Math.max(4, ...A.map((d) => d.k + 1))
  const maxD = Math.max(4, ...D.map((d) => d.k + 1))
  const axisY = 30 + maxA * STEP + 6
  const height = axisY + maxD * STEP + 46
  const ticks = []
  for (let m = T0; m < T1; m += 120) ticks.push(m)

  const dot = (d: { f: (typeof arr)[0]; bx: number; k: number }, up: boolean) => {
    const cd = currentDelay(d.f, t, preds)
    const cls = delayClass(cd.value)
    const cy = up ? axisY - 6 - d.k * STEP : axisY + 6 + d.k * STEP
    const isSel = selected === d.f.id || highlight.has(d.f.id)
    return (
      <circle key={`${up ? 'a' : 'd'}${d.f.id}`} cx={x(d.bx)} cy={cy} r={isSel ? 4.5 : 3}
        className={`wave-dot c-${cls}${cd.kind === 'forecast' ? ' fc' : ''}${isSel ? ' sel' : ''}`}
        onClick={() => onSelect(d.f.id)}
        onMouseMove={(e) => onTip({
          x: e.clientX, y: e.clientY,
          title: `${d.f.no} ${d.f.dep}-${d.f.arr}`,
          lines: [
            `${up ? 'STA' : 'STD'} ${hhmm(up ? d.f.sa : d.f.sd)} · ${d.f.tail}`,
            cd.value === null ? 'beyond forecast horizon'
              : `${cd.kind === 'forecast' ? 'forecast' : 'observed'} ${cd.value >= 0 ? '+' : ''}${Math.round(cd.value)} min`,
          ],
        })}
        onMouseLeave={() => onTip(null)} />
    )
  }

  return (
    <svg width={width} height={height} className="wave" role="img"
      aria-label="Arrival and departure banks at Warsaw coloured by delay">
      <text x={4} y={12} className="axis-label">▲ Arrivals at WAW (scheduled time)</text>
      <text x={4} y={height - 4} className="axis-label">▼ Departures from WAW</text>
      <line x1={40} x2={width - 12} y1={axisY} y2={axisY} className="axis" />
      {ticks.map((m) => (
        <g key={m}>
          <line x1={x(m)} x2={x(m)} y1={axisY - 3} y2={axisY + 3} className="axis" />
          <text x={x(m)} y={axisY + maxD * STEP + 18} textAnchor="middle" className="tick">{hhmm(m)}</text>
        </g>
      ))}
      <rect x={x(T0)} y={20} width={Math.max(0, x(t) - x(T0))} height={height - 60} className="past" />
      <line x1={x(t)} x2={x(t)} y1={18} y2={height - 40} className="now" />
      <text x={x(t) + 4} y={30} className="now-label">{hhmm(t)}</text>
      {A.map((d) => dot(d, true))}
      {D.map((d) => dot(d, false))}
    </svg>
  )
}
