import type { Day, Flight, Frame, Pred } from './api'
import { toPred } from './api'

export type Status = 'scheduled' | 'overdue' | 'airborne' | 'landed'
export type DelayClass = 'good' | 'warning' | 'serious' | 'critical' | 'unknown'

export const CLASS_LABEL: Record<DelayClass, string> = {
  good: 'on time (<=15)',
  warning: '15-30 min',
  serious: '30-60 min',
  critical: '>60 min',
  unknown: 'beyond horizon',
}

export const GROUP_LABEL: Record<string, string> = {
  rotation: 'Inbound aircraft',
  connection: 'Feeder flights',
  weather: 'Weather',
  airport: 'Airport congestion',
}

export function hhmm(m: number): string {
  const v = ((Math.round(m) % 1440) + 1440) % 1440
  return `${String(Math.floor(v / 60)).padStart(2, '0')}:${String(v % 60).padStart(2, '0')}`
}

export function delayClass(d: number | null | undefined): DelayClass {
  if (d === null || d === undefined || Number.isNaN(d)) return 'unknown'
  if (d <= 15) return 'good'
  if (d <= 30) return 'warning'
  if (d <= 60) return 'serious'
  return 'critical'
}

export function frameIndexAt(day: Day, t: number): number {
  let k = 0
  for (let i = 0; i < day.frames.length; i++) if (day.frames[i].t <= t) k = i
  return k
}

export function predMap(frame: Frame): Map<number, Pred> {
  return new Map(frame.preds.map((r) => [r[0], toPred(r)]))
}

export function status(f: Flight, t: number): Status {
  if (f.aa !== null && f.aa <= t) return 'landed'
  if (f.ad !== null && f.ad <= t) return 'airborne'
  if (t > f.sd) return 'overdue'
  return 'scheduled'
}

/** Best delay information about a flight at time t: observed if it happened, else the forecast. */
export function currentDelay(
  f: Flight,
  t: number,
  preds: Map<number, Pred>,
): { value: number | null; kind: 'actual-arr' | 'actual-dep' | 'forecast' | 'none'; leg: 'dep' | 'arr' } {
  const s = status(f, t)
  if (s === 'landed') return { value: f.aa! - f.sa, kind: 'actual-arr', leg: 'arr' }
  if (s === 'airborne') return { value: f.ad! - f.sd, kind: 'actual-dep', leg: 'dep' }
  const p = preds.get(f.id)
  if (p) {
    const v = s === 'overdue' ? Math.max(p.dep[2], t - f.sd) : p.dep[2]
    return { value: v, kind: 'forecast', leg: 'dep' }
  }
  if (s === 'overdue') return { value: t - f.sd, kind: 'forecast', leg: 'dep' }
  return { value: null, kind: 'none', leg: 'dep' }
}
