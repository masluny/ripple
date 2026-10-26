export interface Airport {
  icao: string
  iata: string
  name: string
  lat: number
  lon: number
}

export interface Flight {
  id: number
  no: string
  tail: string
  type: string
  dep: string
  arr: string
  /** minutes after local midnight of the replay day */
  sd: number
  sa: number
  ad: number | null
  aa: number | null
  prev: number | null
  truth: string | null
}

/** [id, d05, d25, d50, d75, d95, a05, a25, a50, a75, a95, cause, deltas, text] */
export type PredRow = [
  number, number, number, number, number, number,
  number, number, number, number, number,
  string, number[], string,
]

export interface Pred {
  id: number
  dep: [number, number, number, number, number]
  arr: [number, number, number, number, number]
  cause: string
  deltas: number[]
  text: string
}

export interface Frame {
  t: number
  preds: PredRow[]
  /** [inbound id, outbound id, scheduled connection minutes, P(miss)] */
  conns: [number, number, number, number][]
}

export interface Day {
  day: string
  model: string
  groups: string[]
  flights: Flight[]
  frames: Frame[]
}

interface Summary {
  n: number
  mae: number
  crps: number
  cov80: number
  cov90: number
  width90: number
}

export interface ModelMetrics {
  horizons: Record<string, { dep: Summary; arr: Summary }>
  connections: Record<string, { brier: number; brier_skill: number; auc: number; rate: number }>
  attribution?: Record<string, { n: number; accuracy: number } | number>
}

export interface Index {
  dataset: string
  days: string[]
  models: string[]
  airports: Airport[]
  metrics?: { models: Record<string, ModelMetrics> }
}

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json() as Promise<T>
}

export const fetchIndex = () => get<Index>('/api/index')
export const fetchDay = (day: string, model: string) => get<Day>(`/api/day/${day}/${model}`)

export async function fetchBriefing(day: string, model: string, frame: number) {
  const r = await fetch('/api/briefing', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ day, model, frame }),
  })
  if (!r.ok) throw new Error(`${r.status}`)
  return (await r.json()) as { text: string; source: string }
}

export function toPred(r: PredRow): Pred {
  return {
    id: r[0],
    dep: [r[1], r[2], r[3], r[4], r[5]],
    arr: [r[6], r[7], r[8], r[9], r[10]],
    cause: r[11],
    deltas: r[12],
    text: r[13],
  }
}
