import { useMemo } from 'react'
import { geoAzimuthalEquidistantRaw, geoInterpolate, geoPath, geoProjection } from 'd3-geo'
import type { GeoProjection } from 'd3-geo'
import { feature, mesh } from 'topojson-client'
import type { GeometryCollection, Topology } from 'topojson-specification'
import worldJson from 'world-atlas/countries-110m.json'
import type { Airport, Day, Pred } from '../api'
import { currentDelay, delayClass, status } from '../model'
import type { Tip } from './Tooltip'

const world = worldJson as unknown as Topology<{ countries: GeometryCollection }>
const LAND = feature(world, world.objects.countries)
const BORDERS = mesh(world, world.objects.countries, (a, b) => a !== b)

/** Azimuthal equidistant around WAW with the radius compressed (r^0.45): every route is a
 * straight spoke, Europe gets room, long-haul stations still fit on screen. */
function fisheyeRaw(l: number, p: number): [number, number] {
  const [x, y] = geoAzimuthalEquidistantRaw(l, p)
  const r = Math.hypot(x, y)
  if (r < 1e-9) return [0, 0]
  const s = Math.pow(r, 0.45) / r
  return [x * s, y * s]
}

interface Props {
  width: number
  height: number
  airports: Airport[]
  day: Day
  t: number
  preds: Map<number, Pred>
  selected: number | null
  highlight: Set<number>
  onSelect: (id: number) => void
  onTip: (tip: Tip | null) => void
}

export function HubMap({ width, height, airports, day, t, preds, selected, highlight, onSelect, onTip }: Props) {
  const byIata = useMemo(() => new Map(airports.map((a) => [a.iata, a])), [airports])
  const used = useMemo(() => {
    const s = new Set<string>()
    day.flights.forEach((f) => { s.add(f.dep); s.add(f.arr) })
    return airports.filter((a) => s.has(a.iata))
  }, [airports, day])

  const { proj, path } = useMemo(() => {
    const p: GeoProjection = geoProjection(fisheyeRaw).rotate([-20.967, -52.166]).clipAngle(115)
    p.fitExtent([[28, 22], [width - 28, height - 22]], {
      type: 'MultiPoint',
      coordinates: used.map((a) => [a.lon, a.lat]),
    })
    return { proj: p, path: geoPath(p) }
  }, [width, height, used])

  // Greedy label placement: busiest airports first, skip labels that would collide.
  const labelled = useMemo(() => {
    const traffic = new Map<string, number>()
    day.flights.forEach((f) => {
      traffic.set(f.dep, (traffic.get(f.dep) ?? 0) + 1)
      traffic.set(f.arr, (traffic.get(f.arr) ?? 0) + 1)
    })
    const placed: [number, number][] = []
    const keep = new Set<string>()
    ;[...used].sort((a, b) => (traffic.get(b.iata) ?? 0) - (traffic.get(a.iata) ?? 0)).forEach((a) => {
      const xy = proj([a.lon, a.lat])
      if (!xy) return
      if (a.iata === 'WAW' || placed.every(([x, y]) => Math.abs(x - xy[0]) > 26 || Math.abs(y - xy[1]) > 11)) {
        placed.push([xy[0], xy[1]])
        keep.add(a.iata)
      }
    })
    return keep
  }, [used, proj, day])

  const landD = useMemo(() => path(LAND) ?? '', [path])
  const borderD = useMemo(() => path(BORDERS) ?? '', [path])
  const routes = useMemo(() => {
    const seen = new Map<string, [Airport, Airport]>()
    day.flights.forEach((f) => {
      const a = byIata.get(f.dep), b = byIata.get(f.arr)
      if (a && b) seen.set([f.dep, f.arr].sort().join('-'), [a, b])
    })
    return [...seen.entries()].map(([k, [a, b]]) => ({
      k,
      d: path({ type: 'LineString', coordinates: [[a.lon, a.lat], [b.lon, b.lat]] }) ?? '',
    }))
  }, [day, byIata, path])

  const aircraft = day.flights
    .filter((f) => status(f, t) === 'airborne')
    .map((f) => {
      const a = byIata.get(f.dep)!, b = byIata.get(f.arr)!
      const p = Math.min(1, Math.max(0, (t - f.ad!) / (f.aa! - f.ad!)))
      const xy = proj(geoInterpolate([a.lon, a.lat], [b.lon, b.lat])(p))
      const cd = currentDelay(f, t, preds)
      return { f, xy, cls: delayClass(cd.value), delay: cd.value }
    })
    .filter((x) => x.xy)

  // A ripple ring for 40 minutes after a flight leaves >= 30 min late.
  const ripples = day.flights
    .filter((f) => f.ad !== null && f.ad - f.sd >= 30 && t >= f.ad && t - f.ad < 40)
    .map((f) => ({ f, xy: proj([byIata.get(f.dep)!.lon, byIata.get(f.dep)!.lat]), age: (t - f.ad!) / 40 }))

  return (
    <svg className="map" width={width} height={height} role="img"
      aria-label="Map of the LOT network centred on Warsaw with aircraft in the air coloured by delay">
      <path d={landD} className="land" />
      <path d={borderD} className="border" />
      {routes.map((r) => <path key={r.k} d={r.d} className="route" />)}
      {ripples.map(({ f, xy, age }) => xy && (
        <circle key={`r${f.id}`} cx={xy[0]} cy={xy[1]} r={6 + age * 26}
          className={`ripple c-${delayClass(f.ad! - f.sd)}`} style={{ opacity: 1 - age }} />
      ))}
      {used.map((a) => {
        const xy = proj([a.lon, a.lat])
        if (!xy) return null
        const hub = a.iata === 'WAW'
        return (
          <g key={a.iata} transform={`translate(${xy[0]},${xy[1]})`}>
            <circle r={hub ? 7 : 2.5} className={hub ? 'hub' : 'airport'} />
            {labelled.has(a.iata) && <text x={hub ? 10 : 5} y={hub ? 4 : 3} className={hub ? 'ap-label hub-label' : 'ap-label'}>{a.iata}</text>}
            <title>{`${a.name} (${a.iata})`}</title>
          </g>
        )
      })}
      {aircraft.map(({ f, xy, cls, delay }) => (
        <circle key={f.id} cx={xy![0]} cy={xy![1]}
          r={selected === f.id || highlight.has(f.id) ? 6.5 : 4.5}
          className={`plane c-${cls}${selected === f.id ? ' sel' : ''}${highlight.has(f.id) ? ' hl' : ''}`}
          onClick={() => onSelect(f.id)}
          onMouseMove={(e) => onTip({
            x: e.clientX, y: e.clientY,
            title: `${f.no} ${f.dep}-${f.arr}`,
            lines: [`${f.tail} · ${f.type}`, `departed ${delay === null ? '?' : `${delay >= 0 ? '+' : ''}${Math.round(delay)} min`}`],
          })}
          onMouseLeave={() => onTip(null)} />
      ))}
    </svg>
  )
}
