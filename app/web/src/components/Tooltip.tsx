export interface Tip {
  x: number
  y: number
  title: string
  lines: string[]
}

export function Tooltip({ tip }: { tip: Tip | null }) {
  if (!tip) return null
  const left = Math.min(tip.x + 14, window.innerWidth - 220)
  return (
    <div className="tooltip" style={{ left, top: tip.y + 14 }} role="tooltip">
      <strong>{tip.title}</strong>
      {tip.lines.map((l, i) => <div key={i}>{l}</div>)}
    </div>
  )
}
