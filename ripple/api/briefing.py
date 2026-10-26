"""Short operations briefing for one replay frame.

Uses Claude when Anthropic credentials are available (``ANTHROPIC_API_KEY`` or an
``ant auth login`` profile), otherwise a deterministic template, so the demo never depends on it.
The model only rewrites numbers the forecaster produced; it is told not to invent anything.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)
MODEL = "claude-opus-5-5"

SYSTEM = """You write the operations briefing for the duty manager of an airline hub control \
centre (Warsaw Chopin, WAW). You receive the output of a probabilistic delay-propagation model.

Rules:
- Use only the flights, numbers and causes in the data. Never invent flights, times or causes.
- Probabilities and delays are model forecasts. Say "forecast" and keep the uncertainty \
(e.g. "~35 min, 90% range 10-80").
- At most 220 words, in Markdown: a one-line situation summary, then "Watch list" (up to 5 \
flights), "Connections at risk" (up to 5), and "Options" (2-4 concrete, low-regret actions such \
as protecting a connection, standby aircraft, or pre-emptive rebooking).
- If the data is marked synthetic, start with "Synthetic replay - not real LOT operations."."""


def _frame_summary(day: dict, k: int) -> dict:
    fl = {f["id"]: f for f in day["flights"]}
    fr = day["frames"][k]
    cols = day["pred_cols"]
    preds = [dict(zip(cols, p, strict=True)) for p in fr["preds"]]
    preds.sort(key=lambda p: -p["d50"])
    t = fr["t"]

    def hhmm(m):
        m = int(round(m)) % 1440
        return f"{m // 60:02d}:{m % 60:02d}"

    watch = [
        {"flight": fl[p["id"]]["no"], "route": f"{fl[p['id']]['dep']}-{fl[p['id']]['arr']}",
         "tail": fl[p["id"]]["tail"], "std_utc_offset_local": hhmm(fl[p["id"]]["sd"]),
         "dep_delay_median": p["d50"], "dep_delay_90pct_range": [p["d05"], p["d95"]],
         "cause": p["text"] or "no dominant driver"}
        for p in preds[:8] if p["d50"] >= 10
    ]  # fmt: skip
    conns = [
        {"inbound": fl[i]["no"], "from": fl[i]["dep"], "outbound": fl[o]["no"], "to": fl[o]["arr"],
         "scheduled_connection_min": s, "p_miss": p}
        for i, o, s, p in fr["conns"][:8] if p >= 0.15
    ]  # fmt: skip
    return {"time_local": hhmm(t), "watch_list": watch, "connections_at_risk": conns,
            "flights_forecast": len(preds)}  # fmt: skip


def _template(s: dict, synthetic: bool) -> str:
    lines = ["*Synthetic replay - not real LOT operations.*", ""] if synthetic else []
    lines.append(f"**{s['time_local']} WAW** - {len(s['watch_list'])} flights forecast >=10 min late, "
                 f"{len(s['connections_at_risk'])} connections with P(miss) >= 15%.")  # fmt: skip
    if s["watch_list"]:
        lines += ["", "**Watch list**"]
        for w in s["watch_list"][:5]:
            lo, hi = w["dep_delay_90pct_range"]
            lines.append(f"- {w['flight']} {w['route']} ({w['tail']}): ~{w['dep_delay_median']:.0f} min "
                         f"(90%: {lo:.0f}-{hi:.0f}). {w['cause']}")  # fmt: skip
    if s["connections_at_risk"]:
        lines += ["", "**Connections at risk**"]
        for c in s["connections_at_risk"][:5]:
            lines.append(f"- {c['inbound']} from {c['from']} -> {c['outbound']} to {c['to']} "
                         f"({c['scheduled_connection_min']} min): P(miss) {c['p_miss']:.0%}")  # fmt: skip
    return "\n".join(lines)


def briefing(day: dict, k: int, synthetic: bool) -> dict:
    s = _frame_summary(day, k)
    try:
        import anthropic
    except ImportError:
        return {"text": _template(s, synthetic), "source": "template"}
    try:
        client = anthropic.Anthropic()
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps({"synthetic": synthetic, **s})}],
        )
        if resp.stop_reason == "refusal":
            return {"text": _template(s, synthetic), "source": "template (model declined)"}
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return {"text": text or _template(s, synthetic), "source": f"claude ({resp.model})"}
    except anthropic.AuthenticationError:
        log.info("no Anthropic credentials; using template briefing")
    except anthropic.RateLimitError:
        log.warning("Anthropic rate limit; using template briefing")
    except anthropic.APIStatusError as e:
        log.warning("Anthropic API error %s; using template briefing", e.status_code)
    except anthropic.APIConnectionError:
        log.warning("Anthropic unreachable; using template briefing")
    except anthropic.AnthropicError as e:
        log.info("Anthropic client unavailable (%s); using template briefing", e)
    except TypeError as e:  # raised by the SDK when no credential source is configured at all
        log.info("no Anthropic credentials (%s); using template briefing", e)
    return {"text": _template(s, synthetic), "source": "template"}
