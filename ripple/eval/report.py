"""Markdown + figures from the metrics dict produced by Experiment.evaluate."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ripple import config  # noqa: E402
from ripple.eval.metrics import TAUS  # noqa: E402

LABELS = {
    "lgbm": "LightGBM (baseline)",
    "lgbm_noweather": "LightGBM - no weather",
    "gnn": "GNN (HeteroGATv2 + GRU)",
    "gnn_norot": "GNN - no rotation edges",
    "gnn_noconn": "GNN - no connection edges",
    "gnn_noapt": "GNN - no airport/time edges",
    "gnn_noweather": "GNN - no weather",
    "gnn_nogru": "GNN - no GRU (static)",
}
COLORS = {"lgbm": "#a3a29b", "gnn": "#2a78d6"}


def _h(d: dict, h: int):
    return d.get(h, d.get(str(h)))


def write_report(exp, res: dict) -> str:
    out = exp.report_dir
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    models = res["models"]
    lines = [f"# Results - dataset `{res['dataset']}`", ""]
    lines += [
        "Test split = last 15% of days (chronological, 1-day gap). All numbers in minutes.",
        "CRPS is approximated from 19 quantiles. Coverage should be close to 80% / 90%.",
        "Both models are recalibrated post hoc on the validation split (per-quantile offsets);",
        "`raw` columns show the uncalibrated model.",
        "",
    ]
    for tgt, name in (("dep", "Departure"), ("arr", "Arrival")):
        lines += [f"## {name} delay", "",
                  "| model | h | MAE | CRPS | cov80 | cov90 | width90 | raw CRPS | raw cov90 |",
                  "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for v, r in models.items():
            for h in config.HORIZONS:
                m = _h(r["horizons"], h)[tgt]
                raw = _h(r.get("horizons_raw", r["horizons"]), h)[tgt]
                lines.append(
                    f"| {LABELS.get(v, v)} | {h}h | {m['mae']:.2f} | {m['crps']:.2f} | "
                    f"{m['cov80']:.1%} | {m['cov90']:.1%} | {m['width90']:.0f} | "
                    f"{raw['crps']:.2f} | {raw['cov90']:.1%} |"
                )
        lines.append("")
    lines += ["## Missed connections at WAW (MCT 35 min, slack <= 150 min)", "",
              "| model | h | Brier | Brier skill vs base rate | AUC | miss rate |", "|---|---|---|---|---|---|"]  # fmt: skip
    for v, r in models.items():
        for h in config.HORIZONS:
            c = _h(r["connections"], h)
            lines.append(
                f"| {LABELS.get(v, v)} | {h}h | {c['brier']:.4f} | {c['brier_skill']:.3f} | "
                f"{c['auc']:.3f} | {c['rate']:.1%} |"
            )
    lines.append("")

    headline = {v: r for v, r in models.items() if v in ("lgbm", "gnn")}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    # Figure 1: CRPS by horizon, headline models only, values labelled on the bars
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    w = 0.38
    for k, (v, r) in enumerate(headline.items()):
        for j, tgt in enumerate(("dep", "arr")):
            vals = [_h(r["horizons"], h)[tgt]["crps"] for h in config.HORIZONS]
            bars = ax[j].bar(np.arange(3) + k * w, vals, w * 0.92, label=LABELS.get(v, v),
                             color=COLORS.get(v))  # fmt: skip
            ax[j].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    for j, tgt in enumerate(("departure", "arrival")):
        ax[j].set(xticks=np.arange(3) + w / 2, xticklabels=[f"{h} h ahead" for h in config.HORIZONS],
                  ylabel="CRPS (min)", title=f"{tgt.capitalize()} delay - CRPS, lower is better")  # fmt: skip
        ax[j].grid(axis="y", color="#e6e4de", lw=0.8)
        ax[j].set_axisbelow(True)
    ax[0].legend(fontsize=8, loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(figs / "crps_by_horizon.png", dpi=120)
    plt.close()

    # Figure 2: quantile calibration + connection reliability (6 h ahead), headline models
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a in ax:
        a.plot([0, 1], [0, 1], color="#a3a29b", ls=":", lw=1)
    for v, r in headline.items():
        ax[0].plot(TAUS, r["calibration_dep"], marker="o", ms=3, lw=2, label=LABELS.get(v, v),
                   color=COLORS.get(v))  # fmt: skip
        rel = np.array(_h(r["connections"], 6)["reliability"])
        if len(rel):
            ax[1].plot(rel[:, 0], rel[:, 1], marker="o", ms=4, lw=2, label=LABELS.get(v, v),
                       color=COLORS.get(v))  # fmt: skip
    ax[0].set(xlabel="nominal quantile", ylabel="observed share of outcomes below it",
              title="Quantile calibration, departure delay (test)")  # fmt: skip
    ax[1].set(xlabel="predicted P(miss)", ylabel="observed miss rate",
              title="Missed-connection reliability, 6 h ahead (test)")  # fmt: skip
    ax[0].legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(figs / "calibration.png", dpi=120)
    plt.close()

    # Figure 3: ablations - change in 6 h departure CRPS relative to the full GNN
    if "gnn" in models and len(models) > 2:
        base = _h(models["gnn"]["horizons"], 6)["dep"]["crps"]
        abl = [(LABELS.get(v, v), 100 * (_h(r["horizons"], 6)["dep"]["crps"] - base) / base)
               for v, r in models.items() if v != "gnn"]  # fmt: skip
        abl.sort(key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(abl) + 1))
        ax.barh([a[0] for a in abl], [a[1] for a in abl], color="#2a78d6", height=0.6)
        for i, (_, val) in enumerate(abl):  # labels right of zero so negatives do not hit tick text
            ax.text(max(val, 0) + 0.3, i, f"{val:+.1f}%", va="center", fontsize=8)
        ax.axvline(0, color="#52514e", lw=1)
        ax.set(xlabel="change in departure-delay CRPS at 6 h vs full GNN (%, higher = worse)",
               title="Ablations")  # fmt: skip
        plt.tight_layout()
        plt.savefig(figs / "ablations.png", dpi=120)
        plt.close()
        lines += ["![ablations](figures/ablations.png)", ""]

    lines += [
        "![CRPS](figures/crps_by_horizon.png)",
        "",
        "![calibration](figures/calibration.png)",
        "",
    ]
    (out / "results.md").write_text("\n".join(lines))
    return "\n".join(lines)
