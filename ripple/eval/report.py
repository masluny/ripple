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
COLORS = {"lgbm": "#718096", "gnn": "#2b6cb0"}


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
        "",
    ]
    for tgt, name in (("dep", "Departure"), ("arr", "Arrival")):
        lines += [f"## {name} delay", "", "| model | h | MAE | CRPS | cov80 | cov90 | width90 |",
                  "|---|---|---|---|---|---|---|"]  # fmt: skip
        for v, r in models.items():
            for h in config.HORIZONS:
                m = _h(r["horizons"], h)[tgt]
                lines.append(
                    f"| {LABELS.get(v, v)} | {h}h | {m['mae']:.2f} | {m['crps']:.2f} | "
                    f"{m['cov80']:.1%} | {m['cov90']:.1%} | {m['width90']:.0f} |"
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

    # Figure 1: CRPS by horizon
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    w = 0.8 / max(len(models), 1)
    for k, (v, r) in enumerate(models.items()):
        for j, tgt in enumerate(("dep", "arr")):
            vals = [_h(r["horizons"], h)[tgt]["crps"] for h in config.HORIZONS]
            ax[j].bar(np.arange(3) + k * w, vals, w, label=LABELS.get(v, v),
                      color=COLORS.get(v))  # fmt: skip
    for j, tgt in enumerate(("departure", "arrival")):
        ax[j].set(xticks=np.arange(3) + w * (len(models) - 1) / 2,
                  xticklabels=[f"{h}h" for h in config.HORIZONS], title=f"CRPS - {tgt} delay (lower is better)")  # fmt: skip
    ax[1].legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(figs / "crps_by_horizon.png", dpi=120)
    plt.close()

    # Figure 2: quantile calibration + connection reliability
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[1].plot([0, 1], [0, 1], "k:", lw=1)
    for v, r in models.items():
        ax[0].plot(TAUS, r["calibration_dep"], marker=".", label=LABELS.get(v, v),
                   color=COLORS.get(v))  # fmt: skip
        rel = np.array(_h(r["connections"], 1)["reliability"])
        if len(rel):
            ax[1].plot(
                rel[:, 0], rel[:, 1], marker="o", label=LABELS.get(v, v), color=COLORS.get(v)
            )
    ax[0].set(
        xlabel="nominal quantile",
        ylabel="observed P(y <= q)",
        title="Quantile calibration (departure)",
    )
    ax[1].set(
        xlabel="predicted P(miss)",
        ylabel="observed miss rate",
        title="Connection reliability (1h ahead)",
    )
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figs / "calibration.png", dpi=120)
    plt.close()
    lines += [
        "![CRPS](figures/crps_by_horizon.png)",
        "",
        "![calibration](figures/calibration.png)",
        "",
    ]
    (out / "results.md").write_text("\n".join(lines))
    return "\n".join(lines)
