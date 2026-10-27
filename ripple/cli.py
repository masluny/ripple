"""Command line entry point: ``uv run ripple --help``."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

import typer
from rich.console import Console

from ripple import config

app = typer.Typer(no_args_is_help=True, add_completion=False)
collect = typer.Typer(no_args_is_help=True, help="Data collectors (run by launchd, see scripts/).")
app.add_typer(collect, name="collect")
console = Console()


@app.callback()
def _main(verbose: bool = typer.Option(False, "-v")) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config.ensure_dirs()


def _yesterday() -> str:
    return (datetime.now(UTC).date() - timedelta(days=1)).isoformat()


@collect.command("states")
def collect_states() -> None:
    """One anonymous OpenSky state-vector snapshot of the WAW area (run every 4 min)."""
    from ripple.collect.opensky import poll_states_once

    n = poll_states_once()
    console.print(f"{datetime.now(UTC):%H:%M:%S} states: {n} aircraft")


@collect.command("taf")
def collect_taf() -> None:
    """Archive current TAFs for the whole network (run hourly)."""
    from ripple.collect.weather import poll_taf

    console.print(f"TAF: {poll_taf()} reports")


@collect.command("metar")
def collect_metar(
    start: str = typer.Option(None, help="YYYY-MM-DD; default = last 3 days"),
    end: str = typer.Option(None, help="YYYY-MM-DD (exclusive)"),
) -> None:
    """Download METARs from the IEM archive."""
    from ripple.collect.weather import backfill_metar, update_metar_archive

    if start:
        end_d = date.fromisoformat(end) if end else datetime.now(UTC).date()
        backfill_metar(date.fromisoformat(start), end_d)
    else:
        update_metar_archive()
    console.print("METAR archive updated")


@collect.command("schedules")
def collect_schedules(
    day: str = typer.Option(None, help="Local date YYYY-MM-DD; default = yesterday"),
    retain_days: int = typer.Option(0, help="Delete raw files older than N days (0 = keep)"),
) -> None:
    """WAW scheduled/actual times from AeroDataBox (needs AERODATABOX_KEY)."""
    from ripple.collect.schedules import collect_day, purge_older_than

    console.print(f"saved {collect_day(day or _yesterday())}")
    if retain_days:
        console.print(f"purged {purge_older_than(retain_days)} raw files")


@collect.command("opensky-flights")
def collect_opensky_flights(
    day: str = typer.Option(None, help="UTC date; default = yesterday"),
) -> None:
    """EPWA departures/arrivals from OpenSky /flights (needs OPENSKY_CLIENT_ID/SECRET)."""
    from ripple.collect.opensky import fetch_hub_flights

    console.print(f"saved {fetch_hub_flights(day or _yesterday())}")


@collect.command("aircraft-db")
def collect_aircraft_db() -> None:
    """Download the OpenSky aircraft database and keep the LOT fleet."""
    from ripple.collect.aircraft import update_fleet

    console.print(f"saved {update_fleet()}")


@collect.command("daily")
def collect_daily() -> None:
    """Everything that runs once a day; each source is skipped if its credentials are missing."""
    import os

    from ripple.collect.weather import update_metar_archive

    update_metar_archive()
    console.print("METAR ok")
    if os.environ.get("OPENSKY_CLIENT_ID"):
        collect_opensky_flights(None)
    else:
        console.print("OpenSky flights skipped (no credentials)")
    if os.environ.get("AERODATABOX_KEY"):
        collect_schedules(None, int(os.environ.get("AERODATABOX_RETAIN_DAYS", "0")))
    else:
        console.print("AeroDataBox skipped (no key)")


@app.command()
def build(
    source: str = typer.Option("synthetic", help="synthetic | real"),
    days: int = typer.Option(365, help="synthetic only"),
    seed: int = typer.Option(7, help="synthetic only"),
) -> None:
    """Assemble flights + weather tables into data/processed/<source>/."""
    from ripple.data.build import build_real, build_synthetic

    path = build_synthetic(days=days, seed=seed) if source == "synthetic" else build_real()
    console.print(f"dataset written to {path}")


@app.command()
def train(
    model: str = typer.Argument(..., help="variant name, e.g. lgbm, gnn, gnn_norot"),
    dataset: str = typer.Option("synthetic"),
    epochs: int = typer.Option(None, help="GNN only"),
) -> None:
    """Train one model variant (see ripple.pipeline.VARIANTS)."""
    from ripple import isolate

    kw = {"epochs": epochs} if epochs else {}
    isolate.run(isolate.train_variant, dataset, model, kw)
    console.print(f"trained {model} on {dataset}")


@app.command()
def evaluate(
    models: str = typer.Option("lgbm,gnn", help="comma-separated variants"),
    dataset: str = typer.Option("synthetic"),
) -> None:
    """Evaluate trained variants on the test split and write reports/<dataset>/."""
    from ripple.eval.report import write_report
    from ripple.pipeline import Experiment

    exp = Experiment(dataset)
    res = exp.evaluate(models.split(","))
    console.print(write_report(exp, res))


@app.command("run-all")
def run_all(
    dataset: str = typer.Option("synthetic", help="synthetic | real"),
    ablations: bool = typer.Option(True, help="also train ablation variants"),
    rebuild: bool = typer.Option(False, help="rebuild the dataset first"),
    quick: bool = typer.Option(False, help="smoke run: 45 simulated days, 1 GNN epoch"),
) -> None:
    """Build data (if needed), train LightGBM + GNN (+ ablations), evaluate, write the report."""
    from ripple import isolate
    from ripple.data.build import build_real, build_synthetic, dataset_dir
    from ripple.eval.report import write_report
    from ripple.pipeline import Experiment

    if quick:
        dataset = f"{dataset}-quick"
    if rebuild or not (dataset_dir(dataset) / "flights.parquet").exists():
        if dataset.startswith("synthetic"):
            build_synthetic(days=45 if quick else 365, name=dataset)
        else:
            build_real()
    variants = ["lgbm", "gnn"]
    if ablations:
        variants += ["lgbm_noweather", "gnn_norot", "gnn_noconn", "gnn_noapt", "gnn_noweather",
                     "gnn_nogru"]  # fmt: skip
    exp = Experiment(dataset)
    for v in variants:
        console.print(f"[bold]training {v}[/bold]")
        isolate.run(
            isolate.train_variant, dataset, v, {"epochs": 1} if quick and "gnn" in v else {}
        )
    console.print(write_report(exp, exp.evaluate(variants)))


@app.command()
def demo(
    dataset: str = typer.Option("synthetic"),
    days: int = typer.Option(4, help="number of test days to precompute"),
) -> None:
    """Precompute dashboard replay data (predictions, P(miss), causes) for a few test days."""
    from ripple.demo import build_demo

    console.print(f"demo data in {build_demo(dataset, n_days=days)}")


@app.command()
def serve(port: int = typer.Option(8000), reload: bool = typer.Option(False)) -> None:
    """Run the FastAPI backend (serves the built dashboard from app/web/dist too)."""
    import uvicorn

    uvicorn.run("ripple.api.main:app", host="127.0.0.1", port=port, reload=reload)


if __name__ == "__main__":
    app()
