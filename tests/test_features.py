import numpy as np
import pandas as pd
import pytest

from ripple.data.features import EPOCH, FeatureStore, horizon_rows
from ripple.data.metar import decode
from ripple.data.network import add_rotations, connections
from ripple.data.synthetic import SimConfig, generate


@pytest.fixture(scope="module")
def sim():
    return generate(SimConfig(days=8, seed=3))


def test_rotations_link_same_tail_and_airport(sim):
    f = add_rotations(sim[0])
    linked = f.dropna(subset=["prev_id"])
    prev = f.set_index("flight_id").loc[linked.prev_id.astype(int)]
    assert (prev["tail"].to_numpy() == linked["tail"].to_numpy()).all()
    assert (prev.arr_airport.to_numpy() == linked.dep_airport.to_numpy()).all()
    assert (prev.sched_arr.to_numpy() < linked.sched_dep.to_numpy()).all()
    assert len(linked) / len(f) > 0.9


def test_connections_are_hub_arrival_to_departure(sim):
    f = sim[0]
    pairs = connections(f)
    idx = f.set_index("flight_id")
    assert (idx.loc[pairs.in_id, "arr_airport"] == "EPWA").all()
    assert (idx.loc[pairs.out_id, "dep_airport"] == "EPWA").all()
    assert pairs.slack.between(35, 240).all()


def test_features_do_not_leak_future(sim):
    """Scrambling every actual time after t must not change any feature computed at t."""
    flights, wx, fc = sim
    store = FeatureStore(flights, wx, fc)
    rows = horizon_rows(store)
    t_cut = float(np.median(rows.t))
    sel = rows[rows.t == t_cut]
    assert len(sel) > 0
    x1 = store.features(sel.idx.to_numpy(), sel.t.to_numpy())

    rng = np.random.default_rng(0)
    f2 = flights.copy()
    cut = EPOCH + pd.Timedelta(minutes=t_cut)
    for c in ("actual_dep", "actual_arr"):
        late = f2[c] > cut
        f2.loc[late, c] = f2.loc[late, c] + pd.to_timedelta(rng.uniform(1, 300, late.sum()), "min")
    store2 = FeatureStore(f2, wx, fc)
    x2 = store2.features(sel.idx.to_numpy(), sel.t.to_numpy())
    np.testing.assert_array_equal(np.nan_to_num(x1, nan=-999), np.nan_to_num(x2, nan=-999))


def test_targets_not_departed_at_prediction_time(sim):
    store = FeatureStore(*sim)
    rows = horizon_rows(store)
    x = store.features(rows.idx.to_numpy(), rows.t.to_numpy())
    names = store.feature_names()
    assert (x[:, names.index("own_state")] == 0).all()
    assert (x[:, names.index("lead")] >= 60).all()


def test_metar_trend_is_separate_from_body():
    d = decode("EPWA 011200Z 27010KT 9999 FEW030 10/05 Q1015 TEMPO 0500 FG")
    assert d["vis_m"] == 9999 and d["fg"] == 0 and d["trend_adverse"] == 1
