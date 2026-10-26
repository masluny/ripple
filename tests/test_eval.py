import numpy as np
from scipy import stats

from ripple.eval import metrics
from ripple.eval.connections import miss_probability, sample_quantiles

TAUS = metrics.TAUS


def test_crps_zero_for_perfect_point_forecast():
    y = np.array([3.0, -2.0, 40.0])
    q = np.repeat(y[:, None], len(TAUS), axis=1)
    assert metrics.crps(y, q) == 0


def test_calibrated_gaussian_scores_well():
    rng = np.random.default_rng(0)
    y = rng.normal(10, 5, 20000)
    q = np.tile(stats.norm.ppf(TAUS, 10, 5), (len(y), 1))
    s = metrics.summarize(y, q)
    assert abs(s["cov90"] - 0.90) < 0.01 and abs(s["cov80"] - 0.80) < 0.01
    assert s["cal_error"] < 0.01
    # Expected CRPS of a calibrated N(mu, sigma) is sigma / sqrt(pi) ~ 2.82. The 19-quantile
    # average leaves out the tails, where pinball loss is small, so it comes out slightly higher.
    assert 2.82 < s["crps"] < 3.1


def test_sampling_recovers_quantiles():
    q = np.tile(stats.norm.ppf(TAUS, 0, 10), (1, 1))
    s = sample_quantiles(q, 200_000, np.random.default_rng(1))[0]
    assert abs(np.median(s)) < 0.3
    assert abs(np.quantile(s, 0.9) - stats.norm.ppf(0.9, 0, 10)) < 0.5


def test_miss_probability_extremes():
    late = np.full((1, len(TAUS)), 120.0)
    on_time = np.zeros((1, len(TAUS)))
    slack = np.array([60.0])
    assert miss_probability(slack, late, on_time)[0] == 1.0
    assert miss_probability(slack, on_time, on_time)[0] == 0.0
    # a held departure (outbound also 120 late) keeps the connection
    assert miss_probability(slack, late, late)[0] == 0.0


def test_brier_skill_and_auc():
    y = np.array([0, 0, 1, 1.0])
    perfect = metrics.binary_summary(y.copy(), y, base_rate=0.5)
    assert perfect["brier"] == 0 and perfect["brier_skill"] == 1 and perfect["auc"] == 1


def test_calibration_offsets_fix_underdispersed_forecast():
    from ripple.eval.calibration import apply_offsets, fit_offsets

    rng = np.random.default_rng(2)
    y_val, y_test = rng.normal(0, 10, 20000), rng.normal(0, 10, 20000)
    narrow = np.tile(stats.norm.ppf(TAUS, 0, 6), (20000, 1))  # too confident
    assert metrics.coverage(y_test, narrow, 0.05, 0.95)[0] < 0.75
    cal = apply_offsets(narrow, fit_offsets(y_val, narrow))
    assert abs(metrics.coverage(y_test, cal, 0.05, 0.95)[0] - 0.90) < 0.01
    assert metrics.crps(y_test, cal) < metrics.crps(y_test, narrow)
