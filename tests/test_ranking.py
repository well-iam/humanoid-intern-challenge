"""The ranking test's own logic: candidate paths, rank agreement and the summary report.

The ranking numbers on Push-T are an experiment (scripts/rank.py), not a test.
"""

import numpy as np
import pytest

from pwm.plots import plot_ranking
from pwm.ranking import RankingReport, candidate_paths, spearman

WORKSPACE = 512.0


def recorded_path():
    """A pusher path at 10 Hz over 4 s: a straight stroke to the right with a slight curve."""
    t = np.linspace(0, 1, 41)
    return np.stack([200 + 150 * t, 250 + 20 * t**2], axis=1)


def test_candidates_start_where_the_pusher_is_and_the_first_is_the_recorded_path():
    path = recorded_path()

    candidates = candidate_paths(path, n=32, seed=0, workspace=WORKSPACE)

    assert candidates.shape == (32, 41, 2)
    np.testing.assert_array_equal(candidates[0], path)
    np.testing.assert_allclose(candidates[:, 0], np.tile(path[0], (32, 1)))


def test_candidates_are_reproducible_from_the_seed_and_spread_out():
    path = recorded_path()

    same = candidate_paths(path, n=32, seed=0, workspace=WORKSPACE)
    again = candidate_paths(path, n=32, seed=0, workspace=WORKSPACE)
    other = candidate_paths(path, n=32, seed=1, workspace=WORKSPACE)

    np.testing.assert_array_equal(same, again)
    assert not np.allclose(same[1:], other[1:])
    ends = np.linalg.norm(same[1:, -1] - path[-1], axis=1)
    assert np.median(ends) > 20  # the perturbed paths really go elsewhere...
    assert ends.max() < 2 * np.linalg.norm(path[-1] - path[0])  # ...but stay paths of the same kind


def test_candidates_stay_inside_the_workspace():
    t = np.linspace(0, 1, 41)
    path = np.stack([300 + 180 * t, 480 + 10 * t], axis=1)  # a stroke along the bottom wall

    candidates = candidate_paths(path, n=50, seed=0, workspace=WORKSPACE)

    assert candidates[1:].min() >= 20 and candidates[1:].max() <= WORKSPACE - 20


def test_spearman_is_one_for_the_same_order_and_minus_one_for_the_reverse():
    judged = np.array([3.0, 10.0, 7.0, 1.0, 5.0])

    assert spearman(judged**2 + 4, judged) == pytest.approx(1.0)  # any increasing function keeps the order
    assert spearman(-judged, judged) == pytest.approx(-1.0)


def test_spearman_gives_tied_values_their_average_rank():
    # Worked example: ranks of x are (1, 2.5, 2.5, 4), of y (1, 2, 3, 4);
    # Pearson correlation of those ranks = 4.5 / sqrt(4.5 * 5) = 0.9487.
    assert spearman(np.array([1.0, 2.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.948683, abs=1e-6)


def test_spearman_is_undefined_when_every_candidate_gets_the_same_value():
    assert np.isnan(spearman(np.array([1.0, 2.0, 3.0]), np.full(3, 40.0)))


def make_report(predicted, control, judged):
    return RankingReport(unit="Push-T units", step=0.5, horizon=8, seed=0,
                         starts=[f"start {i}" for i in range(len(judged))],
                         predicted=predicted, control=control, judged=judged)


def test_a_predictor_that_orders_candidates_like_the_judge_picks_the_best_one():
    judged = [[30.0, 10.0, 50.0, 20.0], [5.0, 25.0, 15.0, 35.0]]
    predicted = [[33.0, 12.0, 41.0, 21.0], [9.0, 30.0, 11.0, 40.0]]  # same order, other values
    control = [[2.0, 1.0, 4.0, 3.0], [4.0, 2.0, 3.0, 1.0]]  # first start: two swapped; second: reversed

    summary = make_report(predicted, control, judged).summary()

    assert summary["predictor"]["spearman_per_start"] == pytest.approx([1.0, 1.0])
    assert summary["predictor"]["spearman_mean"] == pytest.approx(1.0)
    assert summary["predictor"]["top_pick"] == pytest.approx((10 + 5) / 2)  # the best candidate of each start
    assert summary["control"]["spearman_per_start"] == pytest.approx([0.8, -1.0])
    assert summary["control"]["top_pick"] == pytest.approx((10 + 35) / 2)
    assert summary["best_possible"] == pytest.approx(7.5)
    assert summary["mean_candidate"] == pytest.approx((110 / 4 + 80 / 4) / 2)
    assert summary["recorded_path"] == pytest.approx((30 + 5) / 2)  # candidate 0 is the recorded path
    assert summary["random_pick"] >= summary["predictor"]["top_pick"]


def test_starts_where_the_judge_cannot_tell_candidates_apart_are_left_out_of_the_average_correlation():
    judged = [[30.0, 10.0, 50.0], [40.0, 40.0, 40.0]]  # second start: no candidate reaches the T
    predicted = [[3.0, 1.0, 5.0], [1.0, 2.0, 3.0]]

    summary = make_report(predicted, predicted, judged).summary()

    assert summary["predictor"]["spearman_mean"] == pytest.approx(1.0)
    assert summary["undefined_starts"] == 1
    assert np.isnan(summary["predictor"]["spearman_per_start"][1])


def test_the_report_states_every_result_and_round_trips_through_its_file(tmp_path):
    report = make_report([[3.0, 1.0, 5.0]], [[1.0, 3.0, 5.0]], [[30.0, 10.0, 50.0]])

    report.save(tmp_path)
    text = (tmp_path / "report.md").read_text()

    assert RankingReport.load(tmp_path) == report
    for phrase in ("Spearman", "top pick", "random pick", "shuffled actions"):
        assert phrase in text


def test_the_ranking_plot_is_drawn(tmp_path):
    report = make_report([[3.0, 1.0, 5.0], [2.0, 1.0, 3.0]], [[1.0, 3.0, 5.0], [3.0, 2.0, 1.0]],
                         [[30.0, 10.0, 50.0], [20.0, 25.0, 30.0]])

    plot_ranking(report, tmp_path / "ranking.png", "DINOv2-small")

    assert (tmp_path / "ranking.png").read_bytes().startswith(b"\x89PNG")
