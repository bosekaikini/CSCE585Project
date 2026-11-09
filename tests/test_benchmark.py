import numpy as np
import pytest

from src.benchmark import bootstrap_confidence_interval, paired_difference_summary, summarize


def test_summarize_reports_mean_and_std():
    summary = summarize([1.0, 2.0, 3.0, 4.0])
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["std"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(4.0)


def test_bootstrap_confidence_interval_is_bounded():
    ci = bootstrap_confidence_interval([1.0, 2.0, 3.0, 4.0], iterations=200, seed=7)
    assert ci["lower"] <= ci["statistic"] <= ci["upper"]
    assert ci["confidence"] == pytest.approx(0.95)
    assert ci["iterations"] == 200


def test_paired_difference_summary_tracks_mean_shift():
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([2.0, 3.0, 4.0])
    diff = paired_difference_summary(left, right, iterations=200, seed=11)
    assert diff["mean_difference"] == pytest.approx(1.0)
    assert diff["baseline_mean"] == pytest.approx(2.0)
    assert diff["candidate_mean"] == pytest.approx(3.0)
    assert diff["lower"] <= diff["mean_difference"] <= diff["upper"]
