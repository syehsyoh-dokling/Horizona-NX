"""Regression tests for the Track 2 core - each is a reviewed defect, kept found.

    PYTHONPATH=track2_agent pytest track2_agent/tests -q
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from forecast_core import asof, panel_loader, scenario_mixture, text_signal  # noqa: E402


# ---------------------------------------------------------------------------
# scenario_mixture - the variance bug and its neighbours
# ---------------------------------------------------------------------------

def _flat_panel(n_days=250, sigma=1.0, seed=7, assets=("A",)):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for a in assets:
        level = 100 + np.cumsum(rng.normal(0, sigma, n_days))
        for d, v in zip(dates, level):
            rows.append({"date": d, "asset": a, "value": v, "panel_id": "p"})
    return pd.DataFrame(rows)


def test_one_day_horizon_is_not_underdispersed():
    """The linear rescale shrank a 1-day distribution 2.2x too narrow -
    the horizon must be honoured by truncation, not by scaling the std."""
    panel = _flat_panel(sigma=1.0)
    draws, _ = scenario_mixture.build(panel, ["A"], [1], n_draws=800)
    got = draws[draws["horizon"] == 1]["value"].std()
    daily = 1.0
    assert got == pytest.approx(daily, rel=0.35), (
        f"1-day dispersion {got:.3f} vs daily sigma {daily:.3f}")


def test_missing_series_degrades_instead_of_exiting():
    """SystemExit on a missing series produced an inadmissible zero-file run."""
    panel = _flat_panel(assets=("A",))
    draws, notes = scenario_mixture.build(panel, ["A", "GHOST"], [5], n_draws=50)
    assert set(draws["asset"]) == {"A", "GHOST"}
    assert any("GHOST" in n for n in notes)


def test_short_history_degrades_instead_of_exiting():
    panel = _flat_panel(n_days=10)
    draws, notes = scenario_mixture.build(panel, ["A"], [5], n_draws=50)
    assert len(draws) == 50
    assert any("iid fallback" in n for n in notes)


# ---------------------------------------------------------------------------
# text_signal - the lexicon that measured typography
# ---------------------------------------------------------------------------

def test_substring_false_positives_are_gone():
    """'increase', 'decrease' and 'executive' scored as dovish hits."""
    assert text_signal._net_hits(
        "revenues increase and margins decrease; the executive committee") == 0


def test_real_words_still_score():
    assert text_signal._net_hits("the committee decided to cut rates") < 0
    assert text_signal._net_hits("a hike is warranted") > 0


def test_per_document_cap():
    body = "cut " * 100
    assert text_signal._net_hits(body) == -text_signal.MAX_NET_PER_DOC


# ---------------------------------------------------------------------------
# panel_loader - per-file normalisation
# ---------------------------------------------------------------------------

def _write_parquet(path, frame):
    frame.to_parquet(path, index=False)


def test_mixed_asset_spellings_across_files_all_survive(tmp_path):
    """Alias renaming after concat lost every row of the second file."""
    panels = tmp_path / "panels"
    panels.mkdir()
    _write_parquet(panels / "a.parquet", pd.DataFrame(
        {"date": ["2024-01-02"], "asset": ["X"], "value": [1.0]}))
    _write_parquet(panels / "b.parquet", pd.DataFrame(
        {"date": ["2024-01-02"], "asset_id": ["Y"], "value": [2.0]}))
    panel, _ = panel_loader.load(panels, tmp_path, "2024-06-28")
    assert set(panel["asset"]) == {"X", "Y"}


def test_value_column_aliases(tmp_path):
    panels = tmp_path / "panels"
    panels.mkdir()
    _write_parquet(panels / "a.parquet", pd.DataFrame(
        {"date": ["2024-01-02"], "asset": ["X"], "close": [3.5]}))
    panel, notes = panel_loader.load(panels, tmp_path, "2024-06-28")
    assert list(panel["value"]) == [3.5]
    assert any("close" in n for n in notes)


def test_single_series_panel_takes_its_own_name(tmp_path):
    """A lone 'value' column used to melt into an asset literally named 'value'."""
    panels = tmp_path / "panels"
    panels.mkdir()
    _write_parquet(panels / "ust10y.parquet", pd.DataFrame(
        {"date": ["2024-01-02", "2024-01-03"], "value": [4.1, 4.2]}))
    panel, _ = panel_loader.load(panels, tmp_path, "2024-06-28")
    assert set(panel["asset"]) == {"ust10y"}


def test_tz_aware_dates_and_intraday_asof_survive(tmp_path):
    """tz-aware columns raised on compare; intraday rows ON the as-of day
    were dropped by the midnight cutoff."""
    panels = tmp_path / "panels"
    panels.mkdir()
    stamps = pd.to_datetime(["2024-06-28T14:30:00Z", "2024-06-29T09:00:00Z"])
    _write_parquet(panels / "a.parquet", pd.DataFrame(
        {"date": stamps, "asset": ["X", "X"], "value": [1.0, 2.0]}))
    panel, _ = panel_loader.load(panels, tmp_path, "2024-06-28")
    assert list(panel["value"]) == [1.0]  # intraday on the day kept, next day cut


# ---------------------------------------------------------------------------
# asof - the card fallback that makes the default CMD runnable
# ---------------------------------------------------------------------------

def test_asof_prefers_data_end_and_ignores_target_dates():
    card = {"panel": {"end_date": "2024-06-28"},
            "targets": {"target_dates": ["2024-07-31"]},
            "provenance": {"created_date": "2024-08-01"}}
    assert asof.from_card(card) == "2024-06-28"


def test_asof_none_when_card_says_nothing():
    assert asof.from_card({"task": {"id": "x"}}) is None


# ---------------------------------------------------------------------------
# log_return cards - the organizer's "silent and catastrophic" trap
# ---------------------------------------------------------------------------

def test_log_return_panel_is_cumulated_and_anchored():
    """16/104 cards score the SUM of per-period returns over the horizon,
    anchored at 0. The forecast must be built on cumulative levels and then
    have the as-of anchor removed."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    panel = pd.DataFrame({"date": list(dates) * 2,
                          "asset": ["A"] * 4 + ["B"] * 4,
                          "value": [0.1, -0.05, 0.02, 0.03, 0.0, 0.01, 0.01, -0.02],
                          "panel_id": "p"})
    cum, anchors = panel_loader.cumulative_from_returns(panel)
    a = cum[cum["asset"] == "A"].sort_values("date")["value"].to_numpy()
    assert np.allclose(a, [0.1, 0.05, 0.07, 0.10])
    assert anchors == {"A": pytest.approx(0.10), "B": pytest.approx(0.0)}
    # Draw values minus the anchor are the shipped forecast: starts at 0.
    draws = pd.DataFrame({"asset": ["A", "B"], "value": [0.13, -0.01]})
    shipped = draws["value"] - draws["asset"].map(anchors)
    assert np.allclose(shipped, [0.03, -0.01])
