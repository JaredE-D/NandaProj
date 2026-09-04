"""Tests for the 01b plot helpers.

plotly is a remote-only dependency, so the figure-building tests skip in the
local venv. The palette invariants do not need plotly and always run -- those
are the ones that silently rot.
"""

from __future__ import annotations

import numpy as np
import pytest

from nandaproj import viz

go = pytest.importorskip("plotly.graph_objects", reason="plotly is remote-only")


def test_series_line_assigns_slots_in_fixed_order():
    fig = viz.series_line([0, 1, 2], {"a": [0.1, 0.2, 0.3], "b": [0.3, 0.2, 0.1]})
    colours = [t.line.color for t in fig.data]
    assert colours == [viz.SERIES[0], viz.SERIES[1]]


def test_series_line_direct_labels_every_series():
    fig = viz.series_line([0, 1], {"a": [0.1, 0.2], "b": [0.2, 0.1]})
    labelled = {a.text for a in fig.layout.annotations}
    assert labelled == {"a", "b"}


def test_series_line_hides_legend_for_a_single_series():
    fig = viz.series_line([0, 1], {"only": [0.1, 0.2]})
    assert fig.layout.showlegend is False


def test_series_line_rejects_more_series_than_slots():
    series = {str(i): [0.0, 1.0] for i in range(len(viz.SERIES) + 1)}
    with pytest.raises(ValueError, match="fixed slots"):
        viz.series_line([0, 1], series)


def test_prob_heatmap_anchors_at_zero():
    """An autoscaled floor makes 0.02 look like a hit."""
    fig = viz.prob_heatmap(np.full((3, 4), 0.02), x=range(4), y=range(3))
    assert fig.data[0].zmin == 0.0
    assert fig.data[0].zmax == 1.0


def test_grouped_bar_groups_on_one_axis():
    fig = viz.grouped_bar([0, 50, 100], {"easy": [0.1, 0.2, 0.7],
                                         "hard": [0.7, 0.2, 0.1]})
    assert fig.layout.barmode == "group"
    assert all(t.yaxis in (None, "y") for t in fig.data)


def test_grouped_bar_rejects_more_series_than_slots():
    series = {str(i): [1.0] for i in range(len(viz.SERIES) + 1)}
    with pytest.raises(ValueError, match="fixed slots"):
        viz.grouped_bar([0], series)
