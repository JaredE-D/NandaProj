"""Plotly helpers for activation tensors.

Deliberately thin: `imshow` for anything 2D (attention patterns, head-by-layer
scores) and `line` for 1D sweeps. plotly is only imported when called, so this
module is importable in the local venv where plotly is absent.
"""

from __future__ import annotations

from typing import Any


def _to_numpy(tensor: Any):
    """Detach a torch tensor to numpy; pass numpy arrays through untouched."""
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().to("cpu").float().numpy()
    return tensor


def imshow(tensor: Any, title: str = "", xaxis: str = "", yaxis: str = "", **kwargs):
    """Diverging heatmap centred on zero -- the default for activation diffs."""
    import plotly.express as px

    fig = px.imshow(
        _to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        title=title,
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    )
    return fig


def line(tensor: Any, title: str = "", xaxis: str = "", yaxis: str = "", **kwargs):
    """Simple 1D line plot, for layer sweeps and logit-diff curves."""
    import plotly.express as px

    fig = px.line(_to_numpy(tensor), title=title, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis, showlegend=False)
    return fig
