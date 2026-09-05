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


# --------------------------------------------------------------------------
# 01b readout plots.
#
# Palette and mark specs come from the dataviz reference instance: categorical
# slots in fixed order (never cycled), a single-hue blue ramp for magnitude,
# 2px lines, >=8px markers, recessive grid, legend whenever there are two or
# more series. The `imshow` above stays diverging because activation diffs are
# signed; probabilities are not, so they get the sequential ramp instead.
# --------------------------------------------------------------------------

# Categorical slots 1-4, in order. Assign by entity, never by rank.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

# Single-hue sequential ramp, light -> dark, for probabilities in [0, 1].
SEQUENTIAL = [
    [0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"],
    [0.75, "#256abf"], [1.0, "#0d366b"],
]

_TEXT = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e8e7e3"


def _style(fig, title: str, xaxis: str, yaxis: str):
    """Shared chrome: recessive axes, room for direct labels, no chartjunk."""
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font={"color": _TEXT, "size": 13},
        margin={"l": 60, "r": 90, "t": 60, "b": 50},
        legend={"orientation": "h", "y": -0.18, "x": 0, "font": {"color": _MUTED}},
    )
    fig.update_xaxes(gridcolor=_GRID, zeroline=False, linecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False, linecolor=_GRID)
    return fig


def series_line(
    x,
    series: dict,
    title: str = "",
    xaxis: str = "",
    yaxis: str = "",
    y_range: tuple | None = None,
):
    """Per-layer curves, one line per named series (J-lens vs logit lens).

    Two encodings of identity, never colour alone: a legend plus a direct label
    at the last point of each series.
    """
    import plotly.graph_objects as go

    if len(series) > len(SERIES):
        raise ValueError(f"{len(series)} series exceeds the {len(SERIES)} fixed slots")

    fig = go.Figure()
    for slot, (name, ys) in enumerate(series.items()):
        ys = _to_numpy(ys)
        colour = SERIES[slot]
        fig.add_trace(go.Scatter(
            x=list(x), y=list(ys), name=name, mode="lines+markers",
            line={"color": colour, "width": 2}, marker={"size": 8, "color": colour},
            hovertemplate=f"{name}<br>layer %{{x}}<br>%{{y:.3f}}<extra></extra>",
        ))
        fig.add_annotation(
            x=list(x)[-1], y=float(ys[-1]), text=name, showarrow=False,
            xanchor="left", xshift=8, font={"color": colour, "size": 12},
        )

    _style(fig, title, xaxis, yaxis)
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    if len(series) < 2:
        fig.update_layout(showlegend=False)  # one series: the title names it
    return fig


def prob_heatmap(matrix, x, y, title: str = "", xaxis: str = "", yaxis: str = ""):
    """Sequential heatmap for probabilities: one hue, light -> dark, zmin=0.

    Anchoring at zero matters -- an autoscaled floor makes 0.02 look like a hit.
    """
    import plotly.graph_objects as go

    fig = go.Figure(go.Heatmap(
        z=_to_numpy(matrix), x=list(x), y=list(y),
        colorscale=SEQUENTIAL, zmin=0.0, zmax=1.0,
        colorbar={"title": "P", "outlinewidth": 0},
        hovertemplate=f"{xaxis} %{{x}}<br>{yaxis} %{{y}}<br>P %{{z:.3f}}<extra></extra>",
    ))
    return _style(fig, title, xaxis, yaxis)


def grouped_bar(
    categories,
    series: dict,
    title: str = "",
    xaxis: str = "",
    yaxis: str = "",
):
    """Grouped bars over a shared category axis (the confidence distribution).

    Rounded data-ends and a 2px surface gap between adjacent bars, per the mark
    spec; bars share one axis -- never a second y-scale.
    """
    import plotly.graph_objects as go

    if len(series) > len(SERIES):
        raise ValueError(f"{len(series)} series exceeds the {len(SERIES)} fixed slots")

    fig = go.Figure()
    for slot, (name, ys) in enumerate(series.items()):
        fig.add_trace(go.Bar(
            x=list(categories), y=list(_to_numpy(ys)), name=name,
            marker={"color": SERIES[slot], "cornerradius": 4,
                    "line": {"width": 2, "color": "#fcfcfb"}},
            hovertemplate=f"{name}<br>%{{x}}<br>%{{y:.3f}}<extra></extra>",
        ))

    _style(fig, title, xaxis, yaxis)
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05)
    if len(series) < 2:
        fig.update_layout(showlegend=False)
    return fig


def scatter(
    x,
    y,
    labels,
    title: str = "",
    xaxis: str = "",
    yaxis: str = "",
    text=None,
    symbols=None,
):
    """One point per row, coloured by `labels` -- for PCA scores.

    Colour is one encoding and `symbols` is the second, so two labellings can be
    read off the same plot: colour the condition and shape the polarity, and a
    cloud that separates by shape rather than by colour is visible as such
    rather than needing a second figure. `text` goes into the hover, which is
    where item ids belong -- printed on the plot they cover the geometry.

    Note what this cannot do: a scatter plot cannot say whether the separation
    is the label or something correlated with it. `geometry.eta_squared` is the
    number that can, and it is printed beside every one of these.
    """
    import plotly.graph_objects as go

    names = list(dict.fromkeys(labels))
    if len(names) > len(SERIES):
        raise ValueError(f"{len(names)} label values exceeds the {len(SERIES)} colours")
    marks = list(symbols) if symbols is not None else None
    shapes = ("circle", "diamond", "square", "x")

    fig = go.Figure()
    xs, ys = list(_to_numpy(x)), list(_to_numpy(y))
    for slot, name in enumerate(names):
        idx = [i for i, lab in enumerate(labels) if lab == name]
        marker = {"size": 9, "color": SERIES[slot], "line": {"width": 0}}
        if marks is not None:
            kinds = list(dict.fromkeys(marks))
            marker["symbol"] = [shapes[kinds.index(marks[i]) % len(shapes)] for i in idx]
        fig.add_trace(go.Scatter(
            x=[xs[i] for i in idx], y=[ys[i] for i in idx],
            mode="markers", name=str(name), marker=marker,
            text=[str(text[i]) for i in idx] if text is not None else None,
            hovertemplate=(f"{name}<br>%{{text}}<br>%{{x:.2f}}, %{{y:.2f}}<extra></extra>"
                           if text is not None
                           else f"{name}<br>%{{x:.2f}}, %{{y:.2f}}<extra></extra>"),
        ))
    return _style(fig, title, xaxis, yaxis)
