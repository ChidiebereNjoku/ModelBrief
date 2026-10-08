"""Bar chart of per-feature distribution drift (PSI)."""

import matplotlib.pyplot as plt

_COLORS = {"stable": "#4C956C", "moderate_shift": "#E8A33D", "significant_shift": "#D64545"}


def plot_drift(features, top_n=20):
    """Plot per-feature PSI as a horizontal bar chart, most-drifted on top."""
    items = list(features)[:top_n][::-1]
    fig, ax = plt.subplots()
    ax.barh(
        [f["feature"] for f in items],
        [f["psi"] for f in items],
        color=[_COLORS.get(f["status"], "#888888") for f in items],
    )
    ax.axvline(0.1, linestyle="--", color="gray", linewidth=1)
    ax.axvline(0.25, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Feature distribution drift")
    fig.tight_layout()
    return fig
