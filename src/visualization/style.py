import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DARK_BG   = "#0E1117"
AXES_BG   = "#1A1D27"
GRID_CLR  = "#2E3145"
TEXT_CLR  = "#E0E0E0"


METRIC_COLORS = {
    "qi":           "#4C9BE8",   # blue
    "w_mhd":        "#E8734C",   # orange
    "iota_edge":    "#5DBE6E",   # green
    "mirror_ratio": "#C875D4",   # purple
    "total":        "#F0C040",   # gold

    "train":        "#4C9BE8",
    "val":          "#E8734C",
    "accuracy":     "#5DBE6E",
}

METRIC_LABELS = {
    "qi":           "QI residual",
    "w_mhd":        "W_MHD",
    "iota_edge":    "Iota edge",
    "mirror_ratio": "Mirror ratio",
    "total":        "Total (weighted)",
}

def apply_style(ax: plt.Axes, title: str, ylabel: str, log_scale: bool = False):
    
    ax.set_facecolor(AXES_BG)
    ax.set_title(title, color=TEXT_CLR, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", color=TEXT_CLR, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT_CLR, fontsize=11)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(
        True, which="both",
        color=GRID_CLR, linewidth=0.6, linestyle="--", alpha=0.7,
    )
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs="all", numticks=10))
    ax.legend(
        facecolor="#262730", edgecolor=GRID_CLR,
        labelcolor=TEXT_CLR, fontsize=9, framealpha=0.9,
    )


def make_fig():
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    return fig, ax