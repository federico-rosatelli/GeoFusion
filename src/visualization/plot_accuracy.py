import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.ml.multi_surrogate import METRICS
from src.visualization.style import (
    DARK_BG, AXES_BG, GRID_CLR, TEXT_CLR,
    METRIC_COLORS, METRIC_LABELS,
    apply_style, make_fig,
)

_DEFAULT_SINGLE_LOGS = {
    "qi":           "models/qi_model_loss.csv",
    "w_mhd":        "models/w_mhd_model_loss.csv",
    "iota_edge":    "models/iota_edge_model_loss.csv",
    "mirror_ratio": "models/mirror_ratio_model_loss.csv",
}
_DEFAULT_MULTI_LOG = "models/multi_output_model_loss.csv"

def max_r2_single(log_paths):

    results = {}
    for metric, path in log_paths.items():
        if not os.path.exists(path):
            print(f"CSV not found: {path}  | skipping '{metric}'")
            continue
        df = pd.read_csv(path)
        col = "val_accuracy" if "val_accuracy" in df.columns else df.columns[-1]
        results[metric] = float(df[col].max())
    return results


def max_r2_multi(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Multi-output log not found: {log_path}")
    df = pd.read_csv(log_path)
    results: dict[str, float] = {}
    for m in METRICS:
        col = f"{m}_r2"
        if col in df.columns:
            results[m] = float(df[col].max())
    return results


def bar_style(ax: plt.Axes, title):

    ax.set_facecolor(AXES_BG)
    ax.set_title(title, color=TEXT_CLR, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Max Validation R²", color=TEXT_CLR, fontsize=11)
    ax.tick_params(colors=TEXT_CLR, labelsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="#555566", linewidth=1, linestyle=":")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


def annotate_bars(ax: plt.Axes, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.3f}",
            ha="center", va="bottom",
            color=TEXT_CLR, fontsize=9, fontweight="bold",
        )

def plot_single_accuracy_bars(log_paths = None, save_dir = "public/images", show = True,
):
    """
    Bar chart of the maximum R² reached by each of the 4 single surrogates.

    Args:
        log_paths: Mapping metric → CSV path.  Defaults to the standard
                   ``models/<metric>_model_loss.csv`` convention.
        save_dir:  Output directory for PNG (``None`` → skip).
        show:      Call ``plt.show()`` if True.

    Returns:
        The matplotlib Figure.
    """
    paths = log_paths or _DEFAULT_SINGLE_LOGS
    r2 = max_r2_single(paths)

    if not r2:
        raise ValueError("No CSV logs found; cannot build bar chart.")

    fig, ax = make_fig()
    x = np.arange(len(r2))
    labels = [METRIC_LABELS.get(m, m) for m in r2]
    colors = [METRIC_COLORS.get(m, "#888888") for m in r2]
    values = list(r2.values())

    bars = ax.bar(x, values, color=colors, width=0.55, zorder=3,
                  edgecolor=GRID_CLR, linewidth=0.8)
    annotate_bars(ax, bars)
    bar_style(ax, "Single Models – Max Validation R²")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT_CLR)

    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "single_accuracy_bars.png"),
                    dpi=150, facecolor=DARK_BG)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_multi_accuracy_bars(log_path = None, save_dir = "public/images", show = True):
    """
    Bar chart of the maximum per-metric R² reached by the multi-output model.
    """
    path = log_path or _DEFAULT_MULTI_LOG
    r2 = max_r2_multi(path)

    if not r2:
        raise ValueError("No R² columns found in the multi-output log.")

    fig, ax = make_fig()
    x = np.arange(len(r2))
    labels = [METRIC_LABELS.get(m, m) for m in r2]
    colors = [METRIC_COLORS.get(m, "#888888") for m in r2]
    values = list(r2.values())

    bars = ax.bar(x, values, color=colors, width=0.55, zorder=3,
                  edgecolor=GRID_CLR, linewidth=0.8)
    annotate_bars(ax, bars)
    bar_style(ax, "Multi-Output Model – Max Validation R² per Metric")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT_CLR)

    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "multi_accuracy_bars.png"),
                    dpi=150, facecolor=DARK_BG)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_accuracy_comparison(single_log_paths = None, multi_log_path = None, save_dir = "public/images", show = True):
    """
    Plot accuracy comparison between single and multi-output models.
    """

    single_r2 = max_r2_single(single_log_paths or _DEFAULT_SINGLE_LOGS)
    multi_r2  = max_r2_multi(multi_log_path   or _DEFAULT_MULTI_LOG)

    # align to the same metrics
    common = [m for m in METRICS if m in single_r2 and m in multi_r2]
    if not common:
        raise ValueError("No overlapping metrics found between single and multi logs.")

    fig, ax = make_fig()
    fig.set_size_inches(11, 5)          # slightly wider for two groups

    n = len(common)
    x = np.arange(n)
    w = 0.35                             # bar half-width

    for i, m in enumerate(common):
        color = METRIC_COLORS.get(m, "#888888")

        light = lighten(color, 0.35)
        b_single = ax.bar(
            x[i] - w / 2, single_r2[m],
            width=w, color=light, zorder=3,
            edgecolor=color, linewidth=1.2,
        )
        
        b_multi = ax.bar(
            x[i] + w / 2, multi_r2[m],
            width=w, color=color, zorder=3,
            edgecolor=GRID_CLR, linewidth=0.8,
        )
        annotate_bars(ax, b_single)
        annotate_bars(ax, b_multi)

    bar_style(ax, "Single vs Multi-Output – Max Validation R²")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METRIC_LABELS.get(m, m) for m in common], color=TEXT_CLR
    )

    # manual legend
    legend_handles = [
        mpatches.Patch(facecolor=lighten(METRIC_COLORS.get(common[0], "#888"), 0.35),
                       edgecolor=METRIC_COLORS.get(common[0], "#888"), label="Single model"),
        mpatches.Patch(facecolor=METRIC_COLORS.get(common[0], "#888"),
                       edgecolor=GRID_CLR, label="Multi-output model"),
    ]
    ax.legend(handles=legend_handles, facecolor="#262730", edgecolor=GRID_CLR,
              labelcolor=TEXT_CLR, fontsize=9, framealpha=0.9)

    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "accuracy_comparison_bars.png"),
                    dpi=150, facecolor=DARK_BG)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def lighten(hex_color, factor = 0.4) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"