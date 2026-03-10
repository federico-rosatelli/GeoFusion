from __future__ import annotations
from src.visualization.style import (
    DARK_BG, METRIC_COLORS, TEXT_CLR,
    apply_style, make_fig,
)
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go


def plot_stellarator_shape(X, Y, Z, index_sample=0):
    plot_dir = "public/images"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    X = np.concatenate((X, X[0:1, :]), axis=0)
    Y = np.concatenate((Y, Y[0:1, :]), axis=0)
    Z = np.concatenate((Z, Z[0:1, :]), axis=0)

    X = np.concatenate((X, X[:, 0:1]), axis=1)
    Y = np.concatenate((Y, Y[:, 0:1]), axis=1)
    Z = np.concatenate((Z, Z[:, 0:1]), axis=1)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
    fig.update_layout(autosize=True,
                      scene=dict(
                          xaxis_title='X [m]',
                          yaxis_title='Y [m]',
                          zaxis_title='Z [m]',
                          aspectmode='data'
                      ),
                      margin=dict(l=0, r=0, b=0, t=30))
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_coloraxes(showscale=False)
    fig.update_traces(showscale=False)
    fig.update_layout(
        scene=dict(bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    fig.show()
    fig.write_image(os.path.join(plot_dir, f"shapes/stellarator_shape_{index_sample}.png"))



def plot_loss_base(metric_name, history):
    try:
        plot_dir = "public/images"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        if isinstance(history, tuple):
            history_dict = {
                'train_loss': history[0],
                'val_loss': history[1],
                'val_accuracy': history[2]
            }
        elif isinstance(history, dict):
            history_dict = history
        else:
            history_dict = {'train_loss': history, 'val_loss': [], 'val_accuracy': []}

        epochs = range(1, len(history_dict['train_loss']) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history_dict['train_loss'], 'o-', color='#1f77b4', linewidth=2, markersize=5, label='Train Loss')
        plt.title(f"Training Loss: {metric_name.upper()}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{metric_name}_train_loss_plot.png"), dpi=300)
        plt.close()

        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history_dict['val_loss'], 'o-', color='#ff7f0e', linewidth=2, markersize=5, label='Val Loss')
        plt.title(f"Validation Loss: {metric_name.upper()}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{metric_name}_val_loss_plot.png"), dpi=300)
        plt.close()

        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history_dict['val_accuracy'], 'o-', color='#2ca02c', linewidth=2, markersize=5, label='R2 Score')
        plt.title(f"Validation Accuracy (R2): {metric_name.upper()}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("R2 Score", fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{metric_name}_accuracy_plot.png"), dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error loading or plotting loss data for {metric_name}: {e}")



def plot_stellarator_shape(X, Y, Z):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, cmap="viridis", rstride=4, cstride=4,
                    alpha=0.9, antialiased=True)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Plasma Boundary")

    max_range = (
        np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    )
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

def plot_loss(metric_name, history, save_dir = "public/images", show = True, log_scale = False):
    if isinstance(history, tuple):
        h = {
            "train_loss":   list(history[0]),
            "val_loss":     list(history[1]),
            "val_accuracy": list(history[2]) if len(history) > 2 else [],
        }
    elif isinstance(history, dict):
        h = history
    else:
        h = {"train_loss": list(history), "val_loss": [], "val_accuracy": []}


    color = METRIC_COLORS.get(metric_name, METRIC_COLORS["train"])
    label = metric_name.upper().replace("_", " ")
    figs: list[plt.Figure] = []

    train_vals = h.get("train_loss", [])
    if train_vals:
        fig1, ax1 = make_fig()
        epochs = range(1, len(train_vals) + 1)
        ax1.plot(
            epochs, train_vals,
            color=color, linewidth=2.0,
            marker="o", markersize=4,
            label=f"{label} – train",
        )
        apply_style(ax1, f"Train Loss – {label}", "Huber Loss", log_scale=log_scale)
        fig1.tight_layout()
        if save_dir:
            fig1.savefig(
                os.path.join(save_dir, f"{metric_name}_train_loss.png"),
                dpi=150, facecolor=DARK_BG,
            )
        figs.append(fig1)
        if show:
            plt.show()
        else:
            plt.close(fig1)

    val_vals = h.get("val_loss", [])
    if val_vals:
        fig2, ax2 = make_fig()
        epochs = range(1, len(val_vals) + 1)
        ax2.plot(
            epochs, val_vals,
            color=METRIC_COLORS["val"], linewidth=2.0,
            marker="o", markersize=4,
            label=f"{label} – val",
        )
        apply_style(ax2, f"Validation Loss – {label}", "Huber Loss", log_scale=log_scale)
        fig2.tight_layout()
        if save_dir:
            fig2.savefig(
                os.path.join(save_dir, f"{metric_name}_val_loss.png"),
                dpi=150, facecolor=DARK_BG,
            )
        figs.append(fig2)
        if show:
            plt.show()
        else:
            plt.close(fig2)

    acc_vals = h.get("val_accuracy", [])
    if acc_vals:
        fig3, ax3 = make_fig()
        epochs = range(1, len(acc_vals) + 1)
        ax3.plot(
            epochs, acc_vals,
            color=METRIC_COLORS["accuracy"], linewidth=2.0,
            marker="o", markersize=4,
            label=f"{label} – R²",
        )
        ax3.axhline(1.0, color="#555566", linewidth=1, linestyle=":")
        apply_style(ax3, f"Validation R² Accuracy – {label}", "R² Score",
                    log_scale=False)
        fig3.tight_layout()
        if save_dir:
            fig3.savefig(
                os.path.join(save_dir, f"{metric_name}_accuracy.png"),
                dpi=150, facecolor=DARK_BG,
            )
        figs.append(fig3)
        if show:
            plt.show()
        else:
            plt.close(fig3)

    return figs