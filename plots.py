import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science", "ieee"])
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "savefig.dpi": 600,
    "text.usetex": False,
})

FigureSpec = Tuple[plt.Figure, plt.Axes]


def _load_weights(path: Path) -> Tuple[Dict, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find weights file: {path}")
    payload = np.load(path, allow_pickle=True)
    metadata = payload["metadata"].item()
    weights = np.asarray(payload["weight"], dtype=np.float64)
    if weights.ndim != 2:
        weights = np.stack(weights)
    return metadata, weights


def _fixed_point_residual(weights: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    final_w = weights[-1]
    deltas = weights - final_w
    numer = np.linalg.norm(deltas, axis=1, ord=2)

    epochs = np.arange(1, weights.shape[0] + 1)
    return epochs, numer


def _metadata_text(metadata: Dict) -> str:
    opt_cfg = metadata.get("optimizer", {}) or {}
    sched_cfg = metadata.get("lr_scheduler", {}) or {}
    lines = [
        f"Run: {metadata.get('name', 'N/A')}",
        f"Model: {metadata.get('type', 'unknown')}",
        f"Optimizer: {opt_cfg.get('type', 'N/A').split('.')[-1]}",
    ]
    if sched_cfg:
        lines.append(f"Scheduler: {sched_cfg.get('type', 'N/A').split('.')[-1]}")
    return "\n".join(lines)


def _style_axes(ax: plt.Axes, ylabel: str):
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)


def _plot_fixed_point(
    epochs: np.ndarray,
    residual: np.ndarray,
    metadata: Dict,
    title: str | None = None,
) -> FigureSpec:
    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
    ax.plot(epochs, residual, color="#1f77b4", linewidth=1.8, label="Residual")
    ax.scatter(epochs[-1], residual[-1], color="#d62728", edgecolors="white", linewidth=0.6, zorder=5, label="Fixed point")
    ax.annotate(
        fr"Final: {residual[-1]:.2e}",
        xy=(epochs[-1], residual[-1]),
        xytext=(-30, 12),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=0.9),
        fontsize=9,
        color="#2b2b2b",
    )
    bbox = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#4d4d4d", alpha=0.92, linewidth=0.6)
    ax.text(0.02, 0.98, _metadata_text(metadata), transform=ax.transAxes, va="top", fontsize=9, bbox=bbox)
    _style_axes(ax, r"Fixed-point residual $\|w_k - w^*\|_2 / \|w^*\|_2$")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    return fig, ax


def _save_figure(fig: plt.Figure, output_path: Path, fmt: str, dpi: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight", backend="agg")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed-point residual plot from saved weights.")
    parser.add_argument("--weights", type=Path, default=r'weights/train_lasso_1.npz', help="Path to *.npz weights produced by Trainer")
    parser.add_argument("--output", type=Path, default=None, help="Output path (directory or file stem). Defaults to figures/<run>_fixed_point.png")
    parser.add_argument("--fmt", choices=["png", "pdf", "svg"], default="png", help="Figure format")
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI for raster outputs")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title override")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata, weights = _load_weights(args.weights)
    epochs, residual = _fixed_point_residual(weights)
    default_name = f"{args.weights.stem}_fixed_point"
    if args.output is None:
        output_path = Path("figures") / default_name
    else:
        output_path = args.output
        if output_path.is_dir() or output_path.suffix == "":
            output_path = output_path / default_name
    fig, _ = _plot_fixed_point(epochs, residual, metadata, args.title)
    _save_figure(fig, output_path, args.fmt, args.dpi)


if __name__ == "__main__":
    main()
