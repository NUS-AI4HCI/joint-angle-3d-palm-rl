from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing import event_accumulator


DEFAULT_TAGS: tuple[str, ...] = (
    "eval/mean_final_distance",
    "eval/score",
    "eval/success_rate",
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    "rollout/success_rate",
)

TITLE_MAP = {
    "eval/mean_final_distance": "Eval Mean Final Distance",
    "eval/score": "Eval Score",
    "eval/success_rate": "Eval Success Rate",
    "rollout/ep_len_mean": "Rollout Episode Length Mean",
    "rollout/ep_rew_mean": "Rollout Episode Reward Mean",
    "rollout/success_rate": "Rollout Success Rate",
}

YLABEL_MAP = {
    "eval/mean_final_distance": "Distance (m)",
    "eval/score": "Score",
    "eval/success_rate": "Rate",
    "rollout/ep_len_mean": "Steps",
    "rollout/ep_rew_mean": "Reward",
    "rollout/success_rate": "Rate",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot a paper-ready summary figure from one TensorBoard log directory.")
    p.add_argument(
        "--logdir",
        type=str,
        default="joint_angle_3d_palm_rl/tb_logs/exp_20260303_3",
        help="TensorBoard log directory or event file.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="joint_angle_3d_palm_rl/figures/exp_20260303_3_summary.png",
        help="Output image path.",
    )
    p.add_argument(
        "--pdf-output",
        type=str,
        default="joint_angle_3d_palm_rl/figures/exp_20260303_3_summary.pdf",
        help="Optional PDF output path. Use empty string to disable.",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Moving-average window size for the bold curve.",
    )
    p.add_argument(
        "--style",
        type=str,
        default="tensorboard-like",
        choices=("tensorboard-like", "paper"),
        help="Plot style preset.",
    )
    p.add_argument(
        "--tb-smoothing",
        type=float,
        default=0.0,
        help="TensorBoard-like EMA smoothing weight in [0, 1). 0 means raw curve.",
    )
    p.add_argument(
        "--ignore-outliers",
        action="store_true",
        default=True,
        help="Use percentile-based y-limits, similar to TensorBoard's ignore-outliers view.",
    )
    p.add_argument(
        "--ylim-lower-pct",
        type=float,
        default=5.0,
        help="Lower percentile for robust y-limits.",
    )
    p.add_argument(
        "--ylim-upper-pct",
        type=float,
        default=95.0,
        help="Upper percentile for robust y-limits.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG dpi.",
    )
    return p.parse_args()


def _resolve_event_path(logdir: str) -> Path:
    p = Path(logdir).expanduser()
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(f"log path not found: {p}")
    event_files = sorted(x for x in p.iterdir() if x.name.startswith("events.out.tfevents."))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event file found under: {p}")
    return event_files[-1]


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    w = int(max(1, window))
    if arr.size == 0 or w <= 1 or arr.size < w:
        return arr.copy()
    kernel = np.ones(w, dtype=np.float64) / float(w)
    left = w // 2
    right = w - 1 - left
    padded = np.pad(arr, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _tensorboard_ema(values: np.ndarray, weight: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    w = float(np.clip(weight, 0.0, 0.999999))
    if arr.size == 0 or w <= 1e-12:
        return arr.copy()
    out = np.empty_like(arr)
    last = float(arr[0])
    debias = 1.0
    for idx, val in enumerate(arr):
        last = last * w + (1.0 - w) * float(val)
        debias = debias * w + (1.0 - w)
        out[idx] = last / max(debias, 1e-12)
    return out


def _robust_ylim(values: np.ndarray, lower_pct: float, upper_pct: float) -> tuple[float, float] | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo_pct = float(np.clip(lower_pct, 0.0, 100.0))
    hi_pct = float(np.clip(upper_pct, 0.0, 100.0))
    if hi_pct < lo_pct:
        lo_pct, hi_pct = hi_pct, lo_pct
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if abs(hi - lo) < 1e-12:
        pad = max(abs(lo) * 0.05, 1e-3)
        return lo - pad, hi + pad
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def _step_formatter() -> FuncFormatter:
    def _fmt(x: float, _pos: int) -> str:
        ax = abs(float(x))
        if ax >= 1_000_000:
            return f"{x / 1_000_000:.1f}M"
        if ax >= 1_000:
            return f"{x / 1_000:.0f}k"
        return f"{x:.0f}"

    return FuncFormatter(_fmt)


def _load_scalars(event_path: Path, tags: Sequence[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ea = event_accumulator.EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    missing = [tag for tag in tags if tag not in available]
    if missing:
        raise KeyError(f"missing scalar tags: {missing}")
    for tag in tags:
        events = ea.Scalars(tag)
        steps = np.asarray([e.step for e in events], dtype=np.float64)
        vals = np.asarray([e.value for e in events], dtype=np.float64)
        out[tag] = (steps, vals)
    return out


def _iter_axes_grid(axes: np.ndarray) -> Iterable[plt.Axes]:
    for ax in np.asarray(axes).reshape(-1):
        yield ax


def main() -> None:
    args = _parse_args()
    event_path = _resolve_event_path(args.logdir)
    data = _load_scalars(event_path, DEFAULT_TAGS)
    exp_name = event_path.parent.name if event_path.parent.name else event_path.name

    if args.style == "paper":
        plt.rcParams.update(
            {
                "font.family": "DejaVu Serif",
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 8,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
    else:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 9,
                "axes.titlesize": 11,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

    fig, axes = plt.subplots(3, 2, figsize=(13.2, 10.2), sharex=False, constrained_layout=True)
    fig.suptitle(f"Training Summary: {exp_name}", fontsize=15, fontweight="bold")

    raw_color = "#9bb8d3"
    smooth_color = "#0b4f6c"
    highlight_color = "#b02e0c"

    for ax, tag in zip(_iter_axes_grid(axes), DEFAULT_TAGS):
        steps, values = data[tag]
        if args.style == "paper":
            smooth = _moving_average(values, int(args.smooth_window))
            ax.plot(steps, values, color=raw_color, linewidth=1.4, alpha=0.55, label="raw")
            ax.plot(steps, smooth, color=smooth_color, linewidth=2.4, label=f"smoothed ({int(max(1, args.smooth_window))})")
            basis_for_ylim = smooth
        else:
            tb_curve = _tensorboard_ema(values, float(args.tb_smoothing))
            ax.plot(steps, tb_curve, color=highlight_color, linewidth=1.9, alpha=0.95)
            basis_for_ylim = tb_curve

        if values.size > 0:
            end_y = basis_for_ylim[-1]
            ax.scatter([steps[-1]], [end_y], s=18, color=highlight_color, zorder=3)

        ax.set_title(TITLE_MAP.get(tag, tag))
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(YLABEL_MAP.get(tag, "Value"))
        ax.xaxis.set_major_formatter(_step_formatter())
        if args.style == "paper":
            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.28)
        else:
            ax.grid(True, linestyle="-", linewidth=0.7, alpha=0.18)

        if args.ignore_outliers and "success_rate" not in tag:
            ylim = _robust_ylim(
                basis_for_ylim,
                lower_pct=float(args.ylim_lower_pct),
                upper_pct=float(args.ylim_upper_pct),
            )
            if ylim is not None:
                ax.set_ylim(*ylim)
        elif "success_rate" in tag and args.style == "paper":
            ax.set_ylim(-0.02, 1.02)

        if args.style == "tensorboard-like":
            ax.set_xlim(left=0.0)

    if args.style == "paper":
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=False)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(args.dpi), bbox_inches="tight")

    pdf_output = str(args.pdf_output).strip()
    if pdf_output:
        pdf_path = Path(pdf_output).expanduser()
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight")

    print(f"[saved] {output_path}")
    if pdf_output:
        print(f"[saved] {Path(pdf_output).expanduser()}")


if __name__ == "__main__":
    main()
