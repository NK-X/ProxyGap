"""Assemble fixed midpoint frames from the predeclared stage-one videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_root = Path(args.video_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    conditions = [
        ("reference", "w = 0.5"),
        ("ctrl_0p21875", "w = 0.21875"),
        ("ctrl_0p125", "w = 0.125"),
    ]
    seeds = [41101, 41102]
    figure, axes = plt.subplots(3, 2, figsize=(10.4, 9.2))
    for row, (condition_id, label) in enumerate(conditions):
        for column, seed in enumerate(seeds):
            path = video_root / f"{condition_id}_seed{seed}_eval51101_mid.png"
            if not path.exists():
                raise FileNotFoundError(path)
            axis = axes[row, column]
            axis.imshow(plt.imread(path))
            axis.set_axis_off()
            if row == 0:
                axis.set_title(f"Training seed {seed}", fontsize=12, pad=7)
            axis.text(
                0.01,
                0.99,
                label,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="white",
                bbox={"facecolor": "#111820", "edgecolor": "none", "pad": 3},
            )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, wspace=0.02, hspace=0.04)
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
