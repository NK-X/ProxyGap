"""Render the stage-one report equations with Matplotlib mathtext.

The PDF builder intentionally runs in the bundled document environment, while
this helper runs in the existing ProxyGap experiment environment where
Matplotlib is already installed.  No experiment data are changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


EQUATIONS = {
    "reward": (
        r"$R_w = F + S + C - wA,\qquad "
        r"A=\sum_{t=1}^{T}\Vert a_t\Vert_2^2$",
        (11.5, 1.0),
    ),
    "matched_contrast": (
        r"$\Delta R_w^{(s)} = "
        r"\overline{R_w(\pi_{w,s})} - "
        r"\overline{R_w(\pi_{0.5,s})}$",
        (11.5, 1.0),
    ),
    "hypothesis": (
        r"$\exists w\in\mathcal{W}\setminus\{0.5\}:\ "
        r"\Delta R_w^{(s)}>0\ \forall s,\quad "
        r"\exists d:\ H_d^{(s)}\geq m_d\ \forall s$",
        (12.5, 1.0),
    ),
    "locomotion": (
        r"$D_x=x_T-x_0,\qquad "
        r"\bar v_x=\frac{x_T-x_0}{T\Delta t},\qquad "
        r"E_{path}=\frac{x_T-x_0}"
        r"{\sum_{t=1}^{T}\sqrt{(x_t-x_{t-1})^2+(y_t-y_{t-1})^2}}$",
        (13.5, 1.25),
    ),
    "diagnostics": (
        r"$L_{mean}=\frac{1}{T}\sum_{t=1}^{T}|y_t-y_0|,\qquad "
        r"\theta_{rms}=\sqrt{\frac{1}{T}\sum_{t=1}^{T}\theta_t^2}$",
        (12.5, 1.1),
    ),
}


def render_equation(text: str, size: tuple[float, float], output_path: Path) -> None:
    fig = plt.figure(figsize=size, facecolor="white")
    fig.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        color="#16324f",
        fontsize=22,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for name, (text, size) in EQUATIONS.items():
        output_path = args.output_dir / f"equation_{name}.png"
        render_equation(text, size, output_path)
        print(output_path.resolve())


if __name__ == "__main__":
    main()
