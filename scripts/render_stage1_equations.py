"""Render report equations from LaTeX-style Matplotlib mathtext."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EQUATIONS = {
    "reward_equation": (
        r"$R_w(\tau)=\sum_{t=1}^{T}\left(r_t^{\mathrm{forward}}+"
        r"r_t^{\mathrm{survive}}+r_t^{\mathrm{contact}}-w\Vert a_t\Vert_2^2\right)$"
    ),
    "matched_rescore_equation": (
        r"$\Delta R_w=\overline{R_w(\pi_w)}-\overline{R_w(\pi_{0.5})}$"
    ),
    "locomotion_equations": (
        r"$r_t^{\mathrm{forward}}=\frac{x_{t+1}-x_t}{\Delta t},\qquad "
        r"D_x=x_T-x_0,\qquad \bar v_x=\frac{x_T-x_0}{T\Delta t}$"
    ),
    "diagnostic_equations": (
        r"$E_{\mathrm{path}}=\frac{x_T-x_0}{\sum_{t=1}^{T}"
        r"\sqrt{(x_t-x_{t-1})^2+(y_t-y_{t-1})^2}},\quad "
        r"S=\frac{1}{8T}\sum_{t,j}\mathbb{1}(|a_{t,j}|\geq0.95),\quad "
        r"Q=\frac{1}{32(T-1)}\sum_{t=2}^{T}\Vert a_t-a_{t-1}\Vert_2^2$"
    ),
    "seed_consistency_equation": (
        r"$\Pr(X\geq k\mid n,p=0.5)=\sum_{i=k}^{n}"
        r"\frac{n!}{i!(n-i)!}2^{-n},\qquad "
        r"\Pr(X\geq4\mid n=5)=0.1875,\qquad "
        r"\Pr(X\geq7\mid n=8)=0.0352$"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})
    for name, equation in EQUATIONS.items():
        width = 12.0 if name in {"diagnostic_equations", "seed_consistency_equation"} else 9.2
        fig = plt.figure(figsize=(width, 0.72), dpi=180)
        fig.patch.set_alpha(0.0)
        fig.text(0.5, 0.5, equation, ha="center", va="center", fontsize=18)
        output = output_dir / f"{name}.png"
        fig.savefig(output, transparent=True, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
