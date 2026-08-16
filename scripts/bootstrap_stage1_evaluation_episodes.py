"""Quantify paired evaluation-perturbation uncertainty at the endpoint.

The bootstrap resamples paired evaluation seeds within each frozen trained
policy.  It does not estimate uncertainty across training seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "net_forward_progress": ("lower_is_worse", 1.0),
    "forward_path_efficiency": ("lower_is_worse", 0.10),
    "unhealthy_termination": ("higher_is_worse", 0.20),
    "lateral_drift_mean_abs": ("higher_is_worse", 0.50),
    "torso_tilt_rms": ("higher_is_worse", 0.0872664626),
    "action_saturation_rate": ("higher_is_worse", 0.02),
    "normalised_action_roughness": ("higher_is_worse", 0.02),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint", type=int, default=300_000)
    parser.add_argument("--bootstrap_replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap_seed", type=int, default=8_142_026)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean(
    values: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, np.ndarray]:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two paired evaluation episodes are required")
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    bootstrap = values[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper), bootstrap


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("Use at least 1,000 bootstrap replicates")
    input_csv = Path(args.input_csv).resolve()
    result_json = Path(args.result_json).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_csv)
    data = data[data["target_timesteps"] == args.checkpoint].copy()
    result = json.loads(result_json.read_text(encoding="utf-8"))
    strong_weights = sorted(
        screen["candidate_weight"]
        for screen in result["endpoint_screens"]
        if screen["strong_development_candidate"]
    )
    if not strong_weights:
        raise ValueError("No strong development candidates exist to bootstrap")

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.bootstrap_seed)
    for weight in strong_weights:
        candidate = data[data["ctrl_cost_weight"] == weight]
        reference = data[data["ctrl_cost_weight"] == 0.5]
        for training_seed in sorted(candidate["training_seed"].unique()):
            candidate_seed = candidate[candidate["training_seed"] == training_seed]
            reference_seed = reference[reference["training_seed"] == training_seed]
            merged = candidate_seed.merge(
                reference_seed,
                on="seed",
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            if len(merged) != 10:
                raise ValueError(
                    f"Expected 10 paired episodes for w={weight}, seed={training_seed}"
                )

            candidate_proxy = (
                merged["reward_forward_sum_candidate"]
                + merged["reward_survive_sum_candidate"]
                + merged["reward_contact_sum_candidate"]
                - weight * merged["cumulative_squared_action_candidate"]
            )
            reference_proxy = (
                merged["reward_forward_sum_reference"]
                + merged["reward_survive_sum_reference"]
                + merged["reward_contact_sum_reference"]
                - weight * merged["cumulative_squared_action_reference"]
            )
            proxy_deltas = (candidate_proxy - reference_proxy).to_numpy(float)
            mean, lower, upper, bootstrap = bootstrap_mean(
                proxy_deltas, replicates=args.bootstrap_replicates, rng=rng
            )
            rows.append(
                {
                    "candidate_weight": weight,
                    "training_seed": int(training_seed),
                    "quantity": "proxy_advantage_under_R_w",
                    "paired_episode_count": len(merged),
                    "mean_delta_or_directed_harm": mean,
                    "bootstrap_95pct_lower": lower,
                    "bootstrap_95pct_upper": upper,
                    "decision_boundary": 0.0,
                    "bootstrap_fraction_above_boundary": float(
                        np.mean(bootstrap > 0.0)
                    ),
                }
            )

            for metric, (direction, margin) in METRICS.items():
                raw_delta = (
                    merged[f"{metric}_candidate"].astype(float)
                    - merged[f"{metric}_reference"].astype(float)
                ).to_numpy()
                directed_harm = raw_delta if direction == "higher_is_worse" else -raw_delta
                mean, lower, upper, bootstrap = bootstrap_mean(
                    directed_harm, replicates=args.bootstrap_replicates, rng=rng
                )
                rows.append(
                    {
                        "candidate_weight": weight,
                        "training_seed": int(training_seed),
                        "quantity": metric,
                        "paired_episode_count": len(merged),
                        "mean_delta_or_directed_harm": mean,
                        "bootstrap_95pct_lower": lower,
                        "bootstrap_95pct_upper": upper,
                        "decision_boundary": margin,
                        "bootstrap_fraction_above_boundary": float(
                            np.mean(bootstrap >= margin)
                        ),
                    }
                )

    output_csv = output_root / "endpoint_paired_evaluation_bootstrap.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    manifest = {
        "status": "paired_evaluation_bootstrap_complete",
        "scope": "nested evaluation-seed perturbation only",
        "input_csv": str(input_csv),
        "input_csv_sha256": sha256(input_csv),
        "result_json": str(result_json),
        "result_json_sha256": sha256(result_json),
        "checkpoint": args.checkpoint,
        "strong_candidate_weights": strong_weights,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "row_count": len(rows),
        "output_csv": str(output_csv),
        "output_csv_sha256": sha256(output_csv),
        "claim_boundary": (
            "Intervals describe sensitivity to the ten paired initial-state "
            "perturbations within each trained policy. They are not confidence "
            "intervals across independently trained policies."
        ),
    }
    (output_root / "bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
