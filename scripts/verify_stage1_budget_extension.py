"""Independent recomputation for the stage-one budget-extension result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--analysis_root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(value: str) -> float:
    lowered = value.strip().lower()
    if lowered == "true":
        return 1.0
    if lowered == "false":
        return 0.0
    return float(value)


def mean(rows: Iterable[dict[str, str]], field: str) -> float:
    values = [numeric(row[field]) for row in rows]
    if not values:
        raise ValueError(f"No values for {field}")
    return float(fmean(values))


def matched_proxy(rows: list[dict[str, str]], scoring_weight: float) -> float:
    return float(
        mean(rows, "reward_forward_sum")
        + mean(rows, "reward_survive_sum")
        + mean(rows, "reward_contact_sum")
        - float(scoring_weight) * mean(rows, "cumulative_squared_action")
    )


def policy_rows(
    rows: list[dict[str, str]],
    *,
    weight: float,
    training_seed: int,
    checkpoint: int,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if math.isclose(float(row["ctrl_cost_weight"]), weight, abs_tol=1e-12)
        and int(row["training_seed"]) == training_seed
        and int(row["target_timesteps"]) == checkpoint
    ]


def close(actual: float, expected: float, *, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=tolerance):
        raise AssertionError(f"Independent value {actual} != reported value {expected}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    run_root = args.run_root.resolve()
    analysis_root = args.analysis_root.resolve()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    raw = read_csv(run_root / "logs" / "evaluation_metrics.csv")
    reported = json.loads(
        (analysis_root / "stage1_budget_extension_adjudication.json").read_text(
            encoding="utf-8"
        )
    )
    competence_rows = read_csv(analysis_root / "reference_competence_gate.csv")

    endpoint = 1_000_000
    evaluation_seeds = set(config["budget_extension"]["evaluation_seeds"])
    if len(raw) != 180:
        raise AssertionError(f"Raw evaluation row count is {len(raw)}, expected 180")
    raw_keys = {
        (
            float(row["ctrl_cost_weight"]),
            int(row["training_seed"]),
            int(row["target_timesteps"]),
            int(row["seed"]),
        )
        for row in raw
    }
    if len(raw_keys) != 180:
        raise AssertionError("Raw evaluation keys are not unique")

    independent_candidates: dict[str, Any] = {}
    for candidate_weight in [0.21875, 0.125]:
        seed_rows: list[dict[str, float | int]] = []
        reported_rows = {
            int(row["training_seed"]): row
            for row in reported["candidate_audits"][str(candidate_weight)][
                "seed_level_rows"
            ]
        }
        for training_seed in [41101, 41102]:
            reference = policy_rows(
                raw,
                weight=0.5,
                training_seed=training_seed,
                checkpoint=endpoint,
            )
            candidate = policy_rows(
                raw,
                weight=candidate_weight,
                training_seed=training_seed,
                checkpoint=endpoint,
            )
            if len(reference) != 10 or len(candidate) != 10:
                raise AssertionError("Endpoint policy does not contain ten episodes")
            if {int(row["seed"]) for row in reference} != evaluation_seeds:
                raise AssertionError("Reference evaluation seeds do not match the config")
            if {int(row["seed"]) for row in candidate} != evaluation_seeds:
                raise AssertionError("Candidate evaluation seeds do not match the config")
            proxy_delta = matched_proxy(candidate, candidate_weight) - matched_proxy(
                reference, candidate_weight
            )
            progress_delta = mean(candidate, "net_forward_progress") - mean(
                reference, "net_forward_progress"
            )
            efficiency_delta = mean(candidate, "forward_path_efficiency") - mean(
                reference, "forward_path_efficiency"
            )
            tilt_delta = mean(candidate, "torso_tilt_rms") - mean(
                reference, "torso_tilt_rms"
            )
            expected = reported_rows[training_seed]
            close(proxy_delta, float(expected["delta_matched_proxy_return"]))
            close(progress_delta, float(expected["delta_net_forward_progress"]))
            close(
                efficiency_delta,
                float(expected["delta_forward_path_efficiency"]),
            )
            close(tilt_delta, float(expected["delta_torso_tilt_rms"]))
            seed_rows.append(
                {
                    "training_seed": training_seed,
                    "matched_proxy_advantage": proxy_delta,
                    "net_forward_progress_delta": progress_delta,
                    "forward_path_efficiency_delta": efficiency_delta,
                    "torso_tilt_rms_delta": tilt_delta,
                }
            )
        independent_candidates[str(candidate_weight)] = seed_rows

    independent_competence: list[dict[str, Any]] = []
    reported_competence = {
        int(row["training_seed"]): row for row in competence_rows
    }
    for training_seed in [41101, 41102]:
        reference = policy_rows(
            raw,
            weight=0.5,
            training_seed=training_seed,
            checkpoint=endpoint,
        )
        unhealthy_rate = mean(reference, "unhealthy_termination")
        velocity = mean(reference, "mean_forward_velocity")
        joint_pass = unhealthy_rate <= 0.2 and velocity >= 0.1
        expected = reported_competence[training_seed]
        close(unhealthy_rate, float(expected["unhealthy_termination"]))
        close(velocity, float(expected["mean_forward_velocity"]))
        if str(expected["joint_competence_gate_pass"]).lower() != str(joint_pass).lower():
            raise AssertionError("Reference competence Boolean does not match")
        independent_competence.append(
            {
                "training_seed": training_seed,
                "unhealthy_termination_rate": unhealthy_rate,
                "mean_forward_velocity": velocity,
                "joint_gate_pass": joint_pass,
            }
        )

    source_hashes: list[dict[str, Any]] = []
    for source in config["source_policies"]:
        path = (project_root / source["path"]).resolve()
        digest = file_hash(path)
        if digest.lower() != str(source["sha256"]).lower():
            raise AssertionError(f"Source hash changed: {path}")
        source_hashes.append(
            {
                "training_seed": source["training_seed"],
                "ctrl_cost_weight": source["ctrl_cost_weight"],
                "path": str(path),
                "sha256": digest,
            }
        )

    runtime_rows = read_csv(run_root / "logs" / "training_runtime.csv")
    if len(runtime_rows) != 18:
        raise AssertionError("Runtime manifest does not contain 18 checkpoints")
    model_metadata: list[dict[str, Any]] = []
    for row in runtime_rows:
        model_path = Path(row["model_path"])
        if file_hash(model_path).lower() != row["model_sha256"].lower():
            raise AssertionError(f"Continued model hash mismatch: {model_path}")
        model = PPO.load(model_path, device="cpu")
        actual = int(model.num_timesteps)
        expected = int(row["actual_model_timesteps"])
        if actual != expected:
            raise AssertionError(
                f"Model timestep metadata {actual} != runtime record {expected}"
            )
        model_metadata.append(
            {
                "training_seed": int(row["training_seed"]),
                "ctrl_cost_weight": float(row["ctrl_cost_weight"]),
                "target_timesteps": int(row["target_timesteps"]),
                "actual_model_timesteps": actual,
                "sha256": row["model_sha256"],
            }
        )

    verification = {
        "status": "pass",
        "method": (
            "Independent standard-library recomputation from raw episode CSV; "
            "the primary analysis functions were not imported."
        ),
        "raw_episode_rows": len(raw),
        "unique_episode_keys": len(raw_keys),
        "independent_reference_competence": independent_competence,
        "independent_candidate_endpoint_contrasts": independent_candidates,
        "source_model_hashes_verified": len(source_hashes),
        "continued_model_hashes_and_timesteps_verified": len(model_metadata),
        "reported_adjudication_values_matched": True,
        "formal_launch": "prohibited",
        "shaping_launch": "prohibited",
    }
    (analysis_root / "independent_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
