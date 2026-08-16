"""Independent raw-CSV and model verification for the V6 reference run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage1_reference_fresh_1m_v6_20260814.json"
DEFAULT_RUN = PROJECT_ROOT / "artifacts" / "exploration" / "stage1_reference_fresh_1m_v6_20260814"
DEFAULT_ANALYSIS = PROJECT_ROOT / "artifacts" / "analysis" / "stage1_reference_fresh_1m_v6_20260814"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--analysis_root", type=Path, default=DEFAULT_ANALYSIS)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
    return float(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def close(actual: float, expected: float, tolerance: float = 1e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=tolerance):
        raise AssertionError(f"Independent value {actual} != reported value {expected}")


def classify(passing: int) -> str:
    if passing >= 4:
        return "supported"
    if passing >= 2:
        return "inconclusive"
    return "failed"


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    run_root = args.run_root.resolve()
    analysis_root = args.analysis_root.resolve()
    raw = read_csv(run_root / "logs" / "evaluation_metrics.csv")
    runtime = read_csv(run_root / "logs" / "training_runtime.csv")
    reported_rows = {
        int(row["training_seed"]): row
        for row in read_csv(analysis_root / "reference_policy_gate.csv")
    }
    adjudication = json.loads(
        (analysis_root / "stage1_reference_adjudication.json").read_text(encoding="utf-8")
    )

    expected_training = [int(value) for value in config["training_seeds"]]
    expected_evaluation = {int(value) for value in config["evaluation_seeds"]}
    expected_checkpoints = [int(value) for value in config["checkpoint_timesteps"]]
    endpoint = 1_000_000
    if len(raw) != 400 or len(runtime) != 20:
        raise AssertionError("Raw row counts do not match the frozen V6 matrix")
    keys = {
        (int(row["training_seed"]), int(row["target_timesteps"]), int(row["seed"]))
        for row in raw
    }
    if len(keys) != 400:
        raise AssertionError("Raw evaluation keys are not unique")

    max_base_error = 0.0
    max_ctrl_error = 0.0
    for row in raw:
        if row["condition_id"] != "reference" or not math.isclose(number(row["ctrl_cost_weight"]), 0.5):
            raise AssertionError("Non-reference condition found")
        for field in (
            "forward_progress_shaping_weight",
            "lateral_drift_shaping_weight",
            "reward_shaping_sum",
            "reward_forward_shaping_sum",
            "reward_lateral_shaping_sum",
            "reward_effort_shaping_sum",
            "reward_orientation_shaping_sum",
        ):
            if not math.isclose(number(row[field]), 0.0, abs_tol=1e-12):
                raise AssertionError(f"Non-zero shaping found in {field}")
        reconstructed = (
            number(row["reward_forward_sum"])
            + number(row["reward_survive_sum"])
            + number(row["reward_contact_sum"])
            + number(row["reward_ctrl_sum"])
        )
        max_base_error = max(max_base_error, abs(number(row["base_proxy_return"]) - reconstructed))
        max_ctrl_error = max(
            max_ctrl_error,
            abs(number(row["reward_ctrl_sum"]) + 0.5 * number(row["cumulative_squared_action"])),
        )
    if max_base_error > 1e-3 or max_ctrl_error > 1e-3:
        raise AssertionError("Independent reward reconstruction exceeded tolerance")

    independent: list[dict[str, Any]] = []
    for training_seed in expected_training:
        all_policy_rows = [row for row in raw if int(row["training_seed"]) == training_seed]
        if {int(row["target_timesteps"]) for row in all_policy_rows} != set(expected_checkpoints):
            raise AssertionError("Checkpoint coverage drifted")
        for checkpoint in expected_checkpoints:
            selected = [row for row in all_policy_rows if int(row["target_timesteps"]) == checkpoint]
            if len(selected) != 20 or {int(row["seed"]) for row in selected} != expected_evaluation:
                raise AssertionError("Evaluation-seed pairing drifted")
        endpoint_rows = [row for row in all_policy_rows if int(row["target_timesteps"]) == endpoint]
        unhealthy = fmean(number(row["unhealthy_termination"]) for row in endpoint_rows)
        velocity = fmean(number(row["mean_forward_velocity"]) for row in endpoint_rows)
        health_pass = unhealthy <= 0.2
        velocity_pass = velocity >= 0.1
        joint_pass = health_pass and velocity_pass
        reported = reported_rows[training_seed]
        close(unhealthy, float(reported["unhealthy_termination_rate"]))
        close(velocity, float(reported["mean_forward_velocity"]))
        if str(reported["joint_gate_pass"]).lower() != str(joint_pass).lower():
            raise AssertionError("Reported policy gate Boolean differs")
        independent.append(
            {
                "training_seed": training_seed,
                "unhealthy_termination_rate": unhealthy,
                "mean_forward_velocity": velocity,
                "joint_gate_pass": joint_pass,
            }
        )

    passing = sum(bool(row["joint_gate_pass"]) for row in independent)
    classification = classify(passing)
    if passing != int(adjudication["passing_policies"]):
        raise AssertionError("Reported passing-policy count differs")
    if classification != adjudication["configuration_classification"]:
        raise AssertionError("Reported configuration classification differs")

    model_records: list[dict[str, Any]] = []
    for row in runtime:
        model_path = Path(row["model_path"])
        if not model_path.exists():
            raise AssertionError(f"Missing model: {model_path}")
        model = PPO.load(model_path, device="cpu")
        actual = int(model.num_timesteps)
        expected = int(row["actual_model_timesteps"])
        if actual != expected:
            raise AssertionError(f"Model timestep metadata mismatch: {model_path}")
        model_records.append(
            {
                "training_seed": int(row["training_seed"]),
                "target_timesteps": int(row["target_timesteps"]),
                "actual_model_timesteps": actual,
                "path": str(model_path),
                "sha256": sha256(model_path),
            }
        )
    if len(model_records) != 20:
        raise AssertionError("Model verification count drifted")

    with (analysis_root / "model_sha256_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model_records[0]))
        writer.writeheader()
        writer.writerows(model_records)
    verification = {
        "status": "pass",
        "method": "Independent standard-library recomputation from raw CSV; primary reference-analysis functions were not imported.",
        "raw_evaluation_rows": len(raw),
        "unique_evaluation_keys": len(keys),
        "independent_policy_results": independent,
        "passing_policies": passing,
        "configuration_classification": classification,
        "max_abs_base_reward_reconstruction_error": max_base_error,
        "max_abs_ctrl_cost_reconstruction_error": max_ctrl_error,
        "model_hashes_and_timesteps_verified": len(model_records),
        "reported_values_matched": True,
        "formal_launch": "prohibited",
        "shaping_launch": "prohibited"
    }
    (analysis_root / "independent_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
