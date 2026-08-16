"""Validation and analysis for the stage-one fresh reference diagnostic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from statistics import fmean
from typing import Any


SHAPING_FIELDS = (
    "forward_progress_shaping_weight",
    "lateral_drift_shaping_weight",
    "effort_shaping_weight",
    "orientation_shaping_weight",
    "reward_shaping_sum",
    "reward_forward_shaping_sum",
    "reward_lateral_shaping_sum",
    "reward_effort_shaping_sum",
    "reward_orientation_shaping_sum",
)


def numeric(value: Any) -> float:
    """Convert CSV values, including booleans, to finite-analysis numbers."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
    return float(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_reference_config(config: Mapping[str, Any]) -> None:
    """Reject scope drift before the reference-only run is initialised."""
    _require(config.get("schema_version") == 6, "Expected schema_version 6")
    _require(
        config.get("stage_scope")
        == "stage_1_reference_competence_only_no_candidate_no_shaping",
        "The V6 stage scope changed",
    )
    weights = [float(value) for value in config.get("ctrl_cost_weights", [])]
    _require(weights == [0.5], "V6 permits only the 0.5 reference condition")
    _require(config.get("formal_launch") == "prohibited", "Formal launch drifted")
    _require(config.get("shaping_launch") == "prohibited", "Shaping launch drifted")
    _require(config.get("normalisation_enabled") is False, "Normalisation drifted")

    training_seeds = [int(value) for value in config.get("training_seeds", [])]
    _require(len(training_seeds) == 5, "Exactly five training seeds are required")
    _require(len(set(training_seeds)) == 5, "Training seeds must be unique")
    formal_seeds = {
        int(value) for value in config.get("reserved_formal_training_seeds", [])
    }
    _require(
        not set(training_seeds) & formal_seeds,
        "Development and reserved formal seeds overlap",
    )

    evaluation_seeds = [int(value) for value in config.get("evaluation_seeds", [])]
    expected_episodes = int(config.get("eval_episodes_per_checkpoint", 0))
    _require(expected_episodes == 20, "V6 requires 20 evaluation episodes")
    _require(len(evaluation_seeds) == expected_episodes, "Evaluation seed count drifted")
    _require(len(set(evaluation_seeds)) == expected_episodes, "Evaluation seeds repeat")
    _require(
        evaluation_seeds
        == list(
            range(
                int(config.get("evaluation_seed_base", -1)),
                int(config.get("evaluation_seed_base", -1)) + expected_episodes,
            )
        ),
        "Evaluation seeds do not match the declared consecutive seed base",
    )
    _require(
        not set(evaluation_seeds) & set(training_seeds),
        "Training and evaluation seeds overlap",
    )

    _require(
        int(config.get("timesteps_per_condition", 0)) == 1_000_000,
        "Training budget drifted",
    )
    _require(
        [int(value) for value in config.get("checkpoint_timesteps", [])]
        == [250_000, 500_000, 750_000, 1_000_000],
        "Checkpoint set drifted",
    )
    _require(
        int(config.get("eval_max_episode_steps", 0)) == 1_000,
        "Evaluation horizon drifted",
    )

    reward = config.get("reward", {})
    for field in (
        "forward_progress_shaping_weight",
        "lateral_drift_shaping_weight",
        "effort_shaping_weight",
        "orientation_shaping_weight",
    ):
        _require(math.isclose(float(reward.get(field, math.nan)), 0.0), f"{field} is non-zero")
    _require(
        math.isclose(float(reward.get("common_rescore_ctrl_cost_weight", math.nan)), 0.5),
        "Common rescore weight drifted",
    )

    ppo = config.get("ppo", {})
    _require(ppo.get("use_sde") is False, "State-dependent exploration drifted")
    _require(ppo.get("policy") == "MlpPolicy", "PPO policy drifted")
    _require(ppo.get("policy_kwargs", {}).get("activation_fn") == "Tanh", "Activation drifted")
    _require(
        ppo.get("policy_kwargs", {}).get("net_arch") == [64, 64],
        "Network architecture drifted",
    )

    gate = config.get("reference_competence_gate", {})
    per_policy = gate.get("per_policy", {})
    configuration = gate.get("configuration_level", {})
    _require(int(gate.get("checkpoint", 0)) == 1_000_000, "Gate endpoint drifted")
    _require(
        math.isclose(float(per_policy.get("unhealthy_termination_rate_max", math.nan)), 0.2),
        "Health threshold drifted",
    )
    _require(
        math.isclose(
            float(per_policy.get("mean_forward_velocity_min_position_units_per_second", math.nan)),
            0.1,
        ),
        "Velocity threshold drifted",
    )
    _require(
        int(configuration.get("supported_min_passing_policies", 0)) == 4
        and int(configuration.get("inconclusive_min_passing_policies", 0)) == 2
        and int(configuration.get("failed_max_passing_policies", -1)) == 1
        and int(configuration.get("total_policies", 0)) == 5,
        "Configuration-level decision rule drifted",
    )


def classify_configuration(passing_policies: int, total_policies: int = 5) -> str:
    """Apply the frozen descriptive 4/5, 2-3/5, 0-1/5 decision rule."""
    if total_policies != 5 or not 0 <= passing_policies <= total_policies:
        raise ValueError("The V6 classification requires zero to five passing policies")
    if passing_policies >= 4:
        return "supported"
    if passing_policies >= 2:
        return "inconclusive"
    return "failed"


def summarise_reference_endpoint(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Aggregate nested evaluation episodes within each 1M reference policy."""
    validate_reference_config(config)
    endpoint = int(config["reference_competence_gate"]["checkpoint"])
    expected_training_seeds = [int(value) for value in config["training_seeds"]]
    expected_evaluation_seeds = {int(value) for value in config["evaluation_seeds"]}
    expected_episodes = int(config["eval_episodes_per_checkpoint"])
    health_max = float(
        config["reference_competence_gate"]["per_policy"][
            "unhealthy_termination_rate_max"
        ]
    )
    velocity_min = float(
        config["reference_competence_gate"]["per_policy"][
            "mean_forward_velocity_min_position_units_per_second"
        ]
    )

    endpoint_rows = [
        row for row in rows if int(numeric(row["target_timesteps"])) == endpoint
    ]
    _require(
        len(endpoint_rows) == len(expected_training_seeds) * expected_episodes,
        "Unexpected number of endpoint evaluation rows",
    )
    keys = [
        (
            int(numeric(row["training_seed"])),
            int(numeric(row["target_timesteps"])),
            int(numeric(row["seed"])),
        )
        for row in endpoint_rows
    ]
    _require(len(keys) == len(set(keys)), "Duplicate endpoint evaluation key")

    policy_results: list[dict[str, Any]] = []
    for training_seed in expected_training_seeds:
        policy_rows = [
            row
            for row in endpoint_rows
            if int(numeric(row["training_seed"])) == training_seed
        ]
        _require(len(policy_rows) == expected_episodes, "Policy episode count drifted")
        _require(
            {int(numeric(row["seed"])) for row in policy_rows}
            == expected_evaluation_seeds,
            "Policy evaluation seeds drifted",
        )
        for row in policy_rows:
            _require(str(row["condition_id"]) == "reference", "Non-reference row found")
            _require(
                math.isclose(numeric(row["ctrl_cost_weight"]), 0.5, abs_tol=1e-12),
                "Reference control-cost weight drifted",
            )
            for field in SHAPING_FIELDS:
                if field in row and row[field] not in (None, ""):
                    _require(
                        math.isclose(numeric(row[field]), 0.0, abs_tol=1e-12),
                        f"Non-zero shaping detected in {field}",
                    )

        unhealthy = [numeric(row["unhealthy_termination"]) for row in policy_rows]
        velocities = [numeric(row["mean_forward_velocity"]) for row in policy_rows]
        _require(all(math.isfinite(value) for value in unhealthy + velocities), "Non-finite gate metric")
        unhealthy_rate = float(fmean(unhealthy))
        mean_velocity = float(fmean(velocities))
        health_pass = unhealthy_rate <= health_max
        velocity_pass = mean_velocity >= velocity_min
        policy_results.append(
            {
                "training_seed": training_seed,
                "evaluation_episodes": expected_episodes,
                "unhealthy_termination_count": int(round(sum(unhealthy))),
                "unhealthy_termination_rate": unhealthy_rate,
                "mean_forward_velocity": mean_velocity,
                "health_gate_pass": health_pass,
                "velocity_gate_pass": velocity_pass,
                "joint_gate_pass": health_pass and velocity_pass,
            }
        )

    passing = sum(bool(row["joint_gate_pass"]) for row in policy_results)
    return {
        "primary_endpoint": endpoint,
        "independent_replication_unit": "training seed / independently trained policy",
        "evaluation_episode_role": "nested repeated observation",
        "policy_results": policy_results,
        "passing_policies": passing,
        "total_policies": len(policy_results),
        "classification": classify_configuration(passing, len(policy_results)),
    }
