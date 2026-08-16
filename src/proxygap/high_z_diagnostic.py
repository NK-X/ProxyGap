"""Pure helpers for the post-run high-z reference-policy diagnostic."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return bool(value)


def select_common_high_z_seed(
    endpoint_rows: Sequence[Mapping[str, Any]],
    *,
    failing_training_seeds: Sequence[int],
) -> dict[str, Any]:
    """Choose the lowest evaluation seed causing high-z in every failed policy."""
    failing = {int(seed) for seed in failing_training_seeds}
    if not failing:
        raise ValueError("At least one failing training seed is required")

    by_evaluation_seed: dict[int, set[int]] = {}
    for row in endpoint_rows:
        if truthy(row.get("high_z_termination", False)):
            by_evaluation_seed.setdefault(int(row["seed"]), set()).add(
                int(row["training_seed"])
            )
    candidates = sorted(
        evaluation_seed
        for evaluation_seed, high_z_seeds in by_evaluation_seed.items()
        if failing.issubset(high_z_seeds)
    )
    if not candidates:
        raise ValueError("No evaluation seed causes high-z in every failed policy")
    selected = candidates[0]
    return {
        "evaluation_seed": selected,
        "selection_rule": (
            "High-z termination in every failed reference policy; lowest "
            "evaluation seed breaks ties."
        ),
        "eligible_evaluation_seeds": candidates,
        "failed_training_seeds": sorted(failing),
        "unexpected_other_high_z_training_seeds": sorted(
            by_evaluation_seed[selected] - failing
        ),
    }


def summarise_step_trace(
    rows: Sequence[Mapping[str, Any]],
    *,
    dt: float,
) -> dict[str, Any]:
    """Derive transparent kinematic descriptors from one complete step trace."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if not rows:
        raise ValueError("A step trace cannot be empty")
    ordered = sorted(rows, key=lambda row: int(row["step_index"]))
    indices = [int(row["step_index"]) for row in ordered]
    expected = list(range(1, len(ordered) + 1))
    if indices != expected:
        raise ValueError("Step indices must be contiguous and start at one")

    heights = [float(row["torso_height"]) for row in ordered]
    x_positions = [float(row["x_position"]) for row in ordered]
    tilts = [float(row["torso_tilt_rad"]) for row in ordered]
    actions = [float(row["squared_action_step"]) for row in ordered]
    saturation = [float(row["action_saturation_fraction_step"]) for row in ordered]
    inverted = [tilt >= math.pi / 2 for tilt in tilts]
    low_posture = [height < 0.3 for height in heights]
    vertical_velocities = [
        (current - previous) / dt
        for previous, current in zip(heights[:-1], heights[1:])
    ]
    horizontal_velocities = [
        (current - previous) / dt
        for previous, current in zip(x_positions[:-1], x_positions[1:])
    ]
    one_second_start = max(0, len(ordered) - int(round(1.0 / dt)) - 1)
    final = ordered[-1]
    return {
        "episode_length": len(ordered),
        "duration_seconds": len(ordered) * dt,
        "termination_category": str(final["termination_category"]),
        "terminated": truthy(final["terminated"]),
        "truncated": truthy(final["truncated"]),
        "initial_logged_torso_height": heights[0],
        "terminal_torso_height": heights[-1],
        "maximum_torso_height": max(heights),
        "minimum_torso_height": min(heights),
        "torso_height_gain_last_second": heights[-1] - heights[one_second_start],
        "terminal_vertical_velocity": (
            vertical_velocities[-1] if vertical_velocities else float("nan")
        ),
        "maximum_upward_velocity": (
            max(vertical_velocities) if vertical_velocities else float("nan")
        ),
        "terminal_forward_velocity": (
            horizontal_velocities[-1] if horizontal_velocities else float("nan")
        ),
        "maximum_torso_tilt_rad": max(tilts),
        "terminal_torso_tilt_rad": tilts[-1],
        "proportion_steps_torso_tilt_ge_90_deg": sum(inverted) / len(inverted),
        "proportion_steps_torso_height_below_0p3": sum(low_posture)
        / len(low_posture),
        "longest_consecutive_inverted_steps": longest_true_run(inverted),
        "longest_consecutive_low_posture_steps": longest_true_run(low_posture),
        "mean_squared_action_last_second": sum(actions[one_second_start:])
        / len(actions[one_second_start:]),
        "maximum_action_saturation_fraction": max(saturation),
    }


def longest_true_run(values: Sequence[bool]) -> int:
    """Return the longest consecutive run of true values."""
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def finite_summary(summary: Mapping[str, Any]) -> bool:
    """Return whether every numeric diagnostic is finite."""
    for value in summary.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                return False
    return True
