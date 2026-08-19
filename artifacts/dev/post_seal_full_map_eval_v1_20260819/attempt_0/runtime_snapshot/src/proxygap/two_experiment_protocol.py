"""Validation gates for the two-experiment ProxyGap design."""

from __future__ import annotations

from typing import Any, Mapping


def validate_two_experiment_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    """Report development readiness separately from held-out readiness."""
    development_blockers: list[str] = []
    heldout_blockers: list[str] = []
    detection = dict(config.get("experiment_1_detection", {}))
    core = [float(value) for value in detection.get("core_ctrl_cost_weights", [])]
    if core != [0.5, 0.375, 0.25, 0.125]:
        development_blockers.append("Core weights must remain [0.5, 0.375, 0.25, 0.125].")
    if detection.get("boundary_is_eligible_for_primary_candidate") is not False:
        development_blockers.append("The 0.0625 boundary must not be primary-candidate eligible.")
    early = set(detection.get("early_window", []))
    late = set(detection.get("late_window", []))
    checkpoints = set(detection.get("checkpoint_timesteps", []))
    if not early or not late or early & late or not (early | late) <= checkpoints:
        development_blockers.append("Early and late windows must be disjoint checkpoint subsets.")
    rule = dict(detection.get("screening_rule", {}))
    if rule.get("within_fixed_weight_only") is not True:
        development_blockers.append("Detection must compare reward within a fixed weight only.")
    if not rule.get("no_candidate_rule"):
        development_blockers.append("A no-candidate stopping rule is required.")
    development_seeds = set(detection.get("development_training_seeds", []))
    heldout = dict(config.get("heldout_confirmation", {}))
    heldout_seeds = set(heldout.get("training_seeds", []))
    if not development_seeds or not heldout_seeds or development_seeds & heldout_seeds:
        development_blockers.append("Development and held-out training seeds must be non-empty and disjoint.")

    shaping = dict(config.get("experiment_2_shaping", {}))
    if shaping.get("same_ctrl_cost_weight_as_detected_condition") is not True:
        development_blockers.append("Shaping must retain the diagnosed control-cost weight.")
    prohibited = set(shaping.get("prohibited_signals", []))
    if not {"forward_reward", "squared_action_effort"} <= prohibited:
        development_blockers.append("Forward duplication and effort re-penalisation must be prohibited.")

    candidate = heldout.get("candidate_ctrl_cost_weight")
    eligible = set(rule.get("eligible_reduced_weights", []))
    if candidate is None:
        heldout_blockers.append("No development-screened candidate coefficient has been locked.")
    elif float(candidate) not in {float(value) for value in eligible}:
        heldout_blockers.append("The candidate coefficient is outside the eligible core range.")
    for name in (
        "lateral_scale",
        "lateral_cap_lambda",
        "orientation_scale_rad",
        "orientation_cap_lambda",
    ):
        value = shaping.get(name)
        if value is None:
            heldout_blockers.append(f"{name} is not locked.")
        elif float(value) < 0:
            heldout_blockers.append(f"{name} must be non-negative.")

    return {
        "development_status": "ready" if not development_blockers else "blocked",
        "heldout_status": "ready" if not development_blockers and not heldout_blockers else "blocked",
        "development_blockers": development_blockers,
        "heldout_blockers": heldout_blockers,
    }
