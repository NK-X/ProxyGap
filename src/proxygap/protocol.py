"""Validation gates for the prospective ProxyGap v2 protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .experiment import resolve_ppo_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_prospective_protocol(config: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return every blocker that prevents a prospective protocol freeze."""
    blockers: list[dict[str, str]] = []

    def add(code: str, message: str) -> None:
        blockers.append({"code": code, "message": message})

    proposal = dict(config.get("proposal", {}))
    proposal_path = Path(str(proposal.get("path", "")))
    expected_hash = str(proposal.get("sha256", "")).upper()
    if not proposal_path.is_file():
        add("PROPOSAL_FILE_MISSING", "The controlling Proposal_G6 file is unavailable.")
    elif not expected_hash or _sha256(proposal_path) != expected_hash:
        add("PROPOSAL_HASH_MISMATCH", "The controlling Proposal_G6 hash does not match.")

    partitions = dict(config.get("seed_partitions", {}))
    seed_sets: dict[str, set[int]] = {
        key: {int(value) for value in values}
        for key, values in partitions.items()
        if isinstance(values, list)
    }
    required_partitions = {
        "development_training_seeds",
        "development_evaluation_seeds",
        "held_out_training_seeds",
        "held_out_evaluation_seeds",
    }
    missing_partitions = sorted(required_partitions - set(seed_sets))
    if missing_partitions:
        add("SEED_PARTITION_MISSING", f"Missing seed partitions: {missing_partitions}")
    seed_names = sorted(seed_sets)
    for index, name_a in enumerate(seed_names):
        for name_b in seed_names[index + 1 :]:
            overlap = sorted(seed_sets[name_a] & seed_sets[name_b])
            if overlap:
                add(
                    "SEED_PARTITION_OVERLAP",
                    f"{name_a} and {name_b} overlap: {overlap}",
                )

    try:
        resolve_ppo_config(dict(config.get("ppo", {})), require_complete=True)
    except (TypeError, ValueError) as error:
        add("PPO_CONFIG_INCOMPLETE", str(error))

    intervention = dict(config.get("intervention", {}))
    for key in (
        "effort_scale",
        "orientation_scale_rad",
        "effort_cap_lambda",
        "orientation_cap_lambda",
    ):
        value = intervention.get(key)
        if value is None or float(value) <= 0:
            add("INTERVENTION_PARAMETER_UNLOCKED", f"{key} must be frozen and positive.")

    metric_parameters = dict(config.get("metric_parameters", {}))
    distance_min = metric_parameters.get("effort_distance_min")
    if distance_min is None or float(distance_min) <= 0:
        add(
            "EFFORT_DISTANCE_MIN_UNLOCKED",
            "effort_distance_min must be frozen before the timed pilot.",
        )

    decisions = dict(config.get("claim_decisions_requiring_user_approval", {}))
    analysis_route = decisions.get("analysis_route")
    if analysis_route not in {"descriptive_only", "margin_based_mitigation"}:
        add("ANALYSIS_ROUTE_UNDECIDED", "Select a descriptive or margin-based claim route.")
    if analysis_route == "margin_based_mitigation":
        if decisions.get("smallest_meaningful_progress_improvement") is None:
            add("PROGRESS_MARGIN_UNLOCKED", "Freeze the progress improvement margin.")
        if not decisions.get("protected_harm_margins"):
            add("PROTECTED_HARM_MARGINS_UNLOCKED", "Freeze all protected harm margins.")
    attribution_scope = decisions.get("attribution_scope")
    if attribution_scope not in {
        "combined_only_no_component_attribution",
        "effort_orientation_combined_ablation",
    }:
        add("ATTRIBUTION_SCOPE_UNDECIDED", "Select the intervention attribution scope.")

    records = dict(config.get("records", {}))
    for key in ("training_monitor", "evaluation_step_logs_gzip", "resolved_config"):
        if records.get(key) is not True:
            add("RECORDING_REQUIREMENT_DISABLED", f"{key} must be enabled.")

    video_rule = dict(config.get("video_rule", {}))
    if not video_rule.get("policy_selection") or not video_rule.get("episode_selection"):
        add("VIDEO_SELECTION_RULE_MISSING", "The deterministic video rule is incomplete.")

    return blockers


def protocol_freeze_status(config: Mapping[str, Any]) -> dict[str, Any]:
    blockers = validate_prospective_protocol(config)
    return {
        "status": "ready" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
    }
