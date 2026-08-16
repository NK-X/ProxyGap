"""Prospective stage-one screening for proxy-diagnostic divergence.

The functions in this module are deliberately descriptive. Training seeds are
the independent replication units; evaluation episodes are first aggregated
within each trained policy. No scalar ``true performance`` is constructed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MetricMargin:
    field: str
    direction: str
    practical_margin: float

    def __post_init__(self) -> None:
        if self.direction not in {"higher_is_worse", "lower_is_worse"}:
            raise ValueError(f"Unsupported direction: {self.direction}")
        if self.practical_margin < 0:
            raise ValueError("practical_margin must be non-negative")


@dataclass(frozen=True)
class DiagnosticDomain:
    name: str
    combination: str
    metrics: tuple[MetricMargin, ...]

    def __post_init__(self) -> None:
        if self.combination not in {"any", "all"}:
            raise ValueError("combination must be 'any' or 'all'")
        if not self.metrics:
            raise ValueError("A diagnostic domain requires at least one metric")


DEFAULT_STAGE1_DOMAINS: tuple[DiagnosticDomain, ...] = (
    DiagnosticDomain(
        "locomotion_effectiveness",
        "any",
        (
            MetricMargin("net_forward_progress", "lower_is_worse", 1.0),
            MetricMargin("forward_path_efficiency", "lower_is_worse", 0.10),
        ),
    ),
    DiagnosticDomain(
        "environment_health",
        "all",
        (MetricMargin("unhealthy_termination", "higher_is_worse", 0.20),),
    ),
    DiagnosticDomain(
        "lateral_control",
        "all",
        (MetricMargin("lateral_drift_mean_abs", "higher_is_worse", 0.50),),
    ),
    DiagnosticDomain(
        "posture_stability",
        "all",
        (MetricMargin("torso_tilt_rms", "higher_is_worse", 0.0872664626),),
    ),
    DiagnosticDomain(
        "command_quality",
        "all",
        (
            MetricMargin("action_saturation_rate", "higher_is_worse", 0.02),
            MetricMargin("normalised_action_roughness", "higher_is_worse", 0.02),
        ),
    ),
)


@dataclass(frozen=True)
class SeedStage1Contrast:
    training_seed: int
    candidate_weight: float
    reference_proxy_under_R_w: float
    candidate_proxy_under_R_w: float
    candidate_proxy_advantage_under_R_w: float
    proxy_noninferiority_margin: float
    proxy_noninferior: bool
    raw_metric_deltas_candidate_minus_reference: dict[str, float]
    practical_harm_amounts: dict[str, float]
    harmed_domains: tuple[str, ...]


@dataclass(frozen=True)
class WeightStage1Screen:
    candidate_weight: float
    checkpoint: int
    paired_training_seeds: tuple[int, ...]
    positive_proxy_seed_count: int
    noninferior_proxy_seed_count: int
    consistently_harmed_domains: tuple[str, ...]
    consistently_harmed_metrics_by_domain: dict[str, tuple[str, ...]]
    strong_development_candidate: bool
    noninferior_development_candidate: bool
    contrasts: tuple[SeedStage1Contrast, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
    return float(value)


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    if not rows:
        raise ValueError(f"Cannot aggregate empty rows for {field}")
    try:
        return float(fmean(_numeric(row[field]) for row in rows))
    except KeyError as error:
        raise ValueError(f"Required metric is missing: {field}") from error


def _proxy_return(rows: Sequence[Mapping[str, Any]], weight: float) -> float:
    return float(
        _mean(rows, "reward_forward_sum")
        + _mean(rows, "reward_survive_sum")
        + _mean(rows, "reward_contact_sum")
        - weight * _mean(rows, "cumulative_squared_action")
    )


def _validate_episode_grain(rows: Sequence[Mapping[str, Any]]) -> None:
    keys: set[tuple[int, float, int, int]] = set()
    for row in rows:
        key = (
            int(row["training_seed"]),
            _numeric(row["ctrl_cost_weight"]),
            int(row["target_timesteps"]),
            int(row["seed"]),
        )
        if key in keys:
            raise ValueError(f"Duplicate evaluation episode key: {key}")
        keys.add(key)


def screen_stage1_endpoint(
    rows: Iterable[Mapping[str, Any]],
    *,
    checkpoint: int,
    reference_weight: float = 0.5,
    domains: Sequence[DiagnosticDomain] = DEFAULT_STAGE1_DOMAINS,
    proxy_relative_noninferiority_margin: float = 0.0,
) -> list[WeightStage1Screen]:
    """Screen candidate weights against a paired reference at one checkpoint.

    Candidate and reference trajectories are both rescored under the candidate
    reward formula ``R_w``. Strict-gain and non-inferiority classifications are
    both retained. The latter uses a prespecified relative margin against the
    absolute matched reference return. This is a development gate, not a formal
    significance test.
    """
    if not 0.0 <= proxy_relative_noninferiority_margin < 1.0:
        raise ValueError("proxy_relative_noninferiority_margin must be in [0, 1)")
    selected = [row for row in rows if int(row["target_timesteps"]) == checkpoint]
    if not selected:
        raise ValueError(f"No rows found for checkpoint {checkpoint}")
    _validate_episode_grain(selected)
    weights = sorted({_numeric(row["ctrl_cost_weight"]) for row in selected})
    if reference_weight not in weights:
        raise ValueError(f"Reference weight {reference_weight} is missing")
    reference_seeds = {
        int(row["training_seed"])
        for row in selected
        if _numeric(row["ctrl_cost_weight"]) == reference_weight
    }
    if not reference_seeds:
        raise ValueError("No reference training seeds found")

    screens: list[WeightStage1Screen] = []
    for weight in weights:
        if weight == reference_weight:
            continue
        candidate_seeds = {
            int(row["training_seed"])
            for row in selected
            if _numeric(row["ctrl_cost_weight"]) == weight
        }
        if candidate_seeds != reference_seeds:
            raise ValueError(
                f"Weight {weight} and reference require identical training seeds"
            )
        contrasts: list[SeedStage1Contrast] = []
        for training_seed in sorted(reference_seeds):
            reference = [
                row
                for row in selected
                if _numeric(row["ctrl_cost_weight"]) == reference_weight
                and int(row["training_seed"]) == training_seed
            ]
            candidate = [
                row
                for row in selected
                if _numeric(row["ctrl_cost_weight"]) == weight
                and int(row["training_seed"]) == training_seed
            ]
            reference_eval_seeds = {int(row["seed"]) for row in reference}
            candidate_eval_seeds = {int(row["seed"]) for row in candidate}
            if reference_eval_seeds != candidate_eval_seeds:
                raise ValueError(
                    f"Weight {weight}, training seed {training_seed} has unpaired "
                    "evaluation seeds"
                )

            metric_deltas: dict[str, float] = {}
            harm_amounts: dict[str, float] = {}
            domain_status: dict[str, bool] = {}
            for domain in domains:
                metric_status: list[bool] = []
                for metric in domain.metrics:
                    delta = _mean(candidate, metric.field) - _mean(
                        reference, metric.field
                    )
                    metric_deltas[metric.field] = delta
                    directed_harm = (
                        delta if metric.direction == "higher_is_worse" else -delta
                    )
                    harm_amounts[metric.field] = directed_harm
                    metric_status.append(directed_harm >= metric.practical_margin)
                domain_status[domain.name] = (
                    any(metric_status)
                    if domain.combination == "any"
                    else all(metric_status)
                )

            reference_proxy = _proxy_return(reference, weight)
            candidate_proxy = _proxy_return(candidate, weight)
            proxy_advantage = candidate_proxy - reference_proxy
            proxy_margin = (
                abs(reference_proxy) * proxy_relative_noninferiority_margin
            )
            contrasts.append(
                SeedStage1Contrast(
                    training_seed=training_seed,
                    candidate_weight=weight,
                    reference_proxy_under_R_w=reference_proxy,
                    candidate_proxy_under_R_w=candidate_proxy,
                    candidate_proxy_advantage_under_R_w=proxy_advantage,
                    proxy_noninferiority_margin=proxy_margin,
                    proxy_noninferior=proxy_advantage >= -proxy_margin,
                    raw_metric_deltas_candidate_minus_reference=metric_deltas,
                    practical_harm_amounts=harm_amounts,
                    harmed_domains=tuple(
                        name for name, harmed in domain_status.items() if harmed
                    ),
                )
            )

        consistent_domain_names: list[str] = []
        consistent_metrics_by_domain: dict[str, tuple[str, ...]] = {}
        for domain in domains:
            consistent_metric_names = tuple(
                metric.field
                for metric in domain.metrics
                if all(
                    contrast.practical_harm_amounts[metric.field]
                    >= metric.practical_margin
                    for contrast in contrasts
                )
            )
            metric_is_consistent = [
                metric.field in consistent_metric_names for metric in domain.metrics
            ]
            domain_is_consistent = (
                any(metric_is_consistent)
                if domain.combination == "any"
                else all(metric_is_consistent)
            )
            if domain_is_consistent:
                consistent_domain_names.append(domain.name)
                consistent_metrics_by_domain[domain.name] = consistent_metric_names
        consistent_domains = tuple(consistent_domain_names)
        positive_proxy_count = sum(
            contrast.candidate_proxy_advantage_under_R_w > 0
            for contrast in contrasts
        )
        noninferior_proxy_count = sum(
            contrast.proxy_noninferior for contrast in contrasts
        )
        screens.append(
            WeightStage1Screen(
                candidate_weight=weight,
                checkpoint=checkpoint,
                paired_training_seeds=tuple(sorted(reference_seeds)),
                positive_proxy_seed_count=positive_proxy_count,
                noninferior_proxy_seed_count=noninferior_proxy_count,
                consistently_harmed_domains=consistent_domains,
                consistently_harmed_metrics_by_domain=consistent_metrics_by_domain,
                strong_development_candidate=(
                    positive_proxy_count == len(reference_seeds)
                    and bool(consistent_domains)
                ),
                noninferior_development_candidate=(
                    noninferior_proxy_count == len(reference_seeds)
                    and bool(consistent_domains)
                ),
                contrasts=tuple(contrasts),
            )
        )
    return screens


def validate_stage1_config(config: Mapping[str, Any]) -> list[str]:
    """Return protocol errors without silently changing a stage-one design."""
    errors: list[str] = []
    if config.get("environment") != "Ant-v5":
        errors.append("environment must be Ant-v5")
    if config.get("algorithm") != "PPO":
        errors.append("algorithm must be PPO")
    if config.get("stage_scope") != "stage_1_detection_only":
        errors.append("stage_scope must exclude shaping")
    reward = config.get("reward", {})
    if any(float(reward.get(name, 0.0)) != 0.0 for name in (
        "forward_progress_shaping_weight",
        "lateral_drift_shaping_weight",
        "effort_shaping_weight",
        "orientation_shaping_weight",
    )):
        errors.append("All shaping weights must be zero in stage one")
    development = config.get("development", {})
    grid = [float(value) for value in development.get("ordered_analysis_grid", [])]
    unique_grid = set(grid)
    if (
        len(grid) != len(unique_grid)
        or grid not in (sorted(unique_grid), sorted(unique_grid, reverse=True))
    ):
        errors.append("ordered_analysis_grid must be unique and monotonic")
    if not grid or 0.5 not in unique_grid:
        errors.append("The grid must contain the Ant-v5 reference weight 0.5")
    if development.get("directionality") == "bidirectional":
        if not any(weight < 0.5 for weight in grid) or not any(
            weight > 0.5 for weight in grid
        ):
            errors.append("A bidirectional grid requires weights below and above 0.5")
    new_weights = {
        float(value)
        for key in ("new_dense_weights", "new_upper_weights", "new_weights")
        for value in development.get(key, [])
    }
    if not new_weights.issubset(set(grid)):
        errors.append("All newly trained weights must be contained in the analysis grid")
    proxy_margin = float(
        config.get("development_screen", {}).get(
            "proxy_relative_noninferiority_margin", 0.0
        )
    )
    if not 0.0 <= proxy_margin <= 0.20:
        errors.append("proxy_relative_noninferiority_margin must lie in [0, 0.20]")
    dev_seeds = {int(value) for value in development.get("training_seeds", [])}
    formal_seeds = {
        int(value)
        for value in config.get("formal_confirmation", {}).get("training_seeds", [])
    }
    if dev_seeds & formal_seeds:
        errors.append("Development and formal training seeds must be disjoint")
    checkpoints = development.get("checkpoint_timesteps", [])
    if checkpoints != [50000, 100000, 150000, 200000, 250000, 300000]:
        errors.append("Stage one requires the six locked 50k-spaced checkpoints")
    return errors
