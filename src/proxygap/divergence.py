"""Within-objective screening for proxy-diagnostic divergence candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_EXTERNAL_HARM_METRICS: dict[str, str] = {
    "negative_forward_path_efficiency": "higher_is_worse",
    "lateral_drift_mean_abs": "higher_is_worse",
    "torso_tilt_rms": "higher_is_worse",
    "unhealthy_termination": "higher_is_worse",
    "action_saturation_rate": "higher_is_worse",
}


@dataclass(frozen=True)
class SeedDivergenceContrast:
    """Late-minus-early changes for one trained policy trajectory."""

    ctrl_cost_weight: float
    training_seed: int
    condition_objective_delta: float
    diagnostic_deltas: dict[str, float]
    worsened_diagnostics: tuple[str, ...]
    candidate_pattern: bool


@dataclass(frozen=True)
class WeightDivergenceScreen:
    """Descriptive screening result for one fixed reward definition."""

    ctrl_cost_weight: float
    training_seed_count: int
    candidate_seed_count: int
    reward_gain_seed_count: int
    consistently_worsened_diagnostics: tuple[str, ...]
    candidate_for_confirmation: bool
    seed_contrasts: tuple[SeedDivergenceContrast, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["seed_contrasts"] = [asdict(item) for item in self.seed_contrasts]
        return result


@dataclass(frozen=True)
class PairwiseProxyContrast:
    """Candidate-minus-reference contrast under the candidate's fixed proxy."""

    ctrl_cost_weight: float
    training_seed: int
    candidate_proxy_advantage: float
    diagnostic_deltas: dict[str, float]
    worsened_diagnostics: tuple[str, ...]
    candidate_pattern: bool


@dataclass(frozen=True)
class PairwiseWeightScreen:
    """Cross-policy screen with one reward formula held fixed per coefficient."""

    ctrl_cost_weight: float
    training_seed_count: int
    positive_proxy_seed_count: int
    candidate_seed_count: int
    consistently_worsened_diagnostics: tuple[str, ...]
    candidate_for_confirmation: bool
    seed_contrasts: tuple[PairwiseProxyContrast, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["seed_contrasts"] = [asdict(item) for item in self.seed_contrasts]
        return result


def _mean(values: Sequence[float]) -> float:
    if not values or not np.isfinite(values).all():
        raise ValueError("Every required screening cell must contain finite values.")
    return float(np.mean(values))


def _numeric(value: Any) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
    return float(value)


def screen_divergence_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    early_checkpoints: Sequence[int],
    late_checkpoints: Sequence[int],
    external_harm_metrics: Mapping[str, str] | None = None,
    minimum_reward_gain: float = 0.0,
    minimum_harm_changes: Mapping[str, float] | None = None,
    minimum_consistent_seeds: int = 2,
) -> list[WeightDivergenceScreen]:
    """Screen weights without comparing returns across reward definitions.

    Evaluation episodes are averaged within each policy/checkpoint. Early and
    late windows are then contrasted within the same training seed and fixed
    reward definition. A result is only a candidate for fresh-seed confirmation;
    it is not a formal finding of reward hacking.
    """
    materialised = list(rows)
    if not materialised:
        raise ValueError("Screening requires evaluation rows.")
    early = {int(value) for value in early_checkpoints}
    late = {int(value) for value in late_checkpoints}
    if not early or not late or early & late:
        raise ValueError("Early and late checkpoint sets must be non-empty and disjoint.")
    if minimum_consistent_seeds <= 0:
        raise ValueError("minimum_consistent_seeds must be positive.")
    metrics = dict(external_harm_metrics or DEFAULT_EXTERNAL_HARM_METRICS)
    if not metrics or any(direction != "higher_is_worse" for direction in metrics.values()):
        raise ValueError("Current screening supports explicit higher-is-worse diagnostics only.")
    margins = {metric: 0.0 for metric in metrics}
    margins.update({key: float(value) for key, value in (minimum_harm_changes or {}).items()})
    if any(value < 0 for value in margins.values()):
        raise ValueError("Minimum harm changes must be non-negative.")

    normalised_rows: list[dict[str, Any]] = []
    for source in materialised:
        row = dict(source)
        if "forward_path_efficiency" in row:
            row["negative_forward_path_efficiency"] = -_numeric(
                row["forward_path_efficiency"]
            )
        normalised_rows.append(row)
    materialised = normalised_rows
    required = {
        "ctrl_cost_weight",
        "training_seed",
        "target_timesteps",
        "condition_objective_return",
        *metrics,
    }
    by_policy_checkpoint: dict[tuple[float, int, int], list[Mapping[str, Any]]] = {}
    for row in materialised:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Screening row is missing fields: {missing}")
        target = int(row["target_timesteps"])
        if target not in early | late:
            continue
        key = (float(row["ctrl_cost_weight"]), int(row["training_seed"]), target)
        by_policy_checkpoint.setdefault(key, []).append(row)

    by_weight: dict[float, list[SeedDivergenceContrast]] = {}
    policy_keys = sorted({(weight, seed) for weight, seed, _ in by_policy_checkpoint})
    for weight, seed in policy_keys:
        checkpoint_means: dict[int, dict[str, float]] = {}
        for target in sorted(early | late):
            cell = by_policy_checkpoint.get((weight, seed, target))
            if not cell:
                raise ValueError(
                    f"Missing checkpoint {target} for weight={weight}, training_seed={seed}."
                )
            checkpoint_means[target] = {
                name: _mean([_numeric(row[name]) for row in cell])
                for name in ["condition_objective_return", *metrics]
            }
        early_reward = _mean(
            [checkpoint_means[target]["condition_objective_return"] for target in early]
        )
        late_reward = _mean(
            [checkpoint_means[target]["condition_objective_return"] for target in late]
        )
        diagnostic_deltas = {
            metric: _mean([checkpoint_means[target][metric] for target in late])
            - _mean([checkpoint_means[target][metric] for target in early])
            for metric in metrics
        }
        worsened = tuple(
            metric
            for metric, delta in diagnostic_deltas.items()
            if delta > margins.get(metric, 0.0)
        )
        reward_delta = late_reward - early_reward
        contrast = SeedDivergenceContrast(
            ctrl_cost_weight=weight,
            training_seed=seed,
            condition_objective_delta=reward_delta,
            diagnostic_deltas=diagnostic_deltas,
            worsened_diagnostics=worsened,
            candidate_pattern=reward_delta > minimum_reward_gain and bool(worsened),
        )
        by_weight.setdefault(weight, []).append(contrast)

    screens: list[WeightDivergenceScreen] = []
    for weight, contrasts in sorted(by_weight.items()):
        worsened_sets = [set(item.worsened_diagnostics) for item in contrasts]
        consistent = tuple(sorted(set.intersection(*worsened_sets))) if worsened_sets else ()
        reward_gain_count = sum(
            item.condition_objective_delta > minimum_reward_gain for item in contrasts
        )
        candidate_count = sum(item.candidate_pattern for item in contrasts)
        screens.append(
            WeightDivergenceScreen(
                ctrl_cost_weight=weight,
                training_seed_count=len(contrasts),
                candidate_seed_count=candidate_count,
                reward_gain_seed_count=reward_gain_count,
                consistently_worsened_diagnostics=consistent,
                candidate_for_confirmation=(
                    len(contrasts) >= minimum_consistent_seeds
                    and reward_gain_count >= minimum_consistent_seeds
                    and candidate_count >= minimum_consistent_seeds
                    and bool(consistent)
                ),
                seed_contrasts=tuple(contrasts),
            )
        )
    return screens


def choose_minimal_departure_candidate(
    screens: Iterable[WeightDivergenceScreen],
    *,
    eligible_reduced_weights: Iterable[float],
) -> float | None:
    """Choose the largest qualifying reduced weight, or stop with ``None``."""
    eligible = {float(value) for value in eligible_reduced_weights}
    qualifying = [
        item.ctrl_cost_weight
        for item in screens
        if item.candidate_for_confirmation and item.ctrl_cost_weight in eligible
    ]
    return max(qualifying) if qualifying else None


def screen_pairwise_fixed_proxy(
    rows: Iterable[Mapping[str, Any]],
    *,
    checkpoint: int,
    reference_weight: float = 0.5,
    external_harm_metrics: Mapping[str, str] | None = None,
    minimum_consistent_seeds: int = 2,
    minimum_consistent_harm_metrics: int = 2,
) -> list[PairwiseWeightScreen]:
    """Compare candidate and reference policies under one candidate proxy.

    For each candidate coefficient ``w``, both realised trajectory sets are
    rescored using ``R_w``. No return is compared across different formulas.
    External diagnostics remain disaggregated and may show improvements as well
    as harms; a candidate screen is not an overall-performance judgement.
    """
    metrics = dict(external_harm_metrics or DEFAULT_EXTERNAL_HARM_METRICS)
    if minimum_consistent_seeds <= 0 or minimum_consistent_harm_metrics <= 0:
        raise ValueError("Consistency requirements must be positive.")
    required_components = {
        "ctrl_cost_weight",
        "training_seed",
        "target_timesteps",
        "reward_forward_sum",
        "reward_survive_sum",
        "reward_contact_sum",
        "cumulative_squared_action",
        *[name for name in metrics if name != "negative_forward_path_efficiency"],
    }
    materialised: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        missing = sorted(required_components - set(row))
        if missing:
            raise ValueError(f"Pairwise row is missing fields: {missing}")
        if int(row["target_timesteps"]) != int(checkpoint):
            continue
        if "forward_path_efficiency" in row:
            row["negative_forward_path_efficiency"] = -_numeric(
                row["forward_path_efficiency"]
            )
        materialised.append(row)
    if not materialised:
        raise ValueError("No rows match the requested pairwise checkpoint.")

    value_names = [
        "reward_forward_sum",
        "reward_survive_sum",
        "reward_contact_sum",
        "cumulative_squared_action",
        *metrics,
    ]
    cells: dict[tuple[float, int], dict[str, float]] = {}
    keys = sorted(
        {(float(row["ctrl_cost_weight"]), int(row["training_seed"])) for row in materialised}
    )
    for key in keys:
        weight, seed = key
        cell = [
            row
            for row in materialised
            if float(row["ctrl_cost_weight"]) == weight
            and int(row["training_seed"]) == seed
        ]
        cells[key] = {
            name: _mean([_numeric(row[name]) for row in cell]) for name in value_names
        }

    reference_seeds = {
        seed for weight, seed in cells if np.isclose(weight, reference_weight)
    }
    candidate_weights = sorted(
        {weight for weight, _ in cells if not np.isclose(weight, reference_weight)}
    )
    screens: list[PairwiseWeightScreen] = []
    for weight in candidate_weights:
        candidate_seeds = {seed for candidate_weight, seed in cells if candidate_weight == weight}
        if candidate_seeds != reference_seeds:
            raise ValueError(
                f"Candidate {weight} and reference require identical training seeds."
            )
        contrasts: list[PairwiseProxyContrast] = []
        for seed in sorted(reference_seeds):
            candidate = cells[(weight, seed)]
            reference = cells[(reference_weight, seed)]

            def proxy(values: Mapping[str, float]) -> float:
                return float(
                    values["reward_forward_sum"]
                    + values["reward_survive_sum"]
                    + values["reward_contact_sum"]
                    - weight * values["cumulative_squared_action"]
                )

            diagnostic_deltas = {
                metric: candidate[metric] - reference[metric] for metric in metrics
            }
            worsened = tuple(
                metric for metric, delta in diagnostic_deltas.items() if delta > 0.0
            )
            advantage = proxy(candidate) - proxy(reference)
            contrasts.append(
                PairwiseProxyContrast(
                    ctrl_cost_weight=weight,
                    training_seed=seed,
                    candidate_proxy_advantage=advantage,
                    diagnostic_deltas=diagnostic_deltas,
                    worsened_diagnostics=worsened,
                    candidate_pattern=(
                        advantage > 0.0
                        and len(worsened) >= minimum_consistent_harm_metrics
                    ),
                )
            )
        worsened_sets = [set(item.worsened_diagnostics) for item in contrasts]
        consistent = tuple(sorted(set.intersection(*worsened_sets))) if worsened_sets else ()
        proxy_count = sum(item.candidate_proxy_advantage > 0.0 for item in contrasts)
        candidate_count = sum(item.candidate_pattern for item in contrasts)
        screens.append(
            PairwiseWeightScreen(
                ctrl_cost_weight=weight,
                training_seed_count=len(contrasts),
                positive_proxy_seed_count=proxy_count,
                candidate_seed_count=candidate_count,
                consistently_worsened_diagnostics=consistent,
                candidate_for_confirmation=(
                    len(contrasts) >= minimum_consistent_seeds
                    and proxy_count >= minimum_consistent_seeds
                    and candidate_count >= minimum_consistent_seeds
                    and len(consistent) >= minimum_consistent_harm_metrics
                ),
                seed_contrasts=tuple(contrasts),
            )
        )
    return screens
