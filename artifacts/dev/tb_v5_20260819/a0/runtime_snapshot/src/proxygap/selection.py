"""Leakage-resistant development-stage coefficient selection.

The selector deliberately consumes only a fixed benchmark proxy.  Behavioural
diagnostics are not accepted as selection inputs and must remain held out until
the selected coefficient is evaluated on fresh seeds.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isclose
from typing import Any, Iterable, Mapping


SELECTION_METRIC = "common_rescored_return"
SELECTION_INPUT_FIELDS = frozenset(
    {"ctrl_cost_weight", "training_seed", "checkpoint_fraction", SELECTION_METRIC}
)
PROTECTED_DIAGNOSTICS = frozenset(
    {
        "net_forward_progress",
        "lateral_drift_mean_abs",
        "lateral_drift_max_abs",
        "torso_tilt_rms",
        "torso_tilt_p95",
        "unhealthy_termination",
        "fall",
        "action_saturation_rate",
        "episode_length",
    }
)


@dataclass(frozen=True)
class CandidateSummary:
    """One candidate's development estimate, aggregated by training seed."""

    ctrl_cost_weight: float
    training_seed_count: int
    policy_mean_values: tuple[float, ...]
    mean_proxy: float


@dataclass(frozen=True)
class SelectionResult:
    """Auditable result of the predeclared development selection rule."""

    selection_metric: str
    selected_ctrl_cost_weight: float
    tie_break_rule: str
    candidates: tuple[CandidateSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_metric": self.selection_metric,
            "selected_ctrl_cost_weight": self.selected_ctrl_cost_weight,
            "tie_break_rule": self.tie_break_rule,
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }


def _float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Selection rows require {key!r}.")
    return float(value)


def _int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Selection rows require {key!r}.")
    return int(value)


def _validate_rows(rows: Iterable[Mapping[str, Any]], candidates: tuple[float, ...]) -> list[Mapping[str, Any]]:
    materialised = list(rows)
    if not materialised:
        raise ValueError("At least one development evaluation row is required.")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Candidate coefficients must be non-empty and unique.")
    if not any(isclose(value, 0.5, rel_tol=0.0, abs_tol=1e-12) for value in candidates):
        raise ValueError("The fixed 0.5 benchmark comparator must be a candidate.")
    required = SELECTION_INPUT_FIELDS
    for row in materialised:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Selection row is missing required fields: {missing}")
        if not isclose(_float(row, "checkpoint_fraction"), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Weight selection must use final-checkpoint rows only.")
        if _float(row, "ctrl_cost_weight") not in candidates:
            raise ValueError("A selection row contains a coefficient outside the candidate grid.")
        metric = _float(row, SELECTION_METRIC)
        if metric != metric:  # NaN
            raise ValueError("Selection proxy values must be finite.")
        if any(key in row for key in PROTECTED_DIAGNOSTICS):
            raise ValueError(
                "Selection rows must not contain protected diagnostics; keep them held out."
            )
    return materialised


def project_selection_inputs(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop evaluation-only fields before coefficient selection.

    Full evaluation logs intentionally contain protected diagnostics.  This
    explicit projection makes the information boundary visible and auditable;
    the selector then validates that only these four fields are present.
    """
    projected: list[dict[str, Any]] = []
    for row in rows:
        missing = sorted(SELECTION_INPUT_FIELDS - set(row))
        if missing:
            raise ValueError(f"Evaluation row is missing selection fields: {missing}")
        projected.append({key: row[key] for key in SELECTION_INPUT_FIELDS})
    return projected


def select_best_tested_coefficient(
    rows: Iterable[Mapping[str, Any]],
    *,
    candidates: Iterable[float],
) -> SelectionResult:
    """Select the best tested coefficient using only the fixed benchmark proxy.

    Evaluation episodes are first averaged within each training seed, so they
    cannot be mistaken for independent policy replications.  Candidate means
    are then averaged across development training seeds.  Exact ties prefer the
    coefficient closest to the default benchmark 0.5, then the smaller value.
    """
    candidate_values = tuple(float(value) for value in candidates)
    materialised = _validate_rows(rows, candidate_values)
    by_candidate_seed: dict[float, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in materialised:
        weight = _float(row, "ctrl_cost_weight")
        seed = _int(row, "training_seed")
        by_candidate_seed[weight][seed].append(_float(row, SELECTION_METRIC))

    summaries: list[CandidateSummary] = []
    for weight in candidate_values:
        by_seed = by_candidate_seed.get(weight, {})
        if not by_seed:
            raise ValueError(f"Candidate {weight} has no development rows.")
        policy_means = tuple(
            sum(values) / len(values) for _, values in sorted(by_seed.items())
        )
        summaries.append(
            CandidateSummary(
                ctrl_cost_weight=weight,
                training_seed_count=len(policy_means),
                policy_mean_values=policy_means,
                mean_proxy=sum(policy_means) / len(policy_means),
            )
        )

    selected = min(
        summaries,
        key=lambda item: (-item.mean_proxy, abs(item.ctrl_cost_weight - 0.5), item.ctrl_cost_weight),
    )
    return SelectionResult(
        selection_metric=SELECTION_METRIC,
        selected_ctrl_cost_weight=selected.ctrl_cost_weight,
        tie_break_rule="higher mean fixed-proxy value; ties prefer coefficient closest to 0.5, then smaller coefficient",
        candidates=tuple(summaries),
    )
