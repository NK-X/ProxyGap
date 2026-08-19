"""Auditable zero-column observation expansion for Stable-Baselines3 PPO.

The transfer preserves every learned policy and value parameter while adding
new observation columns to the two first MLP layers.  The appended weights are
initialised to exactly zero, so the migrated policy is initially independent
of the new observations.  Optimiser moments are deliberately not restored.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch


FIRST_OBSERVATION_LAYER_WEIGHTS = frozenset(
    {
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    }
)


def _flat_box_dimension(space: gym.Space, *, label: str) -> int:
    if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
        raise TypeError(f"{label} must be a one-dimensional Box space")
    return int(space.shape[0])


def _maximum_absolute_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(
            f"Cannot compare tensors with shapes {tuple(left.shape)} and "
            f"{tuple(right.shape)}"
        )
    if left.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(left - right)).detach().cpu().item())


def transfer_ppo_with_appended_observations(
    source_model: PPO,
    target_model: PPO,
    *,
    appended_feature_names: Sequence[str],
    restore_num_timesteps: bool = True,
) -> dict[str, Any]:
    """Copy a PPO policy into a wider observation space using zero columns.

    The source and target must have identical one-dimensional Box action
    spaces and identical policy state keys.  Only the two first policy/value
    MLP weight matrices may differ in shape.  This fail-closed contract avoids
    silently accepting an architecture change alongside the observation
    intervention.
    """

    source_dimension = _flat_box_dimension(
        source_model.observation_space,
        label="source observation space",
    )
    target_dimension = _flat_box_dimension(
        target_model.observation_space,
        label="target observation space",
    )
    feature_names = tuple(str(name) for name in appended_feature_names)
    if not feature_names or any(not name for name in feature_names):
        raise ValueError("At least one non-empty appended feature name is required")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Appended feature names must be unique")
    appended_columns = len(feature_names)
    if target_dimension != source_dimension + appended_columns:
        raise ValueError(
            "Target observation dimension must equal source dimension plus "
            f"{appended_columns}: {source_dimension} -> {target_dimension}"
        )

    source_action_dimension = _flat_box_dimension(
        source_model.action_space,
        label="source action space",
    )
    target_action_dimension = _flat_box_dimension(
        target_model.action_space,
        label="target action space",
    )
    if source_action_dimension != target_action_dimension:
        raise ValueError("Source and target action dimensions differ")
    np.testing.assert_allclose(
        np.asarray(source_model.action_space.low),
        np.asarray(target_model.action_space.low),
        rtol=0.0,
        atol=0.0,
        err_msg="Source and target action lower bounds differ",
    )
    np.testing.assert_allclose(
        np.asarray(source_model.action_space.high),
        np.asarray(target_model.action_space.high),
        rtol=0.0,
        atol=0.0,
        err_msg="Source and target action upper bounds differ",
    )

    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    if source_keys != target_keys:
        missing = sorted(source_keys - target_keys)
        unexpected = sorted(target_keys - source_keys)
        raise ValueError(
            "Source and target policy parameter keys differ: "
            f"missing={missing}, unexpected={unexpected}"
        )

    copied: list[str] = []
    expanded: list[str] = []
    appended_column_max_abs_by_tensor: dict[str, float] = {}
    for name, source_value in source_state.items():
        target_value = target_state[name]
        if target_value.shape == source_value.shape:
            target_state[name] = source_value.detach().clone()
            copied.append(name)
            continue

        compatible = (
            name in FIRST_OBSERVATION_LAYER_WEIGHTS
            and target_value.ndim == 2
            and source_value.ndim == 2
            and target_value.shape[0] == source_value.shape[0]
            and target_value.shape[1]
            == source_value.shape[1] + appended_columns
        )
        if not compatible:
            raise ValueError(
                f"Unsupported observation transfer shape for {name}: "
                f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
            )
        expanded_value = torch.zeros_like(target_value)
        expanded_value[:, :source_dimension] = source_value.detach()
        target_state[name] = expanded_value
        appended_column_max_abs_by_tensor[name] = float(
            torch.max(torch.abs(expanded_value[:, source_dimension:]))
            .detach()
            .cpu()
            .item()
        )
        expanded.append(name)

    if set(expanded) != set(FIRST_OBSERVATION_LAYER_WEIGHTS):
        raise ValueError(
            "Exactly the first policy and value observation layers must expand; "
            f"observed {sorted(expanded)}"
        )
    target_model.policy.load_state_dict(target_state, strict=True)

    optimiser_entries_before_clear = len(target_model.policy.optimizer.state)
    target_model.policy.optimizer.state.clear()
    optimiser_entries_after_clear = len(target_model.policy.optimizer.state)
    if restore_num_timesteps:
        target_model.num_timesteps = int(source_model.num_timesteps)

    return {
        "schema_version": "proxygap-ppo-appended-observation-transfer-v1",
        "source_observation_dimension": source_dimension,
        "target_observation_dimension": target_dimension,
        "appended_observation_columns": appended_columns,
        "appended_feature_names": list(feature_names),
        "action_dimension": target_action_dimension,
        "copied_parameter_tensors": sorted(copied),
        "expanded_parameter_tensors": sorted(expanded),
        "appended_column_max_abs_by_tensor": appended_column_max_abs_by_tensor,
        "previous_columns_copied_exactly": True,
        "new_columns_initialised_to_zero": True,
        "optimizer_state_restored": False,
        "optimizer_state_entries_before_clear": optimiser_entries_before_clear,
        "optimizer_state_entries_after_clear": optimiser_entries_after_clear,
        "source_num_timesteps": int(source_model.num_timesteps),
        "target_num_timesteps": int(target_model.num_timesteps),
    }


def verify_ppo_appended_observation_equivalence(
    source_model: PPO,
    target_model: PPO,
    *,
    source_observations: np.ndarray,
    target_observations: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    """Verify policy distribution, deterministic action and value parity."""

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Equivalence tolerance must be finite and non-negative")
    source_dimension = _flat_box_dimension(
        source_model.observation_space,
        label="source observation space",
    )
    target_dimension = _flat_box_dimension(
        target_model.observation_space,
        label="target observation space",
    )
    source_values = np.asarray(source_observations, dtype=np.float32)
    target_values = np.asarray(target_observations, dtype=np.float32)
    if source_values.ndim == 1:
        source_values = source_values[None, :]
    if target_values.ndim == 1:
        target_values = target_values[None, :]
    if source_values.ndim != 2 or source_values.shape[1] != source_dimension:
        raise ValueError("Source observations have the wrong shape")
    if target_values.ndim != 2 or target_values.shape[1] != target_dimension:
        raise ValueError("Target observations have the wrong shape")
    if source_values.shape[0] != target_values.shape[0]:
        raise ValueError("Source and target observation batch sizes differ")
    prefix_error = float(
        np.max(np.abs(source_values - target_values[:, :source_dimension]))
    )
    if prefix_error > tolerance:
        raise ValueError(
            f"Target observation prefix differs from source by {prefix_error}"
        )

    source_tensor, _ = source_model.policy.obs_to_tensor(source_values)
    target_tensor, _ = target_model.policy.obs_to_tensor(target_values)
    with torch.no_grad():
        source_distribution = source_model.policy.get_distribution(source_tensor)
        target_distribution = target_model.policy.get_distribution(target_tensor)
        source_torch_distribution = source_distribution.distribution
        target_torch_distribution = target_distribution.distribution
        for attribute in ("loc", "scale"):
            if not hasattr(source_torch_distribution, attribute) or not hasattr(
                target_torch_distribution,
                attribute,
            ):
                raise TypeError(
                    "Equivalence verification requires a diagonal Gaussian "
                    f"distribution exposing {attribute}"
                )
        distribution_location_error = _maximum_absolute_error(
            source_torch_distribution.loc,
            target_torch_distribution.loc,
        )
        distribution_scale_error = _maximum_absolute_error(
            source_torch_distribution.scale,
            target_torch_distribution.scale,
        )
        deterministic_action_error = _maximum_absolute_error(
            source_distribution.get_actions(deterministic=True),
            target_distribution.get_actions(deterministic=True),
        )
        value_error = _maximum_absolute_error(
            source_model.policy.predict_values(source_tensor),
            target_model.policy.predict_values(target_tensor),
        )

    errors = {
        "observation_prefix_max_abs_error": prefix_error,
        "action_distribution_location_max_abs_error": (
            distribution_location_error
        ),
        "action_distribution_scale_max_abs_error": distribution_scale_error,
        "deterministic_action_max_abs_error": deterministic_action_error,
        "value_prediction_max_abs_error": value_error,
    }
    maximum_error = max(errors.values())
    return {
        **errors,
        "maximum_equivalence_error": maximum_error,
        "tolerance": float(tolerance),
        "equivalent_within_tolerance": bool(maximum_error <= tolerance),
        "batch_size": int(source_values.shape[0]),
    }
