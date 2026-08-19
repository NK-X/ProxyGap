from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from torch import nn

from proxygap.ppo_observation_transfer import (
    FIRST_OBSERVATION_LAYER_WEIGHTS,
    transfer_ppo_with_appended_observations,
    verify_ppo_appended_observation_equivalence,
)


class StaticBoxEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, observation_dimension: int) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dimension,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )


def make_model(observation_dimension: int, *, seed: int) -> PPO:
    return PPO(
        "MlpPolicy",
        StaticBoxEnv(observation_dimension),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={"net_arch": [64, 64], "activation_fn": nn.Tanh},
        seed=seed,
        device="cpu",
        verbose=0,
    )


def preview_feature_names() -> list[str]:
    return [
        *(f"relative_height_{index}" for index in range(9)),
        "terrain_normal_forward",
        "terrain_normal_left",
        "terrain_normal_up",
        "signed_forward_slope_rad",
    ]


def test_122_to_135_transfer_preserves_policy_distribution_and_value() -> None:
    source = make_model(122, seed=101)
    target = make_model(135, seed=202)
    source.num_timesteps = 2_465_792

    manifest = transfer_ppo_with_appended_observations(
        source,
        target,
        appended_feature_names=preview_feature_names(),
    )

    rng = np.random.default_rng(303)
    source_observations = rng.normal(size=(7, 122)).astype(np.float32)
    preview_observations = rng.normal(size=(7, 13)).astype(np.float32)
    target_observations = np.concatenate(
        (source_observations, preview_observations),
        axis=1,
    )
    parity = verify_ppo_appended_observation_equivalence(
        source,
        target,
        source_observations=source_observations,
        target_observations=target_observations,
        tolerance=1e-7,
    )

    assert manifest["source_observation_dimension"] == 122
    assert manifest["target_observation_dimension"] == 135
    assert manifest["appended_observation_columns"] == 13
    assert manifest["appended_feature_names"] == preview_feature_names()
    assert manifest["expanded_parameter_tensors"] == sorted(
        FIRST_OBSERVATION_LAYER_WEIGHTS
    )
    assert manifest["optimizer_state_restored"] is False
    assert manifest["optimizer_state_entries_after_clear"] == 0
    assert manifest["source_num_timesteps"] == 2_465_792
    assert manifest["target_num_timesteps"] == 2_465_792
    assert parity["equivalent_within_tolerance"] is True
    assert parity["maximum_equivalence_error"] <= 1e-7

    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()
    for name, source_tensor in source_state.items():
        target_tensor = target_state[name]
        if name in FIRST_OBSERVATION_LAYER_WEIGHTS:
            np.testing.assert_array_equal(
                target_tensor[:, :122].detach().cpu().numpy(),
                source_tensor.detach().cpu().numpy(),
            )
            np.testing.assert_array_equal(
                target_tensor[:, 122:].detach().cpu().numpy(),
                np.zeros((64, 13), dtype=np.float32),
            )
        else:
            np.testing.assert_array_equal(
                target_tensor.detach().cpu().numpy(),
                source_tensor.detach().cpu().numpy(),
            )


def test_transfer_rejects_a_target_with_the_wrong_appended_dimension() -> None:
    source = make_model(122, seed=401)
    target = make_model(134, seed=402)
    with pytest.raises(ValueError, match="Target observation dimension"):
        transfer_ppo_with_appended_observations(
            source,
            target,
            appended_feature_names=preview_feature_names(),
        )


def test_equivalence_check_rejects_a_changed_observation_prefix() -> None:
    source = make_model(122, seed=501)
    target = make_model(135, seed=502)
    transfer_ppo_with_appended_observations(
        source,
        target,
        appended_feature_names=preview_feature_names(),
    )
    source_observation = np.zeros(122, dtype=np.float32)
    target_observation = np.zeros(135, dtype=np.float32)
    target_observation[0] = 0.25
    with pytest.raises(ValueError, match="prefix differs"):
        verify_ppo_appended_observation_equivalence(
            source,
            target,
            source_observations=source_observation,
            target_observations=target_observation,
            tolerance=1e-7,
        )
