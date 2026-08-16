from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from proxygap.budget_extension import (
    continuation_targets,
    resolve_project_path,
    validate_budget_extension_config,
)


def valid_config() -> dict:
    sources = []
    for seed in [41101, 41102]:
        for weight, condition in [
            (0.5, "reference"),
            (0.21875, "ctrl_0p21875"),
            (0.125, "ctrl_0p125"),
        ]:
            sources.append(
                {
                    "condition_id": condition,
                    "ctrl_cost_weight": weight,
                    "training_seed": seed,
                    "path": f"models/{seed}/{condition}.zip",
                    "sha256": "a" * 64,
                    "expected_num_timesteps": 301056,
                }
            )
    return {
        "environment": "Ant-v5",
        "algorithm": "PPO",
        "device": "cpu",
        "stage_scope": "stage_1_detection_only_no_shaping",
        "formal_launch": "prohibited",
        "shaping_launch": "prohibited",
        "reward": {
            "forward_progress_shaping_weight": 0.0,
            "lateral_drift_shaping_weight": 0.0,
            "effort_shaping_weight": 0.0,
            "orientation_shaping_weight": 0.0,
        },
        "budget_extension": {
            "ctrl_cost_weights": [0.5, 0.21875, 0.125],
            "training_seeds": [41101, 41102],
            "evaluation_seeds": list(range(51101, 51111)),
            "checkpoint_timesteps": [500000, 750000, 1000000],
            "target_timesteps": 1000000,
            "source_target_timesteps": 300000,
            "expected_source_model_timesteps": 301056,
            "normalisation_enabled": False,
            "environment_state_restored": False,
        },
        "ppo": {
            "policy": "MlpPolicy",
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
            "use_sde": False,
            "torch_num_threads": 2,
            "policy_kwargs": {"net_arch": [64, 64], "activation_fn": "Tanh"},
        },
        "source_policies": sources,
    }


def test_locked_budget_extension_config_is_valid() -> None:
    assert validate_budget_extension_config(valid_config()) == []


def test_budget_extension_rejects_shaping_and_normalisation() -> None:
    config = valid_config()
    config["reward"]["lateral_drift_shaping_weight"] = 0.1
    config["budget_extension"]["normalisation_enabled"] = True
    errors = validate_budget_extension_config(config)
    assert "All shaping weights must be zero in stage one" in errors
    assert "normalisation must remain disabled in this budget-only extension" in errors


def test_budget_extension_rejects_duplicate_or_missing_source_cells() -> None:
    config = valid_config()
    config["source_policies"][-1] = deepcopy(config["source_policies"][0])
    errors = validate_budget_extension_config(config)
    assert any("Duplicate source policy cell" in error for error in errors)
    assert any("Source policy cells must equal" in error for error in errors)


def test_continuation_targets_must_extend_source() -> None:
    assert continuation_targets(301056, [500000, 750000, 1000000]) == (
        500000,
        750000,
        1000000,
    )
    try:
        continuation_targets(301056, [300000, 1000000])
    except ValueError as error:
        assert "exceed source timesteps" in str(error)
    else:
        raise AssertionError("A non-extending checkpoint must be rejected")


def test_source_file_verification_detects_hash_mismatch(tmp_path: Path) -> None:
    config = valid_config()
    for source in config["source_policies"]:
        path = tmp_path / source["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
        source["sha256"] = hashlib.sha256(b"model").hexdigest()
    assert validate_budget_extension_config(
        config, project_root=tmp_path, verify_source_files=True
    ) == []
    (tmp_path / config["source_policies"][0]["path"]).write_bytes(b"changed")
    errors = validate_budget_extension_config(
        config, project_root=tmp_path, verify_source_files=True
    )
    assert any("SHA-256 mismatch" in error for error in errors)


def test_source_paths_cannot_escape_revision_project(tmp_path: Path) -> None:
    try:
        resolve_project_path(tmp_path, "../outside.zip")
    except ValueError as error:
        assert "outside the revision project" in str(error)
    else:
        raise AssertionError("An external source path must be rejected")
