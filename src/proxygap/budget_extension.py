"""Audited continuation of existing stage-one PPO development policies.

This module deliberately keeps checkpoint continuation separate from fresh
training. A loaded policy retains its parameters, optimiser state and recorded
timestep count, but the MuJoCo environment and pseudorandom streams restart.
That discontinuity is recorded rather than hidden.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from torch import nn

from .ant_wrapper import make_proxygap_ant_env
from .experiment import evaluate_model


LOCKED_WEIGHTS = (0.5, 0.21875, 0.125)
LOCKED_TRAINING_SEEDS = (41101, 41102)
LOCKED_EVALUATION_SEEDS = tuple(range(51101, 51111))
LOCKED_EXTENSION_CHECKPOINTS = (500_000, 750_000, 1_000_000)
SHAPING_FIELDS = (
    "forward_progress_shaping_weight",
    "lateral_drift_shaping_weight",
    "effort_shaping_weight",
    "orientation_shaping_weight",
)


def condition_id(weight: float) -> str:
    """Return the condition identifier used throughout the project."""
    if math.isclose(float(weight), 0.5, rel_tol=0.0, abs_tol=1e-12):
        return "reference"
    return f"ctrl_{str(float(weight)).replace('.', 'p')}"


def sha256(path: Path) -> str:
    """Hash a file without loading the entire model archive into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(project_root: Path, value: str) -> Path:
    """Resolve a project-relative path and reject paths outside the project."""
    root = project_root.resolve()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Source model is outside the revision project: {candidate}") from error
    return candidate


def continuation_targets(
    source_timesteps: int,
    checkpoint_timesteps: Sequence[int],
) -> tuple[int, ...]:
    """Validate targets that strictly extend a saved policy."""
    targets = tuple(int(value) for value in checkpoint_timesteps)
    if not targets or targets != tuple(sorted(set(targets))):
        raise ValueError("Continuation checkpoints must be unique and increasing")
    if any(target <= int(source_timesteps) for target in targets):
        raise ValueError("Every continuation checkpoint must exceed source timesteps")
    return targets


def validate_budget_extension_config(
    config: Mapping[str, Any],
    *,
    project_root: Path | None = None,
    verify_source_files: bool = False,
) -> list[str]:
    """Return every protocol error without silently repairing the design."""
    errors: list[str] = []
    if config.get("environment") != "Ant-v5":
        errors.append("environment must be Ant-v5")
    if config.get("algorithm") != "PPO":
        errors.append("algorithm must be PPO")
    if config.get("device") != "cpu":
        errors.append("device must remain cpu")
    if config.get("stage_scope") != "stage_1_detection_only_no_shaping":
        errors.append("stage_scope must remain stage-one detection without shaping")
    if config.get("formal_launch") != "prohibited":
        errors.append("formal_launch must remain prohibited during development extension")
    if config.get("shaping_launch") != "prohibited":
        errors.append("shaping_launch must remain prohibited during stage one")

    reward = config.get("reward", {})
    if any(float(reward.get(field, 0.0)) != 0.0 for field in SHAPING_FIELDS):
        errors.append("All shaping weights must be zero in stage one")

    extension = config.get("budget_extension", {})
    weights = tuple(float(value) for value in extension.get("ctrl_cost_weights", []))
    seeds = tuple(int(value) for value in extension.get("training_seeds", []))
    evaluation_seeds = tuple(
        int(value) for value in extension.get("evaluation_seeds", [])
    )
    checkpoints = tuple(
        int(value) for value in extension.get("checkpoint_timesteps", [])
    )
    if weights != LOCKED_WEIGHTS:
        errors.append(f"ctrl_cost_weights must equal {list(LOCKED_WEIGHTS)}")
    if seeds != LOCKED_TRAINING_SEEDS:
        errors.append(f"training_seeds must equal {list(LOCKED_TRAINING_SEEDS)}")
    if evaluation_seeds != LOCKED_EVALUATION_SEEDS:
        errors.append(
            f"evaluation_seeds must equal {list(LOCKED_EVALUATION_SEEDS)}"
        )
    if checkpoints != LOCKED_EXTENSION_CHECKPOINTS:
        errors.append(
            f"checkpoint_timesteps must equal {list(LOCKED_EXTENSION_CHECKPOINTS)}"
        )
    if int(extension.get("target_timesteps", 0)) != 1_000_000:
        errors.append("target_timesteps must equal 1000000")
    if int(extension.get("source_target_timesteps", 0)) != 300_000:
        errors.append("source_target_timesteps must equal 300000")
    if int(extension.get("expected_source_model_timesteps", 0)) != 301_056:
        errors.append("expected_source_model_timesteps must equal 301056")
    if bool(extension.get("normalisation_enabled", True)):
        errors.append("normalisation must remain disabled in this budget-only extension")
    if bool(extension.get("environment_state_restored", True)):
        errors.append("environment_state_restored must be false for saved SB3 models")

    ppo = config.get("ppo", {})
    locked_ppo = {
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
    }
    for key, expected in locked_ppo.items():
        actual = ppo.get(key)
        if isinstance(expected, float):
            try:
                equal = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                equal = False
        else:
            equal = actual == expected
        if not equal:
            errors.append(f"PPO field {key} must remain {expected!r}")
    if ppo.get("policy") != "MlpPolicy":
        errors.append("PPO policy must remain MlpPolicy")
    policy_kwargs = ppo.get("policy_kwargs", {})
    if policy_kwargs.get("net_arch") != [64, 64]:
        errors.append("PPO net_arch must remain [64, 64]")
    if policy_kwargs.get("activation_fn") != "Tanh":
        errors.append("PPO activation_fn must remain Tanh")

    source_policies = list(config.get("source_policies", []))
    expected_cells = {(seed, weight) for seed in seeds for weight in weights}
    actual_cells: set[tuple[int, float]] = set()
    for source in source_policies:
        try:
            cell = (int(source["training_seed"]), float(source["ctrl_cost_weight"]))
        except (KeyError, TypeError, ValueError):
            errors.append("Every source policy requires numeric training_seed and weight")
            continue
        if cell in actual_cells:
            errors.append(f"Duplicate source policy cell: {cell}")
        actual_cells.add(cell)
        if source.get("condition_id") != condition_id(cell[1]):
            errors.append(f"Source condition_id does not match weight for cell {cell}")
        if int(source.get("expected_num_timesteps", -1)) != 301_056:
            errors.append(f"Source expected_num_timesteps is not 301056 for cell {cell}")
        digest = str(source.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            errors.append(f"Source SHA-256 is invalid for cell {cell}")
        if verify_source_files:
            if project_root is None:
                errors.append("project_root is required when source files are verified")
                continue
            try:
                path = resolve_project_path(project_root, str(source["path"]))
            except (KeyError, ValueError) as error:
                errors.append(str(error))
                continue
            if not path.is_file():
                errors.append(f"Source model is missing: {path}")
            elif sha256(path).lower() != digest.lower():
                errors.append(f"Source model SHA-256 mismatch: {path}")
    if actual_cells != expected_cells:
        errors.append(
            f"Source policy cells must equal {sorted(expected_cells)}; got {sorted(actual_cells)}"
        )
    if len(source_policies) != len(expected_cells):
        errors.append(f"Exactly {len(expected_cells)} source policies are required")
    return errors


def audit_loaded_model(
    model: PPO,
    *,
    source: Mapping[str, Any],
    ppo: Mapping[str, Any],
) -> list[str]:
    """Check that a loaded source model matches the frozen PPO contract."""
    errors: list[str] = []
    expected_seed = int(source["training_seed"])
    expected_steps = int(source["expected_num_timesteps"])
    scalar_checks = {
        "num_timesteps": (int(model.num_timesteps), expected_steps),
        "seed": (int(model.seed), expected_seed),
        "n_steps": (int(model.n_steps), int(ppo["n_steps"])),
        "batch_size": (int(model.batch_size), int(ppo["batch_size"])),
        "n_epochs": (int(model.n_epochs), int(ppo["n_epochs"])),
        "normalize_advantage": (
            bool(model.normalize_advantage),
            bool(ppo["normalize_advantage"]),
        ),
        "use_sde": (bool(model.use_sde), bool(ppo["use_sde"])),
    }
    for name, (actual, expected) in scalar_checks.items():
        if actual != expected:
            errors.append(f"Loaded model {name}={actual!r}, expected {expected!r}")
    float_checks = {
        "learning_rate": (float(model.lr_schedule(1.0)), float(ppo["learning_rate"])),
        "gamma": (float(model.gamma), float(ppo["gamma"])),
        "gae_lambda": (float(model.gae_lambda), float(ppo["gae_lambda"])),
        "clip_range": (float(model.clip_range(1.0)), float(ppo["clip_range"])),
        "ent_coef": (float(model.ent_coef), float(ppo["ent_coef"])),
        "vf_coef": (float(model.vf_coef), float(ppo["vf_coef"])),
        "max_grad_norm": (float(model.max_grad_norm), float(ppo["max_grad_norm"])),
    }
    for name, (actual, expected) in float_checks.items():
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            errors.append(f"Loaded model {name}={actual}, expected {expected}")
    if tuple(model.observation_space.shape or ()) != (105,):
        errors.append("Loaded model observation shape is not (105,)")
    if tuple(model.action_space.shape or ()) != (8,):
        errors.append("Loaded model action shape is not (8,)")
    if model.device.type != "cpu":
        errors.append("Loaded model is not on CPU")
    if type(model.policy.optimizer).__name__ != "Adam":
        errors.append("Loaded optimiser is not Adam")
    optimiser_defaults = model.policy.optimizer.defaults
    if float(optimiser_defaults.get("weight_decay", -1.0)) != 0.0:
        errors.append("Loaded Adam weight_decay is not zero")
    if not math.isclose(
        float(optimiser_defaults.get("eps", 0.0)), 1e-5, rel_tol=0.0, abs_tol=1e-12
    ):
        errors.append("Loaded Adam epsilon is not 1e-5")
    if tuple(optimiser_defaults.get("betas", ())) != (0.9, 0.999):
        errors.append("Loaded Adam betas are not (0.9, 0.999)")

    expected_layers = tuple(int(value) for value in ppo["policy_kwargs"]["net_arch"])
    policy_layers = tuple(
        module.out_features
        for module in model.policy.mlp_extractor.policy_net
        if isinstance(module, nn.Linear)
    )
    value_layers = tuple(
        module.out_features
        for module in model.policy.mlp_extractor.value_net
        if isinstance(module, nn.Linear)
    )
    if policy_layers != expected_layers or value_layers != expected_layers:
        errors.append(
            f"Loaded network layers are pi={policy_layers}, vf={value_layers}; "
            f"expected {expected_layers}"
        )
    activations = [
        module
        for network in (
            model.policy.mlp_extractor.policy_net,
            model.policy.mlp_extractor.value_net,
        )
        for module in network
        if not isinstance(module, nn.Linear)
    ]
    if len(activations) != 4 or not all(isinstance(module, nn.Tanh) for module in activations):
        errors.append("Loaded network does not contain four Tanh hidden activations")
    return errors


def continue_policy(
    *,
    project_root: Path,
    output_root: Path,
    source: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Continue one immutable 300k policy and evaluate locked checkpoints."""
    extension = config["budget_extension"]
    ppo = config["ppo"]
    reward = config["reward"]
    training_seed = int(source["training_seed"])
    weight = float(source["ctrl_cost_weight"])
    cid = str(source["condition_id"])
    source_path = resolve_project_path(project_root, str(source["path"]))
    source_hash_before = sha256(source_path)
    expected_hash = str(source["sha256"]).lower()
    if source_hash_before.lower() != expected_hash:
        raise ValueError(f"Source model hash mismatch before load: {source_path}")

    torch.set_num_threads(int(ppo["torch_num_threads"]))
    raw_env = make_proxygap_ant_env(
        ctrl_cost_weight=weight,
        condition_id=cid,
        seed=training_seed,
        forward_progress_shaping_weight=float(
            reward["forward_progress_shaping_weight"]
        ),
        lateral_drift_shaping_weight=float(reward["lateral_drift_shaping_weight"]),
        effort_shaping_weight=float(reward["effort_shaping_weight"]),
        orientation_shaping_weight=float(reward["orientation_shaping_weight"]),
    )
    monitor_path = output_root / "logs" / "training_extension.monitor.csv"
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    env = Monitor(raw_env, filename=str(monitor_path))
    try:
        model = PPO.load(source_path, env=env, device="cpu")
        audit_errors = audit_loaded_model(model, source=source, ppo=ppo)
        if audit_errors:
            raise ValueError(f"Loaded source model failed audit: {audit_errors}")
        source_steps = int(model.num_timesteps)
        targets = continuation_targets(
            source_steps,
            extension["checkpoint_timesteps"],
        )

        # Saved SB3 archives do not contain the live MuJoCo state or complete RNG
        # stream. Re-seeding makes the declared continuation reproducible while
        # preserving the fact that it is not bitwise-equivalent to an uninterrupted run.
        model.set_random_seed(training_seed)
        runtime_rows: list[dict[str, Any]] = []
        evaluation_rows: list[dict[str, Any]] = []
        model_dir = output_root / "models" / cid
        evaluation_seeds = tuple(int(value) for value in extension["evaluation_seeds"])
        if evaluation_seeds != tuple(
            range(evaluation_seeds[0], evaluation_seeds[0] + len(evaluation_seeds))
        ):
            raise ValueError("evaluate_model requires consecutive evaluation seeds")

        final_target = int(extension["target_timesteps"])
        for target in targets:
            start_steps = int(model.num_timesteps)
            chunk = target - start_steps
            started = time.perf_counter()
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            train_elapsed = time.perf_counter() - started
            actual_steps = int(model.num_timesteps)

            model_path = model_dir / f"checkpoint_{target:07d}.zip"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(model_path)
            model_hash = sha256(model_path)
            rows, evaluation_elapsed = evaluate_model(
                model,
                condition_id=cid,
                ctrl_cost_weight=weight,
                checkpoint_fraction=target / final_target,
                seed=evaluation_seeds[0],
                episodes=len(evaluation_seeds),
                target_timesteps=target,
                actual_model_timesteps=actual_steps,
                training_seed=training_seed,
                max_episode_steps=int(extension["evaluation_max_episode_steps"]),
            )
            evaluation_rows.extend(rows)
            runtime_rows.append(
                {
                    "condition_id": cid,
                    "ctrl_cost_weight": weight,
                    "training_seed": training_seed,
                    "source_model": str(source_path),
                    "source_model_sha256": source_hash_before,
                    "source_actual_timesteps": source_steps,
                    "environment_state_restored": False,
                    "continuation_rng_seed": training_seed,
                    "checkpoint_fraction": target / final_target,
                    "target_timesteps": target,
                    "start_actual_timesteps": start_steps,
                    "actual_model_timesteps": actual_steps,
                    "chunk_timesteps_requested": chunk,
                    "train_elapsed_sec": round(train_elapsed, 3),
                    "train_steps_per_sec": round(
                        (actual_steps - start_steps) / max(train_elapsed, 1e-8), 2
                    ),
                    "evaluation_seeds": ";".join(str(seed) for seed in evaluation_seeds),
                    "eval_episodes": len(evaluation_seeds),
                    "eval_elapsed_sec": round(evaluation_elapsed, 3),
                    "model_path": str(model_path),
                    "model_sha256": model_hash,
                    "torch_num_threads": torch.get_num_threads(),
                }
            )

        source_hash_after = sha256(source_path)
        if source_hash_after != source_hash_before:
            raise RuntimeError(f"Source model changed during continuation: {source_path}")
        source_audit = {
            "condition_id": cid,
            "ctrl_cost_weight": weight,
            "training_seed": training_seed,
            "source_model": str(source_path),
            "source_model_sha256_expected": expected_hash,
            "source_model_sha256_before": source_hash_before,
            "source_model_sha256_after": source_hash_after,
            "source_hash_unchanged": True,
            "source_actual_timesteps": source_steps,
            "loaded_model_audit": "pass",
            "environment_state_restored": False,
            "rng_stream_restored": False,
            "continuation_rng_seed": training_seed,
        }
        return runtime_rows, evaluation_rows, source_audit
    finally:
        env.close()
