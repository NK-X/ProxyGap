from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_fixed_standard_pair0_adaptation_l2b_extension.py"
CONFIG = (
    ROOT
    / "configs"
    / "fixed_standard_pair0_adaptation_l2b_extension_v3_20260819.json"
)
SPEC = importlib.util.spec_from_file_location("pair0_adaptation_l2b", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load_validated() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    MODULE.validate_config(config)
    return config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_nested(mapping: dict, path: tuple[str, ...], value: object) -> None:
    cursor = mapping
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def iter_leaf_paths(value: object, prefix: tuple[object, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_leaf_paths(child, prefix + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_leaf_paths(child, prefix + (index,))
    else:
        yield prefix, value


def set_nested(container: object, path: tuple[object, ...], value: object) -> None:
    cursor = container
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def changed_leaf(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, str):
        return value + "__mutated"
    if value is None:
        return "__mutated_from_none"
    raise TypeError(f"Unhandled frozen leaf type: {type(value).__name__}")


def safe_aggregate() -> dict:
    return {
        "nonfinite_episode_count": 0,
        "energy_components_finite": True,
        "fall_count": 0,
        "torso_ground_episode_count": 0,
        "sustained_nonfoot_contact_episode_count": 0,
        "force_qualified_slip_evaluable": True,
        "corrected_sustained_slip_per_force_qualified_supported_fraction": 0.0,
        "corrected_slip_events_per_100_force_qualified_supported_substeps": 0.0,
        "pooled_full_interval_zero_foot_fraction": 0.0,
        "qualified_slip_per_force_qualified_supported_fraction": 0.0,
        "mean_best_progress_m": 0.0,
    }


def gate_aggregate(
    *,
    zero_foot: float,
    support: float,
    progress: float,
    uphill: float,
    downhill: float,
) -> dict:
    return {
        "nonfinite_episode_count": 0,
        "energy_components_finite": True,
        "force_qualified_slip_evaluable": True,
        "pooled_full_interval_zero_foot_fraction": zero_foot,
        "mean_support_count": support,
        "mean_best_progress_m": progress,
        "per_scene": {
            "flat": {"mean_best_progress_m": progress},
            "uphill_8deg": {"mean_best_progress_m": uphill},
            "downhill_8deg": {"mean_best_progress_m": downhill},
            "bowl_exit": {"mean_best_progress_m": progress},
        },
        "success_count": 1,
        "fall_count": 0,
        "torso_ground_episode_count": 0,
        "sustained_nonfoot_contact_episode_count": 0,
        "corrected_sustained_slip_per_force_qualified_supported_fraction": 0.0,
        "corrected_slip_events_per_100_force_qualified_supported_substeps": 0.0,
    }


def passing_heldout() -> dict[str, dict]:
    return {
        "DEFAULT_CONTINUE": gate_aggregate(
            zero_foot=0.20,
            support=1.00,
            progress=10.0,
            uphill=10.0,
            downhill=10.0,
        ),
        "PAIR0_ADAPT": gate_aggregate(
            zero_foot=0.04,
            support=1.30,
            progress=9.5,
            uphill=9.0,
            downhill=9.0,
        ),
    }


def passing_continuity() -> dict[str, dict]:
    return {
        "DEFAULT_CONTINUE": gate_aggregate(
            zero_foot=0.20,
            support=1.00,
            progress=8.0,
            uphill=7.0,
            downhill=9.0,
        ),
        "PAIR0_ADAPT": gate_aggregate(
            zero_foot=0.04,
            support=1.30,
            progress=8.0,
            uphill=7.0,
            downhill=9.0,
        ),
    }


def test_config_freezes_both_l2_checkpoints_and_hashes() -> None:
    config = load_validated()
    assert config["schema_version"] == (
        "proxygap-fixed-standard-pair0-adaptation-l2b-extension-v3"
    )
    assert config["config_id"] == (
        "fixed_standard_pair0_adaptation_l2b_extension_v3_20260819"
    )
    assert config["status"] == (
        "frozen_l2b_v3_gate_only_audit_repair_once_only_matched_budget_extension"
    )
    sources = config["source"]
    assert (
        sources["l2_manifest_sha256"]
        == "89b7075e737b21e7ecda5c54cae12b133f56190efa715180da330004e2578568"
    )
    assert file_sha256(ROOT / sources["l2_manifest"]) == sources["l2_manifest_sha256"]
    assert sources["checkpoint_timesteps"] == 2_662_400
    assert sources["conditions"] == {
        "DEFAULT_CONTINUE": {
            "checkpoint": "artifacts/dev/fixed_standard_pair0_adaptation_l2_pilot_v1_20260819/attempt_0/default_continue/models/checkpoint_2662400.zip",
            "checkpoint_sha256": "6549c279ca5795636d3b1d6f61c36782f4f843a32107276adf0630c39871cb6f",
            "explicit_pair_count": 0,
        },
        "PAIR0_ADAPT": {
            "checkpoint": "artifacts/dev/fixed_standard_pair0_adaptation_l2_pilot_v1_20260819/attempt_0/pair0_adapt/models/checkpoint_2662400.zip",
            "checkpoint_sha256": "9eb1268352aeb90024f681b70ca3b42cb036f6e5ea882e56dbb85262bd8c500e",
            "explicit_pair_count": 4,
        },
    }
    for source in sources["conditions"].values():
        assert file_sha256(ROOT / source["checkpoint"]) == source["checkpoint_sha256"]

    changed = copy.deepcopy(config)
    changed["source"]["conditions"]["PAIR0_ADAPT"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        MODULE.validate_config(changed)


def test_every_leaf_in_every_frozen_v3_section_fails_closed_when_mutated() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected_leaf_counts = {
        "source": 25,
        "authorisation": 20,
        "conditions": 4,
        "contact_contract": 28,
        "training": 29,
        "evaluation": 35,
        "prospective_final_gate": 26,
        "checkpoint_early_stopping": 12,
        "ppo": 17,
        "energy_boundary": 9,
        "invariants": 8,
        "smoke": 5,
        "execution": 11,
        "claim_boundary": 1,
    }
    observed_leaf_counts: dict[str, int] = {}
    for section, expected_count in expected_leaf_counts.items():
        leaves = list(iter_leaf_paths(config[section]))
        observed_leaf_counts[section] = len(leaves)
        assert len(leaves) == expected_count
        for path, original in leaves:
            changed = copy.deepcopy(config)
            if path:
                set_nested(changed[section], path, changed_leaf(original))
            else:
                changed[section] = changed_leaf(original)
            with pytest.raises(ValueError, match=f"Frozen V3 section changed: {section}"):
                MODULE.validate_config(changed)
    assert observed_leaf_counts == expected_leaf_counts
    assert sum(observed_leaf_counts.values()) == 230

    changed = copy.deepcopy(config)
    changed["unexpected_top_level_field"] = True
    with pytest.raises(ValueError, match="top-level keys"):
        MODULE.validate_config(changed)

    runtime_leaves = list(iter_leaf_paths(config["runtime_dependency_contract"]))
    assert len(runtime_leaves) == 24
    for path, original in runtime_leaves:
        changed = copy.deepcopy(config)
        set_nested(
            changed["runtime_dependency_contract"], path, changed_leaf(original)
        )
        with pytest.raises((ValueError, FileNotFoundError)):
            MODULE.validate_config(changed)


def test_v3_once_only_safety_energy_and_execution_contract_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_validated()
    expected_early_stopping = {
        "performance_futility_stopping_enabled": False,
        "intermediate_promotion_enabled": False,
        "fail_on_any_nonfinite_or_contract_mismatch": True,
        "maximum_fall_count_each_condition": 0,
        "maximum_torso_ground_episode_count_each_condition": 0,
        "maximum_sustained_nonfoot_contact_episode_count_each_condition": 0,
        "nonfoot_contact_minimum_sustained_seconds": 0.2,
        "maximum_pair0_pooled_full_interval_zero_foot_fraction": 0.08,
        "maximum_corrected_sustained_slip_per_force_qualified_supported_fraction_each_condition": 0.02,
        "maximum_corrected_slip_events_per_100_force_qualified_supported_substeps_each_condition": 0.2,
        "zero_force_qualified_supported_denominator_is_failure": True,
        "qualified_transient_warning_only": True,
    }
    assert config["checkpoint_early_stopping"] == expected_early_stopping
    assert config["authorisation"]["maximum_l2b_protocol_retry_index"] == 0
    assert config["authorisation"]["maximum_contact_budget_extensions_after_l2"] == 1
    assert config["execution"]["maximum_protocol_retry_index"] == 0
    assert config["execution"]["fail_if_attempt_root_exists"] is True
    assert config["execution"]["subprocess_start_method"] == "spawn"
    assert config["execution"]["fixed_map_evaluation"] is False
    assert config["execution"]["video_rendering"] is False
    assert config["execution"]["promotion"] is False
    assert config["execution"]["hard_stop_after_this_extension"] is True
    assert config["energy_boundary"]["formula_unchanged"] is True
    assert config["energy_boundary"]["reward_weight"] == 0.0
    assert config["energy_boundary"]["nonfinite_energy_component_is_run_failure"] is True
    assert config["energy_boundary"]["electrical_battery_energy_claim_permitted"] is False

    monkeypatch.setattr(MODULE, "_verify_checkpoint", lambda *args: None)
    mutations = [
        (("authorisation", "maximum_l2b_protocol_retry_index"), 1),
        (("authorisation", "maximum_contact_budget_extensions_after_l2"), 2),
        (("execution", "maximum_protocol_retry_index"), 1),
        (("execution", "fail_if_attempt_root_exists"), False),
        (("execution", "subprocess_start_method"), "fork"),
        (("execution", "fixed_map_evaluation"), True),
        (("execution", "video_rendering"), True),
        (("execution", "promotion"), True),
        (("execution", "hard_stop_after_this_extension"), False),
        (("energy_boundary", "formula_unchanged"), False),
        (("energy_boundary", "reward_weight"), 0.01),
        (("energy_boundary", "nonfinite_energy_component_is_run_failure"), False),
        (("energy_boundary", "electrical_battery_energy_claim_permitted"), True),
        (("checkpoint_early_stopping", "performance_futility_stopping_enabled"), True),
        (("checkpoint_early_stopping", "intermediate_promotion_enabled"), True),
        (("checkpoint_early_stopping", "fail_on_any_nonfinite_or_contract_mismatch"), False),
        (("checkpoint_early_stopping", "maximum_fall_count_each_condition"), 1),
        (("checkpoint_early_stopping", "maximum_torso_ground_episode_count_each_condition"), 1),
        (("checkpoint_early_stopping", "maximum_sustained_nonfoot_contact_episode_count_each_condition"), 1),
        (("checkpoint_early_stopping", "nonfoot_contact_minimum_sustained_seconds"), 0.1),
        (("checkpoint_early_stopping", "maximum_pair0_pooled_full_interval_zero_foot_fraction"), 0.09),
        (("checkpoint_early_stopping", "maximum_corrected_sustained_slip_per_force_qualified_supported_fraction_each_condition"), 0.03),
        (("checkpoint_early_stopping", "maximum_corrected_slip_events_per_100_force_qualified_supported_substeps_each_condition"), 0.3),
        (("checkpoint_early_stopping", "zero_force_qualified_supported_denominator_is_failure"), False),
        (("checkpoint_early_stopping", "qualified_transient_warning_only"), False),
    ]
    for path, changed_value in mutations:
        changed = copy.deepcopy(config)
        replace_nested(changed, path, changed_value)
        with pytest.raises(ValueError):
            MODULE.validate_config(changed)


def test_runtime_dependency_closure_has_exact_paths_and_hashes() -> None:
    config = load_validated()
    expected_paths = (
        "scripts/run_fixed_standard_pair0_adaptation_l2b_extension.py",
        "scripts/run_fixed_standard_pair0_adaptation_l2_pilot.py",
        "scripts/evaluate_fixed_standard_distal_margin0_paired.py",
        "scripts/evaluate_local_preview_final_paired_direct_goal.py",
        "scripts/run_fixed_goal_support_priority_pilot.py",
        "scripts/run_fixed_standard_support_curriculum.py",
        "scripts/run_fixed_goal_terrain_training.py",
        "scripts/run_curved_gait_training.py",
        "src/proxygap/__init__.py",
        "src/proxygap/ant_wrapper.py",
        "src/proxygap/curved_gait.py",
        "src/proxygap/fixed_goal_terrain.py",
        "src/proxygap/metrics.py",
        "src/proxygap/planar_transition.py",
        "src/proxygap/experiment.py",
        "src/proxygap/divergence.py",
        "src/proxygap/protocol.py",
        "src/proxygap/selection.py",
        "src/proxygap/two_experiment_protocol.py",
    )
    dependency_map = MODULE.validate_runtime_dependency_map(config)
    assert dependency_map == config["runtime_dependency_contract"][
        "exact_relative_path_sha256"
    ]
    assert tuple(dependency_map) == expected_paths
    assert (
        dependency_map["scripts/run_fixed_standard_pair0_adaptation_l2b_extension.py"]
        == "6dfd283aaa7661e3806e9ca26f874a6088ebf0a39985d3ef3bd0c7edb6e493aa"
    )
    assert (
        dependency_map["scripts/run_fixed_standard_pair0_adaptation_l2_pilot.py"]
        == "1c426d7a78cd73bd7e9448e2ecd7f6ab5688871894281d607ca61c61fdd7e7dd"
    )
    assert config["source"]["l2_runtime_dependency"] == (
        "scripts/run_fixed_standard_pair0_adaptation_l2_pilot.py"
    )
    assert config["source"]["l2_runtime_dependency_sha256"] == dependency_map[
        config["source"]["l2_runtime_dependency"]
    ]
    for flag in (
        "verify_live_before_each_worker",
        "verify_snapshot_before_each_worker",
        "verify_live_and_snapshot_after_each_worker",
        "copy_preserving_relative_paths",
    ):
        assert config["runtime_dependency_contract"][flag] is True
        changed = copy.deepcopy(config)
        changed["runtime_dependency_contract"][flag] = False
        with pytest.raises(ValueError):
            MODULE.validate_runtime_dependency_map(changed)
    for relative_path, expected_sha256 in dependency_map.items():
        assert file_sha256(ROOT / relative_path) == expected_sha256

    changed = copy.deepcopy(config)
    changed["runtime_dependency_contract"]["exact_relative_path_sha256"][
        "scripts/run_fixed_standard_pair0_adaptation_l2_pilot.py"
    ] = "0" * 64
    with pytest.raises(ValueError):
        MODULE.validate_runtime_dependency_map(changed)

    changed = copy.deepcopy(config)
    del changed["runtime_dependency_contract"]["exact_relative_path_sha256"][
        "src/proxygap/two_experiment_protocol.py"
    ]
    with pytest.raises(ValueError):
        MODULE.validate_runtime_dependency_map(changed)

    changed = copy.deepcopy(config)
    changed["runtime_dependency_contract"]["exact_relative_path_sha256"][
        "README.md"
    ] = file_sha256(ROOT / "README.md")
    with pytest.raises(ValueError):
        MODULE.validate_runtime_dependency_map(changed)


def test_runtime_dependency_snapshot_copies_live_l2_runner_with_exact_hash(
    tmp_path: Path,
) -> None:
    config = load_validated()
    snapshot_root, live_map = MODULE.snapshot_runtime_dependencies(config, tmp_path)
    assert snapshot_root == tmp_path / "runtime_snapshot"
    assert MODULE.validate_runtime_snapshot(config, snapshot_root) == live_map
    l2_relative_path = "scripts/run_fixed_standard_pair0_adaptation_l2_pilot.py"
    frozen_l2_runner = snapshot_root / l2_relative_path
    assert frozen_l2_runner.is_file()
    assert file_sha256(frozen_l2_runner) == (
        "1c426d7a78cd73bd7e9448e2ecd7f6ab5688871894281d607ca61c61fdd7e7dd"
    )
    unexpected = snapshot_root / "scripts" / "unexpected_runtime_file.py"
    unexpected.write_text("# unexpected\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="path membership changed"):
        MODULE.validate_runtime_snapshot(config, snapshot_root)


def test_smoke_manifest_records_parent_worker_and_snapshot_dependency_maps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol, reward = MODULE.validate_config(config)
    expected_map = MODULE.validate_runtime_dependency_map(config)
    process_ids = {"DEFAULT_CONTINUE": 71001, "PAIR0_ADAPT": 71002}

    monkeypatch.setattr(
        MODULE.l2,
        "prepare_condition_scenes",
        lambda *args, **kwargs: ({}, {}),
    )
    monkeypatch.setattr(MODULE.l2, "_git_record", lambda: {"test_stub": True})

    def fake_condition_subprocess(*args, **kwargs):
        condition_id = kwargs.get("condition_id")
        if condition_id is None:
            condition_id = next(
                value
                for value in args
                if isinstance(value, str) and value in process_ids
            )
        return {
            "condition_id": condition_id,
            "process_id": process_ids[condition_id],
            "early_stopped": False,
            "runtime_dependency_verification": {
                "live_before_env_model": expected_map,
                "snapshot_before_env_model": expected_map,
                "live_after_training": expected_map,
                "snapshot_after_training": expected_map,
            },
        }

    monkeypatch.setattr(
        MODULE,
        "_run_condition_subprocess",
        fake_condition_subprocess,
    )
    output_root = tmp_path / "attempt_0"
    MODULE.run_extension(
        CONFIG,
        config,
        protocol,
        reward,
        output_root,
        smoke=True,
        attempt=0,
    )
    manifest = json.loads(
        (output_root / "manifest.json").read_text(encoding="utf-8")
    )
    closure = manifest["runtime_dependency_closure"]
    assert Path(closure["snapshot_root"]).resolve() == (
        output_root / "runtime_snapshot"
    ).resolve()
    assert closure["expected_relative_path_sha256"] == expected_map
    assert closure["parent_live_before_workers"] == expected_map
    assert closure["parent_snapshot_before_workers"] == expected_map
    assert closure["parent_live_after_workers"] == expected_map
    assert closure["parent_snapshot_after_workers"] == expected_map
    for condition_id in ("DEFAULT_CONTINUE", "PAIR0_ADAPT"):
        assert closure["per_condition_parent_checks"][condition_id] == {
            "live_before_worker": expected_map,
            "snapshot_before_worker": expected_map,
            "live_after_worker": expected_map,
            "snapshot_after_worker": expected_map,
        }
        assert manifest["condition_training"][condition_id][
            "runtime_dependency_verification"
        ] == {
            "live_before_env_model": expected_map,
            "snapshot_before_env_model": expected_map,
            "live_after_training": expected_map,
            "snapshot_after_training": expected_map,
        }
    frozen_l2_runner = (
        output_root
        / "runtime_snapshot"
        / "scripts"
        / "run_fixed_standard_pair0_adaptation_l2_pilot.py"
    )
    assert file_sha256(frozen_l2_runner) == (
        "1c426d7a78cd73bd7e9448e2ecd7f6ab5688871894281d607ca61c61fdd7e7dd"
    )


def test_config_freezes_sequential_worker_seeds_budget_and_checkpoints() -> None:
    config = load_validated()
    training = config["training"]
    assert training["master_seed"] == 62806
    assert training["scene_order"] == [
        "flat",
        "uphill_8deg",
        "downhill_8deg",
        "bowl_exit",
    ]
    assert training["worker_effective_seeds_by_scene"] == {
        "flat": 62806,
        "uphill_8deg": 62807,
        "downhill_8deg": 62808,
        "bowl_exit": 62809,
    }
    assert training["parallel_environments"] == 4
    assert training["same_master_seed_for_both_conditions"] is True
    assert training["same_worker_effective_seeds_for_both_conditions"] is True
    assert training["condition_run_order"] == ["DEFAULT_CONTINUE", "PAIR0_ADAPT"]
    assert training["independent_clean_process_required_for_each_condition"] is True
    assert training["additional_timesteps_per_condition"] == 65_536
    assert training["checkpoint_interval_timesteps"] == 16_384
    assert training["checkpoint_additional_timesteps"] == [
        16_384,
        32_768,
        49_152,
        65_536,
    ]
    assert training["checkpoint_absolute_timesteps"] == [
        2_678_784,
        2_695_168,
        2_711_552,
        2_727_936,
    ]
    assert config["prospective_final_gate"][
        "only_absolute_final_checkpoint_may_be_evaluated_for_promotion_decision"
    ] == 2_727_936

    changed = copy.deepcopy(config)
    changed["training"]["worker_effective_seeds_by_scene"]["bowl_exit"] = 62810
    with pytest.raises(ValueError):
        MODULE.validate_config(changed)


def test_checkpoint_stopping_is_safety_only_not_performance_futility() -> None:
    config = load_validated()
    assert config["checkpoint_early_stopping"][
        "performance_futility_stopping_enabled"
    ] is False
    assert config["checkpoint_early_stopping"]["intermediate_promotion_enabled"] is False

    poor_but_safe = safe_aggregate()
    poor_but_safe["mean_best_progress_m"] = -100.0
    decision = MODULE.checkpoint_stop_decision(
        config,
        poor_but_safe,
        condition_id="PAIR0_ADAPT",
        checkpoint_additional_timesteps=32_768,
    )
    assert decision["early_stop_triggered"] is False
    assert decision["performance_futility_checked"] is False

    unsafe = safe_aggregate()
    unsafe["fall_count"] = 1
    decision = MODULE.checkpoint_stop_decision(
        config,
        unsafe,
        condition_id="DEFAULT_CONTINUE",
        checkpoint_additional_timesteps=16_384,
    )
    assert decision["catastrophe_checks"]["fall"] is True
    assert decision["early_stop_triggered"] is True


def test_force_qualified_zero_denominator_fails_closed() -> None:
    config = load_validated()
    aggregate = safe_aggregate()
    aggregate["force_qualified_slip_evaluable"] = False
    aggregate[
        "corrected_sustained_slip_per_force_qualified_supported_fraction"
    ] = None
    aggregate[
        "corrected_slip_events_per_100_force_qualified_supported_substeps"
    ] = None
    decision = MODULE.checkpoint_stop_decision(
        config,
        aggregate,
        condition_id="PAIR0_ADAPT",
        checkpoint_additional_timesteps=16_384,
    )
    assert decision["catastrophe_checks"]["force_qualified_slip_non_evaluable"] is True
    assert decision["early_stop_triggered"] is True

    for condition_id in ("DEFAULT_CONTINUE", "PAIR0_ADAPT"):
        for value_mode in ("false", "missing"):
            continuity = passing_continuity()
            if value_mode == "false":
                continuity[condition_id]["force_qualified_slip_evaluable"] = False
            else:
                del continuity[condition_id]["force_qualified_slip_evaluable"]
            gate = MODULE.final_gate(config, passing_heldout(), continuity)
            assert gate["evaluable"] is False
            assert gate["passed"] is False
            assert gate["continuity"]["reason"] == "Zero force-qualified denominator"


def test_final_gate_requires_both_heldout_and_continuity() -> None:
    config = load_validated()
    passed = MODULE.final_gate(config, passing_heldout(), passing_continuity())
    assert passed["evaluable"] is True
    assert passed["heldout"]["passed"] is True
    assert passed["continuity"]["passed"] is True
    assert passed["passed"] is True

    continuity_failure = passing_continuity()
    continuity_failure["PAIR0_ADAPT"]["mean_best_progress_m"] = 7.0
    result = MODULE.final_gate(config, passing_heldout(), continuity_failure)
    assert result["heldout"]["passed"] is True
    assert result["continuity"]["passed"] is False
    assert result["passed"] is False

    heldout_failure = passing_heldout()
    heldout_failure["PAIR0_ADAPT"]["mean_support_count"] = 1.10
    result = MODULE.final_gate(config, heldout_failure, passing_continuity())
    assert result["heldout"]["passed"] is False
    assert result["continuity"]["passed"] is True
    assert result["passed"] is False

    missing = MODULE.final_gate(config, passing_heldout(), {})
    assert missing["evaluable"] is False
    assert missing["passed"] is False


@pytest.mark.parametrize("split", ["heldout", "continuity"])
@pytest.mark.parametrize("condition_id", ["DEFAULT_CONTINUE", "PAIR0_ADAPT"])
def test_final_gate_rejects_nonfinite_condition(
    split: str, condition_id: str
) -> None:
    config = load_validated()
    heldout = passing_heldout()
    continuity = passing_continuity()
    target = heldout if split == "heldout" else continuity
    target[condition_id]["nonfinite_episode_count"] = 1
    gate = MODULE.final_gate(config, heldout, continuity)
    assert gate["evaluable"] is False
    assert gate["passed"] is False
    assert gate[split]["evaluable"] is False
    assert gate[split]["failed_conditions"] == [condition_id]


@pytest.mark.parametrize("split", ["heldout", "continuity"])
@pytest.mark.parametrize("condition_id", ["DEFAULT_CONTINUE", "PAIR0_ADAPT"])
def test_final_gate_rejects_nonfinite_energy_component(
    split: str, condition_id: str
) -> None:
    config = load_validated()
    heldout = passing_heldout()
    continuity = passing_continuity()
    target = heldout if split == "heldout" else continuity
    target[condition_id]["energy_components_finite"] = False
    gate = MODULE.final_gate(config, heldout, continuity)
    assert gate["evaluable"] is False
    assert gate["passed"] is False
    assert gate[split]["evaluable"] is False
    assert gate[split]["failed_conditions"] == [condition_id]


def test_energy_and_no_fixed_map_video_or_promotion_are_frozen() -> None:
    config = load_validated()
    assert config["energy_boundary"] == {
        "status": "measurement_only_not_reward_or_gate",
        "formula_unchanged": True,
        "reward_weight": 0.0,
        "raw_components_required": [
            "cumulative_squared_action",
            "actuator_abs_torque_time_integral_total_n_m_s",
            "actuator_positive_mechanical_work_total_j",
            "actuator_abs_mechanical_work_total_j",
        ],
        "nonfinite_energy_component_is_run_failure": True,
        "electrical_battery_energy_claim_permitted": False,
    }
    assert config["invariants"]["energy_formula_unchanged"] is True
    assert config["execution"]["fixed_map_evaluation"] is False
    assert config["execution"]["video_rendering"] is False
    assert config["execution"]["promotion"] is False
    assert config["execution"]["hard_stop_after_this_extension"] is True
    assert (
        config["execution"]["next_intervention_if_final_gate_fails"]
        == "retain_the_existing_13d_local_terrain_preview_and_redesign_or_strengthen_terrain_feature_utilisation_and_the_terrain_normal_downhill_controller_in_a_separately_predeclared_architecture_experiment_with_reward_held_fixed_initially"
    )

    changed = copy.deepcopy(config)
    changed["energy_boundary"]["reward_weight"] = 0.01
    with pytest.raises(ValueError):
        MODULE.validate_config(changed)
    changed = copy.deepcopy(config)
    changed["execution"]["promotion"] = True
    with pytest.raises(ValueError):
        MODULE.validate_config(changed)


def test_existing_attempt_root_is_never_overwritten(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    protocol, reward = MODULE.validate_config(config)
    with pytest.raises(FileExistsError, match="Refusing to overwrite attempt root"):
        MODULE.run_extension(
            CONFIG,
            config,
            protocol,
            reward,
            tmp_path,
            smoke=True,
            attempt=0,
        )


def test_v3_attempt_zero_and_canonical_formal_root_cannot_be_bypassed(
    tmp_path: Path,
) -> None:
    config = load_validated()
    MODULE.validate_parent_config_path(CONFIG)
    with pytest.raises(ValueError, match="canonical configuration path"):
        MODULE.validate_parent_config_path(tmp_path / CONFIG.name)
    assert config["execution"]["smoke_output_root"] == (
        "artifacts/smoke/pair0_l2b_v3_20260819"
    )
    assert config["execution"]["development_output_root"] == (
        "artifacts/dev/pair0_l2b_v3_20260819"
    )
    canonical_development_base = ROOT / config["execution"][
        "development_output_root"
    ]
    MODULE.validate_attempt_semantics(
        config,
        attempt=0,
        base=canonical_development_base,
        smoke=False,
        custom_output_root_used=False,
    )
    with pytest.raises(ValueError, match="only canonical attempt_0"):
        MODULE.validate_attempt_semantics(
            config,
            attempt=1,
            base=canonical_development_base,
            smoke=False,
            custom_output_root_used=False,
        )
    with pytest.raises(ValueError, match="custom output roots are forbidden"):
        MODULE.validate_attempt_semantics(
            config,
            attempt=0,
            base=tmp_path,
            smoke=False,
            custom_output_root_used=True,
        )
    with pytest.raises(ValueError, match="canonical development root"):
        MODULE.validate_attempt_semantics(
            config,
            attempt=0,
            base=tmp_path,
            smoke=False,
            custom_output_root_used=False,
        )


def test_smoke_early_stop_fails_closed_instead_of_reporting_passed() -> None:
    config = load_validated()
    records = {
        "DEFAULT_CONTINUE": {"early_stopped": False},
        "PAIR0_ADAPT": {"early_stopped": True},
    }
    with pytest.raises(RuntimeError, match="Engineering smoke failed closed"):
        MODULE.gate_from_condition_records(config, records, smoke=True)


def test_formal_early_stop_is_complete_but_scientifically_non_evaluable() -> None:
    config = load_validated()
    records = {
        "DEFAULT_CONTINUE": {
            "early_stopped": False,
            "final_heldout": None,
            "final_continuity": None,
        },
        "PAIR0_ADAPT": {
            "early_stopped": True,
            "final_heldout": None,
            "final_continuity": None,
        },
    }
    status, gate = MODULE.gate_from_condition_records(config, records, smoke=False)
    assert status == "l2b_once_only_extension_complete"
    assert gate["evaluable"] is False
    assert gate["passed"] is False
    assert (
        gate["decision"]
        == "l2b_early_stop_non_evaluable_stop_contact_budget_extension"
    )
