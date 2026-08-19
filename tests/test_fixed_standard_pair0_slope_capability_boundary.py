from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_fixed_standard_pair0_slope_capability_boundary.py"
CONFIG = ROOT / "configs" / "fixed_standard_pair0_slope_capability_boundary_v1_20260819.json"
SPEC = importlib.util.spec_from_file_location("pair0_slope_boundary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def safe_row(*, progress: float = 10.0, zero: int = 0) -> dict:
    return {
        "finite": True,
        "fall": False,
        "fixed_goal_success": False,
        "fixed_goal_best_progress_m": progress,
        "fixed_goal_net_progress_m": progress,
        "control_steps": 600,
        "physics_substeps": 3000,
        "full_interval_zero_foot_count": zero,
        "support_count_sum_physics_substeps": 4200,
        "supported_physics_substep_count": 2500,
        "force_qualified_supported_physics_substep_count": 2400,
        "qualified_slip_physics_substep_count": 0,
        "corrected_sustained_slip_physics_substep_count": 0,
        "corrected_slip_event_count": 0,
        "torso_ground_any": False,
        "sustained_nonfoot_contact": False,
        "cumulative_squared_action": 1.0,
        "actuator_abs_torque_time_integral_total_n_m_s": 1.0,
        "actuator_positive_mechanical_work_total_j": 1.0,
        "actuator_abs_mechanical_work_total_j": 1.0,
    }


def test_frozen_config_validates_and_checkpoint_is_final_pair0() -> None:
    config = load_config()
    protocol, reward, checkpoint = MODULE.validate_config(config)
    assert protocol["task_adapter"]["augment_local_terrain_observation"] is True
    assert reward
    assert checkpoint.name == "checkpoint_2727936.zip"
    assert config["source"]["checkpoint_sha256"] == MODULE.sha256(checkpoint)


def test_matrix_and_new_heldout_seeds_are_exact() -> None:
    config = load_config()
    assert config["evaluation"]["heldout_seeds"] == [94131, 94137, 94151, 94153, 94169]
    assert not any(str(seed).startswith(("828", "838")) for seed in config["evaluation"]["heldout_seeds"])
    specs = MODULE.scene_specs(config)
    assert len(specs) == 11
    assert [row["scene_name"] for row in specs] == config["evaluation"]["scene_order"]


def test_no_training_checkpoint_write_video_fixed_map_or_promotion_path() -> None:
    config = load_config()
    assert config["execution"] == {
        "formal_output_root": "artifacts/dev/fixed_standard_pair0_slope_capability_boundary_v1_20260819/attempt_0",
        "fail_if_output_root_exists": True,
        "training": False,
        "checkpoint_write": False,
        "fixed_map": False,
        "video": False,
        "promotion": False,
    }
    source = SCRIPT.read_text(encoding="utf-8")
    assert ".learn(" not in source
    assert "model.save(" not in source


def test_contact_friction_energy_and_observation_boundaries_are_frozen() -> None:
    config = load_config()
    assert len(config["contact_contract"]["distal_geoms"]) == 4
    assert config["contact_contract"]["geom_friction"] == [1.0, 0.5, 0.5]
    assert config["contact_contract"]["explicit_pair_margin_m"] == 0.0
    assert config["contact_contract"]["adhesion"] == 0.0
    assert config["invariants"]["explicit_pair_count"] == 4
    assert config["invariants"]["observation_dimension"] == 135
    assert config["energy_measurement"]["status"] == "measurement_only_not_reward_or_gate"


@pytest.mark.parametrize(
    ("direction", "progress", "expected"),
    [
        ("uphill", 6.1857992362, True),
        ("uphill", 6.1857992361, False),
        ("downhill", 8.8113570803, True),
        ("downhill", 8.8113570802, False),
        ("flat", -100.0, True),
    ],
)
def test_progress_gates_are_exact_and_flat_is_safety_only(
    direction: str, progress: float, expected: bool
) -> None:
    config = load_config()
    spec = {"scene_name": f"{direction}_test", "direction": direction, "angle_degrees": 8}
    result = MODULE.gate_scene(config, spec, [safe_row(progress=progress) for _ in range(5)])
    assert result["checks"]["effective_progress"] is expected


def test_safety_gate_fails_closed_on_zero_denominator_slip_and_zero_foot() -> None:
    config = load_config()
    spec = {"scene_name": "uphill_8deg", "direction": "uphill", "angle_degrees": 8}
    rows = [safe_row(progress=10.0, zero=35) for _ in range(5)]
    rows[0]["force_qualified_supported_physics_substep_count"] = 0
    rows[0]["corrected_sustained_slip_physics_substep_count"] = 1
    rows[0]["corrected_slip_event_count"] = 1
    for row in rows[1:]:
        row["force_qualified_supported_physics_substep_count"] = 0
    result = MODULE.gate_scene(config, spec, rows)
    assert result["passed"] is False
    assert "force_qualified_denominator_evaluable" in result["failed_checks"]
    assert "zero_corrected_sustained_slip" in result["failed_checks"]
    assert "zero_corrected_slip_events" in result["failed_checks"]
    assert "zero_foot_within_gate" in result["failed_checks"]


def test_boundary_inference_is_conservative_and_reports_nonmonotonicity() -> None:
    results = {
        f"uphill_{angle}deg": {"angle_degrees": angle, "passed": passed}
        for angle, passed in zip((4, 8, 12, 16, 20), (True, True, False, True, False))
    }
    boundary = MODULE.infer_tested_bounds(results, "uphill")
    assert boundary["highest_passing_tested_angle_degrees"] == 16
    assert boundary["conservative_tested_lower_bound_degrees"] == 8
    assert boundary["first_failing_tested_angle_degrees"] == 12
    assert boundary["nonmonotonic_pass_after_failure"] is True


def test_runtime_dependency_map_and_runner_hash_are_exact() -> None:
    config = load_config()
    observed = MODULE.validate_runtime_dependencies(config)
    assert observed == config["runtime_dependency_contract"]["exact_relative_path_sha256"]
    assert observed["scripts/evaluate_fixed_standard_pair0_slope_capability_boundary.py"] == MODULE.sha256(SCRIPT)


def test_runtime_closure_rejects_missing_extra_and_snapshot_mutation(
    tmp_path: Path,
) -> None:
    config = load_config()
    missing = copy.deepcopy(config)
    missing["runtime_dependency_contract"]["exact_relative_path_sha256"].pop(
        "src/proxygap/two_experiment_protocol.py"
    )
    with pytest.raises(ValueError, match="membership/order"):
        MODULE.validate_runtime_dependencies(missing)
    extra = copy.deepcopy(config)
    extra["runtime_dependency_contract"]["exact_relative_path_sha256"][
        "README.md"
    ] = MODULE.sha256(ROOT / "README.md")
    with pytest.raises(ValueError, match="membership/order"):
        MODULE.validate_runtime_dependencies(extra)

    MODULE.snapshot_runtime(config, tmp_path)
    snapshot_root = tmp_path / "runtime_snapshot"
    assert MODULE.validate_runtime_snapshot(config, snapshot_root)
    target = snapshot_root / "src" / "proxygap" / "metrics.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# mutation\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="snapshot changed"):
        MODULE.validate_runtime_snapshot(config, snapshot_root)
    unexpected = snapshot_root / "unexpected.txt"
    unexpected.write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact membership"):
        MODULE.validate_runtime_snapshot(config, snapshot_root)


def test_post_root_failure_is_recorded_and_non_retryable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = load_config()
    protocol, reward, checkpoint = MODULE.validate_config(config)
    monkeypatch.setattr(
        MODULE,
        "prepare_scenes",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected scene failure")),
    )
    output_root = tmp_path / "attempt_0"
    with pytest.raises(RuntimeError, match="injected scene failure"):
        MODULE.run(CONFIG, config, protocol, reward, checkpoint, output_root)
    failure = json.loads(
        (output_root / "FAILURE_RECORD.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "formal_attempt_failed_closed_non_evaluable"
    assert failure["failed_stage"] == "prepare_and_audit_scenes"
    assert failure["scientifically_evaluable"] is False
    assert failure["all_slope_decisions_withheld"] is True
    assert failure["retry_permitted"] is False
    assert failure["canonical_attempt_root_permanently_reserved"] is True
    assert failure["exception_type"] == "RuntimeError"
    assert "injected scene failure" in failure["traceback"]
    with pytest.raises(FileExistsError):
        MODULE.run(CONFIG, config, protocol, reward, checkpoint, output_root)


def test_mutated_checkpoint_hash_and_released_execution_fail_closed() -> None:
    config = load_config()
    changed = copy.deepcopy(config)
    changed["source"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        MODULE.validate_config(changed)
    changed = copy.deepcopy(config)
    changed["execution"]["training"] = True
    with pytest.raises(ValueError):
        MODULE.validate_config(changed)
