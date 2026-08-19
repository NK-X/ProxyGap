from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_post_seal_full_map_v1 as subject  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "post_seal_full_map_eval_v1_20260819.json"


def run_fresh_validation() -> dict[str, object]:
    """Run the production validator in a clean interpreter process."""

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_post_seal_full_map_v1.py"),
            "--mode",
            "validate",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


def fresh_loaded_project_runtime_paths() -> set[str]:
    """Inspect the imported project closure without pytest collection pollution."""

    code = (
        "import json,sys; "
        "sys.path.insert(0,'scripts'); "
        "import evaluate_post_seal_full_map_v1 as subject; "
        "print(json.dumps(sorted(subject.loaded_project_runtime_paths())))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return set(json.loads(completed.stdout))


def test_frozen_contract_validates_and_is_single_fresh_seed() -> None:
    validation = run_fresh_validation()
    assert validation["status"] == "validated_no_evaluation_run"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fixed = json.loads(
        (ROOT / config["fixed_map"]["configuration"]).read_text(encoding="utf-8")
    )
    assert config["evaluation"]["formal_seed"] == 1763594348
    assert config["evaluation"]["formal_seed_derivation"]["sha256"] == (
        "691e506c7d159656fe0a7e59b8da97bdbf977e8b9d0273742e052ba5f9433122"
    )
    assert config["evaluation"]["horizon_control_steps"] == 12000
    assert config["evaluation"]["deterministic_policy"] is True
    assert config["evaluation"]["physics_substeps_per_control_step"] == 5
    assert config["engineering_smoke"]["seed"] == 96818
    assert config["engineering_smoke"]["horizon_control_steps"] == 20
    assert config["execution"]["formal_output_root"] != config["execution"]["smoke_output_root"]
    assert config["execution"]["smoke_output_root"].endswith("attempt_2")
    assert fixed["approved_map"]["start_xy_m"] == [-34.0, -34.0]
    assert fixed["approved_map"]["goal_xy_m"] == [34.0, 34.0]


def test_final_checkpoint_and_local_preview_contract_are_frozen() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source = config["source"]
    assert source["sealed_checkpoint"].endswith(
        "pair0_adapt/models/checkpoint_2727936.zip"
    )
    assert source["sealed_checkpoint_sha256"] == (
        "5121abeff92859205e1537f123f0df1e97edb5ea1fa80be1a72959a5931fac1c"
    )
    observation = config["observation_contract"]
    assert observation["dimension"] == 135
    assert observation["base_observation_dimension"] == 122
    assert observation["local_terrain_preview_dimension"] == 13
    assert observation["augment_local_terrain_observation"] is True
    assert observation["policy_observes_global_position"] is False
    assert observation["policy_observes_goal_coordinates"] is False
    assert observation["high_level_controller_uses_global_position"] is True


def test_pair0_injection_contract_contains_exactly_four_named_pairs() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    pair = subject.pair_contract_for_injection(config)
    block = subject.pair_tools.explicit_pair_block(pair)
    assert block.count("<pair ") == 4
    for foot in config["contact_contract"]["distal_geoms"]:
        assert f'geom2="{foot}"' in block
    assert 'margin="0"' in block
    assert 'gap="0"' in block
    assert 'condim="3"' in block
    assert 'friction="1 1 0.5 0.5 0.5"' in block
    assert config["fixed_map"]["prepared_spawn_scene_expected_sha256"] == (
        "bc72dd66e6b171c0444e8e55d19b145b0a2095d4b584e407df9d76c4033096a7"
    )
    assert config["fixed_map"]["pair0_artifact_scene_expected_sha256"] == (
        "0b75be52ac76ef0147fb6f5e9eccce025067490d144892006c8098dfe1e5be46"
    )


def test_arrival_tracker_requires_entry_then_continuous_stable_dwell() -> None:
    tracker = subject.direct.ArrivalDwellTracker(
        arrival_radius_m=1.5,
        hold_radius_m=2.0,
        required_hold_steps=40,
    )
    for step in range(1, 80):
        tracker.update(step=step, distance_m=1.8, stable=True)
    assert tracker.goal_entered is False
    assert tracker.spatial_success is False
    assert tracker.strict_dwell_success is False
    tracker.update(step=80, distance_m=1.4, stable=True)
    assert tracker.goal_entered is True
    assert tracker.entry_step == 80
    for step in range(81, 119):
        tracker.update(step=step, distance_m=1.8, stable=True)
    assert tracker.strict_dwell_success is False
    tracker.update(step=119, distance_m=1.8, stable=True)
    assert tracker.spatial_success is True
    assert tracker.strict_dwell_success is True


def test_duration_corrected_slip_uses_physics_dt_grace_force_and_duration() -> None:
    tracker = subject.l2.DurationCorrectedSlipTracker(
        dt=0.01,
        speed_threshold=0.2,
        minimum_normal_force=1.0,
        landing_grace_seconds=0.1,
        minimum_sustained_seconds=0.2,
    )
    contact = np.asarray([True, False, False, False])
    speed = np.asarray([0.3, 0.0, 0.0, 0.0])
    force = np.asarray([2.0, 0.0, 0.0, 0.0])
    for _ in range(10):
        _, qualified = tracker.update(
            contact_mask=contact,
            tangential_speeds=speed,
            normal_forces=force,
        )
        assert not np.any(qualified)
    for _ in range(20):
        tracker.update(
            contact_mask=contact,
            tangential_speeds=speed,
            normal_forces=force,
        )
    result = tracker.finalise()
    assert len(result["events"]) == 1
    assert result["events"][0]["duration_steps"] == 20
    assert result["events"][0]["duration_seconds"] == 0.2


def test_runner_has_no_training_or_checkpoint_serialisation_call() -> None:
    source = (ROOT / "scripts" / "evaluate_post_seal_full_map_v1.py").read_text(
        encoding="utf-8"
    )
    assert ".learn(" not in source
    assert ".save(" not in source
    assert "PPO.load(" in source
    assert "--mode" in source
    assert "default=VALIDATE_MODE" in source
    assert "except BaseException" in source
    assert 'newline=""' in source


def test_runtime_dependency_map_is_the_actual_imported_closure() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    declared = set(config["runtime_dependencies"])
    assert len(declared) == 19
    assert "scripts/evaluate_post_seal_full_map_v1.py" in declared
    assert declared == fresh_loaded_project_runtime_paths()
    assert subject.live_runtime_dependency_map(config) == config["runtime_dependencies"]


def test_qualification_and_energy_boundaries_are_predeclared() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    qualification = config["safety_and_qualification"]
    assert qualification["requires_strict_stable_dwell_success"] is True
    assert qualification["requires_no_fall"] is True
    assert qualification["requires_zero_full_control_intervals_with_all_four_distal_feet_airborne"] is True
    assert qualification["requires_zero_duration_corrected_slip_events"] is True
    assert config["energy_boundary"]["status"] == "measurement_only_not_reward_or_gate"
    assert config["energy_boundary"]["reward_weight"] == 0.0
    assert config["energy_boundary"]["electrical_battery_energy_claim_permitted"] is False
