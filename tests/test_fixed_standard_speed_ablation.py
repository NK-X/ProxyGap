from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_fixed_standard_speed_ablation import (  # noqa: E402
    apply_full_substep_gate,
    prepare_native_plane_comparator,
    summarise_speed_matrix,
    validate_ablation_config,
)
from run_fixed_standard_support_curriculum import (  # noqa: E402
    prepare_standard_scenes,
    robot_signature,
)


CONFIG_PATH = ROOT / "configs" / "fixed_standard_speed_ablation_v1_20260819.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def minimal_row(model_id: str, speed: float, *, airborne: float, support: float, progress: float) -> dict:
    return {
        "model_id": model_id,
        "speed_m_per_s": speed,
        "scene_name": "flat",
        "fall": False,
        "fixed_goal_success": False,
        "fixed_goal_final_distance_m": 12.0 - progress,
        "termination_category": "none",
        "fixed_goal_best_progress_m": progress,
        "fixed_goal_net_progress_m": progress,
        "task_airborne_step_fraction": airborne,
        "longest_airborne_run_seconds": 0.5,
        "mean_support_count": support,
        "relative_torso_tilt_rms_rad": 0.05,
        "corrected_sustained_slip_step_fraction": 0.01,
        "endpoint_nonfoot_robot_ground_fraction": 0.0,
        "cumulative_squared_action": 100.0,
        "actuator_abs_torque_time_integral_total_n_m_s": 1000.0,
        "actuator_positive_mechanical_work_total_j": 500.0,
        "actuator_abs_mechanical_work_total_j": 600.0,
    }


def test_configuration_and_checkpoint_identities_validate() -> None:
    protocol, _ = validate_ablation_config(load_config())
    assert protocol["standard_scenes"]["fixed_friction"] == [1.0, 0.5, 0.5]


def test_speed_selection_rejects_stall_and_requires_substep_gate() -> None:
    config = load_config()
    rows = []
    for model in config["models"]:
        model_id = model["model_id"]
        rows.extend(
            [
                minimal_row(model_id, 0.20, airborne=0.40, support=0.70, progress=1.0),
                minimal_row(model_id, 0.30, airborne=0.60, support=0.45, progress=3.6),
                minimal_row(model_id, 0.40, airborne=0.69, support=0.31, progress=4.0),
                minimal_row(model_id, 0.55, airborne=0.70, support=0.30, progress=8.0),
            ]
        )
    summary = summarise_speed_matrix(config, rows)
    assert summary["selected_high_frequency_audit_speed_m_per_s"] == 0.3
    assert summary["endpoint_selection_gate_passed"] is True
    assert summary["speed_reduction_recommended"] is False
    high_frequency = {
        model["model_id"]: {
            "0.30": [{"full_interval_zero_foot_fraction": 0.20}],
            "0.55": [{"full_interval_zero_foot_fraction": 0.30}],
        }
        for model in config["models"]
    }
    final = apply_full_substep_gate(config, summary, high_frequency)
    assert final["full_substep_gate"]["passed"] is True
    assert final["speed_reduction_recommended"] is True


def test_full_substep_gate_can_veto_endpoint_candidate() -> None:
    config = load_config()
    summary = {
        "selected_high_frequency_audit_speed_m_per_s": 0.3,
        "endpoint_selection_gate_passed": True,
    }
    high_frequency = {
        model["model_id"]: {
            "0.30": [{"full_interval_zero_foot_fraction": 0.29}],
            "0.55": [{"full_interval_zero_foot_fraction": 0.30}],
        }
        for model in config["models"]
    }
    final = apply_full_substep_gate(config, summary, high_frequency)
    assert final["full_substep_gate"]["passed"] is False
    assert final["speed_reduction_recommended"] is False


def test_native_plane_comparator_preserves_robot_and_contact(tmp_path: Path) -> None:
    config = load_config()
    protocol, _ = validate_ablation_config(config)
    scenes, _ = prepare_standard_scenes(protocol, tmp_path)
    plane = prepare_native_plane_comparator(scenes["flat"], tmp_path)
    flat_model = mujoco.MjModel.from_xml_path(str(Path(scenes["flat"]["xml_path"])))
    plane_model = mujoco.MjModel.from_xml_path(str(Path(plane["xml_path"])))
    assert robot_signature(plane_model) == robot_signature(flat_model)
    floor_id = mujoco.mj_name2id(plane_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert int(plane_model.geom_type[floor_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
    assert np.array_equal(
        plane_model.geom_friction[floor_id], np.asarray([1.0, 0.5, 0.5])
    )
    assert int(plane_model.geom_condim[floor_id]) == 3
