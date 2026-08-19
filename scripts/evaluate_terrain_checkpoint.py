"""Pilot-evaluate one frozen curved-gait policy on fixed smooth terrains.

This script is diagnostic.  It does not train, alter reward coefficients or
claim generalisation.  Terrain bundles are verified before use, and ground
friction must remain exactly ``[1.0, 0.5, 0.5]`` with ``condim=3``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from proxygap.curved_gait import make_curved_gait_env  # noqa: E402
from proxygap_terrain import (  # noqa: E402
    TerrainBundle,
    ascii_bundle_xml,
    load_terrain_bundle,
)
from run_curved_gait_training import common_env_kwargs  # noqa: E402


EXPECTED_GROUND_FRICTION = np.asarray([1.0, 0.5, 0.5], dtype=np.float64)
EXPECTED_GROUND_CONDIM = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Verified terrain bundle. May be supplied more than once.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=[61001, 61002, 61003],
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--without-flat",
        action="store_true",
        help="Do not include the stock flat Ant-v5 floor as a baseline.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bundle_arguments(values: list[str]) -> list[tuple[str, TerrainBundle]]:
    parsed: list[tuple[str, TerrainBundle]] = []
    seen: set[str] = set()
    for value in values:
        label, separator, raw_path = value.partition("=")
        label = label.strip()
        if not separator or not label or label in seen:
            raise ValueError(f"bundle must be a unique LABEL=PATH value: {value!r}")
        bundle = load_terrain_bundle(Path(raw_path).expanduser().resolve())
        contact = bundle.manifest["contact"]
        friction = np.asarray(contact["ground_friction"], dtype=np.float64)
        condim = int(contact["ground_condim"])
        if not np.array_equal(friction, EXPECTED_GROUND_FRICTION):
            raise ValueError(f"{label} does not use the frozen ground friction")
        if condim != EXPECTED_GROUND_CONDIM:
            raise ValueError(f"{label} does not use condim=3")
        parsed.append((label, bundle))
        seen.add(label)
    return parsed


def compiled_ground_contract(env: Any) -> dict[str, Any]:
    model = env.unwrapped.model
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise RuntimeError("compiled model has no floor geom")
    friction = np.asarray(model.geom_friction[floor_id], dtype=np.float64)
    condim = int(model.geom_condim[floor_id])
    return {
        "friction": friction.tolist(),
        "friction_matches": bool(np.array_equal(friction, EXPECTED_GROUND_FRICTION)),
        "condim": condim,
        "condim_matches": condim == EXPECTED_GROUND_CONDIM,
    }


def terrain_height_and_normal(
    bundle: TerrainBundle | None,
    x_m: float,
    y_m: float,
) -> tuple[float, np.ndarray, float]:
    if bundle is None:
        return 0.0, np.asarray([0.0, 0.0, 1.0]), 0.0
    if (
        abs(x_m) > bundle.half_extent_x_m
        or abs(y_m) > bundle.half_extent_y_m
    ):
        raise ValueError("robot left the finite heightfield")
    dx = 2.0 * bundle.half_extent_x_m / (bundle.cols - 1)
    dy = 2.0 * bundle.half_extent_y_m / (bundle.rows - 1)
    x_low = max(-bundle.half_extent_x_m, x_m - dx)
    x_high = min(bundle.half_extent_x_m, x_m + dx)
    y_low = max(-bundle.half_extent_y_m, y_m - dy)
    y_high = min(bundle.half_extent_y_m, y_m + dy)
    height = float(bundle.height_at(x_m, y_m))
    dz_dx = float(bundle.height_at(x_high, y_m) - bundle.height_at(x_low, y_m)) / (
        x_high - x_low
    )
    dz_dy = float(bundle.height_at(x_m, y_high) - bundle.height_at(x_m, y_low)) / (
        y_high - y_low
    )
    normal = np.asarray([-dz_dx, -dz_dy, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    slope = float(math.hypot(dz_dx, dz_dy))
    return height, normal, slope


def torso_up_vector(qpos: np.ndarray) -> np.ndarray:
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, np.asarray(qpos[3:7], dtype=np.float64))
    return rotation.reshape(3, 3)[:, 2]


def safe_rms(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else float("nan")


def run_rollout(
    *,
    model: PPO,
    config: dict[str, Any],
    label: str,
    bundle: TerrainBundle | None,
    xml_file: Path | None,
    evaluation_seed: int,
    steps: int,
    speed: float,
) -> dict[str, Any]:
    env = make_curved_gait_env(
        condition_id=f"TERRAIN_PILOT_{label.upper()}",
        seed=evaluation_seed,
        xml_file=xml_file,
        max_episode_steps=steps,
        profile="straight",
        speed_min=speed,
        speed_max=speed,
        max_abs_curvature=0.0,
        max_abs_lateral_speed=0.0,
        fixed_lateral_speed=0.0,
        heading_termination_enabled=False,
        **common_env_kwargs(config),
    )
    contract = compiled_ground_contract(env)
    if not contract["friction_matches"] or not contract["condim_matches"]:
        env.close()
        raise RuntimeError(f"compiled ground contract changed for {label}")
    observation, _ = env.reset(seed=evaluation_seed)
    clearances: list[float] = []
    relative_tilts: list[float] = []
    local_slopes: list[float] = []
    encountered_heights: list[float] = []
    out_of_bounds = False
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
        try:
            terrain_height, normal, slope = terrain_height_and_normal(
                bundle,
                float(qpos[0]),
                float(qpos[1]),
            )
        except ValueError:
            out_of_bounds = True
            break
        up = torso_up_vector(qpos)
        cosine = float(np.clip(np.dot(up, normal), -1.0, 1.0))
        clearances.append(float(qpos[2] - terrain_height))
        relative_tilts.append(float(math.acos(cosine)))
        local_slopes.append(slope)
        encountered_heights.append(terrain_height)
    summary = env.episode_summary()
    qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
    result = {
        "terrain_label": label,
        "terrain_id": (
            None if bundle is None else bundle.manifest["seed"]["terrain_id"]
        ),
        "terrain_kind": "flat" if bundle is None else bundle.manifest["seed"]["terrain_kind"],
        "evaluation_seed": evaluation_seed,
        "requested_steps": steps,
        "observed_steps": int(summary["episode_length"]),
        "full_horizon_completed": bool(
            int(summary["episode_length"]) == steps
            and not bool(summary["terminated"])
            and not out_of_bounds
        ),
        "terminated": bool(summary["terminated"]),
        "termination_category": summary["termination_category"],
        "out_of_bounds": out_of_bounds,
        "net_x_displacement_m": float(qpos[0]),
        "net_y_displacement_m": float(qpos[1]),
        "surface_relative_torso_clearance_min_m": (
            float(min(clearances)) if clearances else float("nan")
        ),
        "surface_relative_tilt_rms_rad": safe_rms(relative_tilts),
        "surface_relative_tilt_max_rad": (
            float(max(relative_tilts)) if relative_tilts else float("nan")
        ),
        "encountered_local_slope_max": (
            float(max(local_slopes)) if local_slopes else float("nan")
        ),
        "encountered_local_slope_max_degrees": (
            float(math.degrees(math.atan(max(local_slopes))))
            if local_slopes
            else float("nan")
        ),
        "encountered_height_min_m": (
            float(min(encountered_heights)) if encountered_heights else float("nan")
        ),
        "encountered_height_max_m": (
            float(max(encountered_heights)) if encountered_heights else float("nan")
        ),
        "airborne_step_fraction": float(summary["airborne_step_fraction"]),
        "longest_airborne_run_seconds": float(summary["longest_airborne_run_seconds"]),
        "foot_contact_slip_distance_m_by_foot": summary[
            "foot_contact_slip_distance_m_by_foot"
        ],
        "foot_contact_slip_distance_total_m": float(
            sum(summary["foot_contact_slip_distance_m_by_foot"])
        ),
        "torso_world_tilt_rms_rad": float(summary["torso_tilt_rms"]),
        "actuator_positive_mechanical_work_j_total": float(
            sum(summary["actuator_positive_mechanical_work_j_by_actuator"])
        ),
        "actuator_negative_mechanical_work_abs_j_total": float(
            sum(summary["actuator_negative_mechanical_work_abs_j_by_actuator"])
        ),
        "compiled_ground": contract,
    }
    env.close()
    return result


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or not math.isfinite(args.speed) or args.speed <= 0:
        raise ValueError("steps and speed must be positive")
    config_path = args.config.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = PPO.load(model_path, device=args.device)
    terrain_cases = parse_bundle_arguments(args.bundle)
    cases: list[tuple[str, TerrainBundle | None]] = []
    if not args.without_flat:
        cases.append(("flat", None))
    cases.extend(terrain_cases)
    if not cases:
        raise ValueError("at least one flat or terrain condition is required")
    rows: list[dict[str, Any]] = []
    with ExitStack() as stack:
        xml_paths: dict[str, Path | None] = {}
        for label, bundle in cases:
            context = nullcontext(None) if bundle is None else ascii_bundle_xml(bundle)
            xml_paths[label] = stack.enter_context(context)
        for label, bundle in cases:
            for evaluation_seed in args.evaluation_seeds:
                rows.append(
                    run_rollout(
                        model=model,
                        config=config,
                        label=label,
                        bundle=bundle,
                        xml_file=xml_paths[label],
                        evaluation_seed=int(evaluation_seed),
                        steps=args.steps,
                        speed=args.speed,
                    )
                )
    payload = {
        "schema_version": "proxygap-terrain-policy-pilot-v1",
        "status": "development_diagnostic_not_formal_evidence",
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "evaluation_seeds": [int(value) for value in args.evaluation_seeds],
        "steps": int(args.steps),
        "speed_m_per_s": float(args.speed),
        "friction_contract": EXPECTED_GROUND_FRICTION.tolist(),
        "ground_condim_contract": EXPECTED_GROUND_CONDIM,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "rows": len(rows)}))


if __name__ == "__main__":
    main()
