"""Pilot forward-progress shaping on the formally identified divergent setting."""

from __future__ import annotations

import atexit
import ctypes
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.experiment import (  # noqa: E402
    save_run_config,
    summarise_evaluation,
    train_condition,
    write_standard_outputs,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "shaping_pilot_v1_20260808.json"
PILOT_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RUN_ID = str(PILOT_CONFIG["run_id"])
TIMESTEPS = int(PILOT_CONFIG["timesteps_per_condition"])
CHECKPOINTS = [int(value) for value in PILOT_CONFIG["checkpoint_timesteps"]]
EVAL_EPISODES = int(PILOT_CONFIG["eval_episodes_per_checkpoint"])
TRAINING_SEED = int(PILOT_CONFIG["training_seed"])
EVALUATION_SEED_BASE = int(PILOT_CONFIG["evaluation_seed_base"])
CTRL_COST_WEIGHT = float(PILOT_CONFIG["ctrl_cost_weight"])
SHAPING_WEIGHTS = [
    float(value) for value in PILOT_CONFIG["forward_progress_shaping_weights"]
]


def request_windows_awake() -> None:
    if sys.platform != "win32":
        return
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
        es_continuous | es_system_required
    )
    if result == 0:
        raise OSError("Windows rejected the shaping-pilot sleep prevention request")
    atexit.register(
        ctypes.windll.kernel32.SetThreadExecutionState,  # type: ignore[attr-defined]
        es_continuous,
    )


def condition_id(weight: float) -> str:
    if weight == 0.0:
        return "unshaped_ctrl_0p0625"
    return f"forward_shape_{str(weight).replace('.', 'p')}"


def main() -> None:
    request_windows_awake()
    output_root = PROJECT_ROOT / "artifacts" / "pilot" / RUN_ID
    if output_root.exists():
        raise FileExistsError(f"Pilot output already exists: {output_root}")

    config = dict(PILOT_CONFIG)
    config["source_config"] = str(CONFIG_PATH)
    save_run_config(output_root, config)

    all_runtime_rows = []
    all_eval_rows = []
    for weight in SHAPING_WEIGHTS:
        cid = condition_id(weight)
        print(f"Starting {cid}", flush=True)
        runtime_rows, eval_rows = train_condition(
            output_root=output_root,
            condition_id=cid,
            ctrl_cost_weight=CTRL_COST_WEIGHT,
            forward_progress_shaping_weight=weight,
            total_timesteps=TIMESTEPS,
            checkpoint_timesteps=CHECKPOINTS,
            seed=TRAINING_SEED,
            evaluation_seed_base=EVALUATION_SEED_BASE,
            eval_episodes=EVAL_EPISODES,
            eval_max_episode_steps=int(PILOT_CONFIG["eval_max_episode_steps"]),
            ppo_n_steps=int(PILOT_CONFIG["ppo"]["n_steps"]),
            ppo_batch_size=int(PILOT_CONFIG["ppo"]["batch_size"]),
            ppo_n_epochs=int(PILOT_CONFIG["ppo"]["n_epochs"]),
        )
        all_runtime_rows.extend(runtime_rows)
        all_eval_rows.extend(eval_rows)
        write_standard_outputs(
            output_root,
            runtime_rows=all_runtime_rows,
            eval_rows=all_eval_rows,
            summary_rows=summarise_evaluation(all_eval_rows),
        )
        print(f"Completed {cid}", flush=True)

    (output_root / "completed.json").write_text(
        json.dumps({"run_id": RUN_ID, "status": "completed"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved shaping pilot: {output_root}", flush=True)


if __name__ == "__main__":
    main()
