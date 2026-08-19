from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_CONFIG = ROOT / "configs/v4_pair0_multiobjective_full_map_final_v1_20260820.json"
VIDEO_CONFIG = ROOT / "configs/v4_pair0_multiobjective_full_map_video_v1_20260820.json"
FINAL_ROOT = ROOT / "artifacts/dev/v4_pair0_multiobjective_full_map_final_v1_20260820/attempt_0"
VIDEO_ROOT = ROOT / "artifacts/dev/v4_pair0_multiobjective_full_map_video_v1_20260820"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_config_and_seed_derivation_validate() -> None:
    module = load_module(
        "run_v4_pair0_multiobjective_full_map_final_test",
        ROOT / "scripts/run_v4_pair0_multiobjective_full_map_final.py",
    )
    config = json.loads(FINAL_CONFIG.read_text(encoding="utf-8"))
    module.validate(config)
    assert config["evaluation"]["formal_seeds"] == [690223864, 1864999454, 952993985]
    assert config["candidate_selection"]["feasible_candidate_count"] == 15
    assert config["execution"]["training_permitted"] is False


def test_final_result_is_six_of_six_without_fall_or_sustained_slip() -> None:
    summary = json.loads((FINAL_ROOT / "summary.json").read_text(encoding="utf-8"))
    assert summary["all_objectives_passed"] is True
    assert summary["episode_count"] == 6
    assert summary["unique_contract_count"] == 2
    for objective in summary["objectives"].values():
        assert objective["success_count"] == 3
        assert objective["episode_count"] == 3
        assert objective["fall_count"] == 0
        assert objective["total_duration_corrected_slip_events"] == 0


def test_video_config_binds_three_predeclared_formal_episodes() -> None:
    module = load_module(
        "render_v4_pair0_multiobjective_full_map_videos_test",
        ROOT / "scripts/render_v4_pair0_multiobjective_full_map_videos.py",
    )
    config = module.validate_config(VIDEO_CONFIG)
    assert [episode["objective_id"] for episode in config["episodes"]] == [
        "time_priority",
        "balanced",
        "energy_priority",
    ]
    assert [episode["evaluation_seed"] for episode in config["episodes"]] == [
        690223864,
        1864999454,
        952993985,
    ]


def test_video_archive_is_exact_and_inventory_closed() -> None:
    manifest = json.loads((VIDEO_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "final_verified_three_objective_exact_formal_replays"
    assert manifest["episode_count"] == 3
    assert manifest["all_formal_episodes_successful"] is True
    assert manifest["all_duration_corrected_slip_event_counts_zero"] is True
    assert manifest["all_replays_exact"] is True
    declared = {item["relative_path"]: item for item in manifest["artifact_inventory_excludes_manifest_and_digest"]}
    actual = {
        path.relative_to(VIDEO_ROOT).as_posix(): path
        for path in VIDEO_ROOT.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}
    }
    assert set(declared) == set(actual)
    for relative, path in actual.items():
        assert declared[relative]["bytes"] == path.stat().st_size
        assert declared[relative]["sha256"] == sha256(path)
    for objective in ("time_priority", "balanced", "energy_priority"):
        episode = json.loads((VIDEO_ROOT / objective / "episode_manifest.json").read_text(encoding="utf-8"))
        assert episode["status"] == "final_verified_exact_formal_replay"
        assert episode["exactness"]["state_mismatch_count"] == 0
        assert episode["exactness"]["substep_mismatch_count"] == 0
        assert episode["video"]["qa"]["decoded_frames"] == episode["video"]["frames"]
