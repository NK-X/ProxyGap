from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_fixed_standard_pair0_turn_balance_continuation_v3_gate_repair as runner  # noqa: E402


CONFIG_PATH = (
    ROOT
    / "configs"
    / "fixed_standard_pair0_turn_balance_continuation_v3_gate_repair_20260819.json"
)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_v3_is_gate_only_and_preserves_the_complete_v2_scientific_protocol() -> None:
    config = load_config()
    frozen = config["frozen_v2_protocol"]
    assert runner.sha256(ROOT / frozen["configuration"]) == frozen[
        "configuration_sha256"
    ]
    assert runner.sha256(ROOT / frozen["runner"]) == frozen["runner_sha256"]
    for key in (
        "scientific_protocol_changed",
        "seeds_changed",
        "budgets_changed",
        "commands_changed",
        "reward_changed",
        "friction_changed",
        "energy_changed",
        "gates_changed",
    ):
        assert frozen[key] is False
    source = (ROOT / runner.RUNTIME_SELF).read_text(encoding="utf-8")
    assert ".learn(" not in source
    assert "model.save(" not in source
    assert config["execution"]["smoke_rerun"] is False
    assert config["post_formal_boundary"]["further_optimisation_authorised"] is False


def test_exact_two_scene_manifest_allowlist_is_frozen_by_path_size_and_hash() -> None:
    config = load_config()
    observed = tuple(
        (row["relative_path"], row["size_bytes"], row["sha256"])
        for row in config["exact_scene_manifest_allowlist"]
    )
    assert observed == runner.EXPECTED_ALLOWLIST
    changed = copy.deepcopy(config)
    changed["exact_scene_manifest_allowlist"][0]["size_bytes"] += 1
    with pytest.raises(ValueError, match="exact scene-manifest allowlist"):
        runner.validate_config(changed)
    extra = copy.deepcopy(config)
    extra["exact_scene_manifest_allowlist"].append(
        {"relative_path": "extra/manifest.json", "size_bytes": 1, "sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="exact scene-manifest allowlist"):
        runner.validate_config(extra)


def test_completed_v2_smoke_passes_only_the_gate_repair_and_is_not_mutated() -> None:
    config = load_config()
    smoke_manifest = ROOT / config["immutable_completed_smoke"]["manifest"]
    before = runner.sha256(smoke_manifest)
    v2_config, *_ = runner.validate_config(config)
    evidence = runner.validate_completed_v2_smoke(config, v2_config)
    after = runner.sha256(smoke_manifest)
    assert before == after == config["immutable_completed_smoke"]["manifest_sha256"]
    assert evidence[
        "V2_original_strict_validation_reached_only_allowlist_rejection"
    ] is True
    assert evidence["full_V2_smoke_semantics_runtime_and_inventory_validated"] is True
    assert evidence["scientific_protocol_changed"] is False
    assert evidence["smoke_rerun"] is False
    assert evidence["checkpoint_written"] is False
    assert evidence["exact_scene_manifest_allowlist"] == config[
        "exact_scene_manifest_allowlist"
    ]


def test_gate_runtime_and_immutable_smoke_hashes_are_exact() -> None:
    config = load_config()
    assert tuple(config["gate_runtime_contract"]["exact_relative_path_sha256"]) == (
        runner.RUNTIME_SELF,
        "configs/fixed_standard_pair0_turn_balance_continuation_v2_20260819.json",
        "scripts/run_fixed_standard_pair0_turn_balance_continuation_v2.py",
    )
    assert all(
        digest != "<TO_FREEZE_RUNNER>"
        for digest in config["gate_runtime_contract"]["exact_relative_path_sha256"].values()
    )
    assert runner.validate_gate_runtime(config) == config["gate_runtime_contract"][
        "exact_relative_path_sha256"
    ]
    smoke = config["immutable_completed_smoke"]
    manifest = ROOT / smoke["manifest"]
    assert manifest.stat().st_size == smoke["manifest_size_bytes"]
    assert runner.sha256(manifest) == smoke["manifest_sha256"]


def test_unique_formal_root_is_sealed_as_a_pretraining_failure() -> None:
    config = load_config()
    root = runner.formal_root(config)
    assert root == (
        ROOT / "artifacts/dev/pair0_turn_balance_v2_20260819/attempt_0"
    ).resolve()
    failure = root / "FAILURE_RECORD.json"
    assert root.is_dir()
    assert runner.sha256(failure) == (
        "21ccdebc692af2f32ec96a2e33795cd0eac45ea4aac852eface8a02f26709d23"
    )
    assert not list(root.rglob("*.zip"))
    assert not list(root.rglob("training_record.json"))
