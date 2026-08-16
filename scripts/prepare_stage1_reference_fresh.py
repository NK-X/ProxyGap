"""Initialise the frozen V6 fresh-reference output without overwriting data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap.reference_baseline import validate_reference_config  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage1_reference_fresh_1m_v6_20260814.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_project_path(value: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    if PROJECT_ROOT not in path.parents:
        raise ValueError(f"Path escapes the project root: {value}")
    return path


def verify_sources(config: dict[str, Any]) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    for record in config["source_hashes"]:
        path = resolve_project_path(str(record["path"]))
        actual = sha256(path)
        if actual != str(record["sha256"]).upper():
            raise ValueError(f"Frozen source hash mismatch: {path}")
        verified.append({"path": str(path), "sha256": actual})
    return verified


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_reference_config(config)
    sources = verify_sources(config)

    parent = config["parent_gate"]
    parent_path = resolve_project_path(str(parent["path"]))
    if sha256(parent_path) != str(parent["sha256"]).upper():
        raise ValueError("The parent V5 gate hash changed")

    protocol_path = resolve_project_path(str(config["protocol_document"]))
    output_root = resolve_project_path(str(config["output_root"]))
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    run_config_path = output_root / "run_config.json"
    run_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "initialised",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "run_config_path": str(run_config_path),
        "run_config_sha256": sha256(run_config_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "parent_gate_path": str(parent_path),
        "parent_gate_sha256": sha256(parent_path),
        "verified_training_sources": sources,
        "expected_policies": 5,
        "expected_model_checkpoints": 20,
        "expected_evaluation_rows": 400,
        "formal_launch": "prohibited",
        "shaping_launch": "prohibited",
    }
    manifest_path = output_root / "initialisation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
