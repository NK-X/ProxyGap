"""Rebuild the completed L2 attempt-0 manifest after a provenance audit.

This utility changes only the manifest.  Raw episode metrics, substep traces,
checkpoints, stop decisions and the prospective gate are deliberately excluded
from mutation.  It is single-use and verifies the original manifest hash before
writing the corrected, inventory-bearing manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = (
    ROOT
    / "artifacts"
    / "dev"
    / "fixed_standard_pair0_adaptation_l2_pilot_v1_20260819"
    / "attempt_0"
)
MANIFEST = ATTEMPT / "manifest.json"
PROVENANCE = ATTEMPT / "PROVENANCE_CORRECTION.json"
FROZEN_RUNNER = ATTEMPT / "frozen_runner.py"
FROZEN_CONFIG = ATTEMPT / "frozen_config.json"
LIVE_RUNNER = ROOT / "scripts" / "run_fixed_standard_pair0_adaptation_l2_pilot.py"
LIVE_CONFIG = ROOT / "configs" / "fixed_standard_pair0_adaptation_l2_pilot_v1_20260819.json"
ORIGINAL_MANIFEST_SHA256 = "d0ba008e55aefe74393eb4d041bbbc62d4f57286557c8fef2a5d8df745657cbc"
RUNTIME_RUNNER_SHA256 = "34c5fe1c660fc0c2d72bf617646857c9c90cebf1c2cec131ca832e648bb23346"
RUNTIME_CONFIG_SHA256 = "3fde34618a02ce0fb7134f8b852eb5b8ed0b4c72f041b83da70fae47dd931be2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(ATTEMPT.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        records.append(
            {
                "relative_path": path.relative_to(ATTEMPT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def main() -> None:
    observed_manifest_hash = sha256(MANIFEST)
    if observed_manifest_hash != ORIGINAL_MANIFEST_SHA256:
        raise RuntimeError(
            "Refusing to rebuild an unexpected manifest: "
            f"{observed_manifest_hash} != {ORIGINAL_MANIFEST_SHA256}"
        )
    if sha256(FROZEN_RUNNER) != RUNTIME_RUNNER_SHA256:
        raise RuntimeError("The exact runtime runner freeze changed")
    if sha256(FROZEN_CONFIG) != RUNTIME_CONFIG_SHA256:
        raise RuntimeError("The exact runtime config freeze changed")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["schema_version"] = "proxygap-pair0-adaptation-l2-manifest-v2-corrected"
    manifest["status"] = "l2_exploratory_pilot_complete_with_post_run_provenance_correction"
    manifest["original_manifest_sha256_before_correction"] = ORIGINAL_MANIFEST_SHA256
    manifest["runtime_freeze"] = {
        "runner_path": "frozen_runner.py",
        "runner_sha256": RUNTIME_RUNNER_SHA256,
        "config_path": "frozen_config.json",
        "config_sha256": RUNTIME_CONFIG_SHA256,
    }
    manifest["post_run_live_sources_not_used_to_generate_attempt_0"] = {
        "runner_path": str(LIVE_RUNNER.resolve()),
        "runner_sha256": sha256(LIVE_RUNNER),
        "config_path": str(LIVE_CONFIG.resolve()),
        "config_sha256": sha256(LIVE_CONFIG),
        "purpose": "Future-run provenance, force-qualified denominator and fail-closed corrections only",
    }
    manifest["provenance_correction"] = {
        "path": "PROVENANCE_CORRECTION.json",
        "sha256": sha256(PROVENANCE),
    }
    manifest["attempt_0_measurement_boundary"] = {
        "raw_metrics_or_gate_recalculated": False,
        "raw_metrics_or_gate_overwritten": False,
        "runtime_slip_denominator": "any-contact supported physics substeps",
        "future_primary_slip_denominator": "force-qualified (normal force >= 1 N) supported physics substeps",
        "future_zero_primary_denominator": "non_evaluable",
        "bounded_unchanged_result": "Both conditions recorded zero sustained-slip substeps and zero sustained-slip events under the runtime definition; relative qualified-rate inference remains denominator-sensitive.",
    }
    manifest["artifact_hashes_excluding_manifest"] = inventory()
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(MANIFEST), "sha256": sha256(MANIFEST), "artifacts": len(manifest["artifact_hashes_excluding_manifest"])}, indent=2))


if __name__ == "__main__":
    main()
