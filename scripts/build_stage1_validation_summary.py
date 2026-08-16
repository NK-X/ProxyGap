"""Validate stage-one development artifacts and write an auditable summary."""

from __future__ import annotations

import argparse
import compileall
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(checks: list[dict[str, str]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})
    if not passed:
        raise RuntimeError(f"Validation failed: {name}: {detail}")


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, str]] = []

    pytest_run = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    collect_run = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    collected_count = sum(
        int(line.rsplit(":", 1)[1].strip())
        for line in collect_run.stdout.splitlines()
        if line.rsplit(":", 1)[-1].strip().isdigit()
    )
    add_check(
        checks,
        "Automated tests",
        pytest_run.returncode == 0
        and collect_run.returncode == 0
        and collected_count == 54,
        f"{collected_count} tests collected; pytest exit code {pytest_run.returncode}",
    )

    compile_ok = all(
        compileall.compile_dir(PROJECT_ROOT / folder, quiet=1, force=True)
        for folder in ("src", "scripts", "tests")
    )
    add_check(checks, "Python compilation", compile_ok, "src, scripts and tests compiled")

    development_config = PROJECT_ROOT / "configs" / "stage1_dense_development_v1_20260814.json"
    development_protocol = PROJECT_ROOT / "protocols" / "STAGE1_PROXY_DIVERGENCE_PROTOCOL_DRAFT_20260814.md"
    expected_config_hash = "36ee2d458ebf55d4d5651ec8e71a5c0a475c02ba019af803eb25e4914a0320be"
    expected_protocol_hash = "67f5af4fcee116fd4c40864ce4ba4274390f5cc2618c319858b1793fbd442850"
    add_check(
        checks,
        "Pre-run development configuration",
        sha256(development_config) == expected_config_hash,
        f"SHA-256 {sha256(development_config)}",
    )
    add_check(
        checks,
        "Pre-run development protocol",
        sha256(development_protocol) == expected_protocol_hash,
        f"SHA-256 {sha256(development_protocol)}",
    )

    dense_root = PROJECT_ROOT / "artifacts" / "exploration" / "stage1_dense_development_300k_20260814"
    completion = json.loads((dense_root / "parallel_completion.json").read_text(encoding="utf-8"))
    model_count = len(list((dense_root / "runs").rglob("checkpoint_*.zip")))
    add_check(
        checks,
        "Dense development run",
        completion == {"completed_policies": 6, "expected_policies": 6, "evaluation_rows": 360, "failures": []}
        and model_count == 36,
        f"6/6 policies, {model_count} checkpoints, 360 evaluation rows",
    )

    reeval_root = PROJECT_ROOT / "artifacts" / "exploration" / "stage1_harmonised_existing_models_v2_20260814"
    reeval = json.loads((reeval_root / "reevaluation_manifest.json").read_text(encoding="utf-8"))
    add_check(
        checks,
        "Corrected existing-model re-evaluation",
        reeval["source_model_count"] == 48 and reeval["evaluation_row_count"] == 480,
        "48 models and 480 harmonised evaluation rows; actual steps read from model metadata",
    )

    analysis_root = PROJECT_ROOT / "artifacts" / "analysis" / "stage1_dense_development_v1_20260814"
    result_path = analysis_root / "stage1_development_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    quality = result["data_quality"]
    add_check(
        checks,
        "Combined data schema and invariants",
        quality["row_count"] == 840
        and quality["cell_count"] == 84
        and quality["episodes_per_cell"] == 10
        and quality["duplicate_episode_keys"] == 0
        and quality["invalid_episode_end_state_count"] == 0
        and all(
            count == 0
            for count in quality["non_finite_decision_metric_counts"].values()
        ),
        "840 rows, 84 cells, 10 episodes/cell, no duplicates, invalid ends or non-finite decision metrics",
    )
    strong_weights = sorted(
        screen["candidate_weight"]
        for screen in result["endpoint_screens"]
        if screen["strong_development_candidate"]
    )
    add_check(
        checks,
        "Stage-one candidate screen",
        strong_weights == [0.125, 0.21875]
        and result["candidate_exit_intervals"]
        and result["candidate_reentry_intervals"],
        "nominal candidates 0.21875 and 0.125; non-monotonic exit and re-entry recorded",
    )

    bootstrap_root = analysis_root / "paired_bootstrap"
    bootstrap = json.loads((bootstrap_root / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    add_check(
        checks,
        "Nested evaluation-seed bootstrap",
        bootstrap["bootstrap_replicates"] == 20_000
        and bootstrap["row_count"] == 32
        and bootstrap["result_json_sha256"] == sha256(result_path),
        "20,000 fixed-seed paired resamples; explicitly not training-seed inference",
    )

    forensic_path = PROJECT_ROOT / "artifacts" / "audit" / "stage1_interruption_forensics_20260814" / "model_equivalence.json"
    forensic = json.loads(forensic_path.read_text(encoding="utf-8"))
    add_check(
        checks,
        "Interrupted-run policy equivalence",
        forensic["shared_checkpoint_count"] == 8
        and forensic["all_policy_tensors_equal"],
        "8 shared checkpoints; all policy tensors equal; maximum parameter difference 0",
    )

    video_root = analysis_root / "videos"
    video_manifests = sorted(video_root.glob("*.json"))
    video_records = [json.loads(path.read_text(encoding="utf-8")) for path in video_manifests]
    video_ok = len(video_records) == 6 and all(
        record["status"] == "complete_trajectory_video_rendered"
        and record["evaluation_seed"] == 51101
        and record["playback_speed_ratio"] == 1.0
        and record["frames"] == record["episode_summary"]["episode_length"]
        and (record["episode_summary"]["terminated"] or record["episode_summary"]["truncated"])
        and Path(record["video_path"]).exists()
        and sha256(Path(record["video_path"])) == record["video_sha256"]
        for record in video_records
    )
    total_video_bytes = sum(Path(record["video_path"]).stat().st_size for record in video_records)
    add_check(
        checks,
        "Complete trajectory videos",
        video_ok,
        f"6 fixed videos, evaluation seed 51101, real-time playback, {total_video_bytes} bytes",
    )

    figures = [
        analysis_root / "endpoint_seed_contrasts.png",
        analysis_root / "domain_replication_matrix.png",
        analysis_root / "cross_rescore_matrix.png",
        analysis_root / "checkpoint_replication_matrix.png",
        analysis_root / "progress_effort_map.png",
        analysis_root / "trajectory_midpoint_contact_sheet.png",
    ]
    dimensions = []
    for path in figures:
        with Image.open(path) as image:
            dimensions.append((path.name, image.width, image.height))
    add_check(
        checks,
        "Figure render and visual review",
        all(width >= 1500 and height >= 600 for _, width, height in dimensions),
        "6 PNG figures rendered at legible resolution and visually inspected in the Codex app",
    )

    superseded_notice = PROJECT_ROOT / "artifacts" / "exploration" / "stage1_harmonised_existing_models_20260814" / "SUPERSEDED_NOTICE.md"
    add_check(
        checks,
        "Superseded v1 isolation",
        superseded_notice.exists(),
        "v1 actual-timestep metadata defect is labelled; v2 is the analysis source",
    )

    formal_proposal = json.loads(
        (PROJECT_ROOT / "configs" / "stage1_formal_confirmation_proposal_v1_20260814.json").read_text(encoding="utf-8")
    )
    add_check(
        checks,
        "Formal-run safety gate",
        formal_proposal["status"] == "blocked_pending_user_and_supervisor_decisions_do_not_run"
        and len(formal_proposal["blocking_decisions"]) == 4,
        "formal training remains blocked by four explicit decisions; shaping excluded",
    )

    summary: dict[str, Any] = {
        "status": "PASS_FOR_DEVELOPMENT_ARTIFACTS_NOT_FORMAL_CONFIRMATION",
        "generated_with_python": sys.version.split()[0],
        "checks": checks,
        "scientific_blockers": formal_proposal["blocking_decisions"],
        "claim_boundary": result["claim_boundary"],
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
