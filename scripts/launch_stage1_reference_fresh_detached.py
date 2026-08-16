"""Launch the frozen V6 runner independently of the Codex tool host."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "exploration"
    / "stage1_reference_fresh_1m_v6_20260814"
)


def main() -> None:
    if sys.platform != "win32":
        raise RuntimeError("This detached launcher is intentionally Windows-only")
    if not (RUN_ROOT / "run_config.json").exists():
        raise FileNotFoundError("The V6 run must be initialised before launch")
    if (RUN_ROOT / "runs").exists():
        raise FileExistsError("Refusing to launch into an existing runs directory")

    stdout_path = RUN_ROOT / "execution_stdout.log"
    stderr_path = RUN_ROOT / "execution_stderr.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "resume_development_parallel.py"),
        "--run_root",
        str(RUN_ROOT),
        "--max_workers",
        "4",
    ]
    creation_flags = (
        subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.CREATE_NO_WINDOW
    )
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
            close_fds=True,
        )

    time.sleep(2.0)
    return_code = process.poll()
    record = {
        "status": "started" if return_code is None else "exited_during_launch_check",
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": process.pid,
        "return_code_after_two_seconds": return_code,
        "python_executable": sys.executable,
        "command": command,
        "working_directory": str(PROJECT_ROOT),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "creation_flags": [
            "DETACHED_PROCESS",
            "CREATE_NEW_PROCESS_GROUP",
            "CREATE_NO_WINDOW",
        ],
        "scientific_configuration_change": "none",
    }
    (RUN_ROOT / "background_process.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2), flush=True)
    if return_code is not None:
        raise RuntimeError(f"Detached runner exited immediately with {return_code}")


if __name__ == "__main__":
    main()
