"""Validate whether the prospective v2 protocol may be frozen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from proxygap import protocol_freeze_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = protocol_freeze_status(config)
    result["config"] = str(args.config.resolve())
    result["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest().upper()
    result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if result["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
