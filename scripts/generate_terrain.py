"""Generate, validate and save one configured terrain bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terrain_generator import generate_terrain, load_config, save_terrain_bundle  # noqa: E402
from terrain_validation import assert_terrain_valid, save_validation_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "terrain_development.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "manifests",
    )
    parser.add_argument("--stem", default="terrain_development")
    args = parser.parse_args()

    config = load_config(args.config)
    terrain = generate_terrain(config)
    validation = assert_terrain_valid(terrain)
    paths = save_terrain_bundle(terrain, args.output_dir, args.stem)
    validation_path = args.output_dir / f"{args.stem}_validation.json"
    save_validation_result(validation, validation_path)
    print(
        json.dumps(
            {
                "height_sha256": terrain.height_sha256,
                "normalised_sha256": terrain.normalised_sha256,
                "validation_passed": validation.passed,
                "files": {name: str(path.resolve()) for name, path in paths.items()},
                "validation": str(validation_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
