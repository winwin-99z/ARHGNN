"""Create a local, privacy-preserving audit manifest for a completed run."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PATIENT_LEVEL_FILENAMES = {
    "predictions_primary_test.csv",
    "predictions_external.csv",
}


def write_results_manifest(
    output_dir: str | Path,
    config_path: str | Path,
    config: dict[str, Any],
    label_names: list[str],
    cohort_metadata: dict[str, Any],
    seed: int,
) -> Path:
    """Write hashes and provenance without copying raw clinical data.

    The manifest is intended for controlled-access archiving. Patient-level
    predictions remain restricted even though their hashes can be recorded.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    config_snapshot = output_dir / "final_configuration.yaml"
    shutil.copy2(config_path, config_snapshot)

    environment_path = output_dir / "environment.json"
    environment_path.write_text(
        json.dumps(_environment(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    file_records = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "results_manifest.json":
            continue
        file_records.append(
            {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "access": "restricted" if path.name in PATIENT_LEVEL_FILENAMES else "controlled",
            }
        )

    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "access_classification": "controlled-access research output; no raw clinical data are included",
        "random_seed": int(seed),
        "label_order": list(label_names),
        "threshold": float(config.get("evaluation", {}).get("threshold", 0.5)),
        "configuration": {
            "source_file": str(config_path),
            "sha256": _sha256(config_snapshot),
            "model": config.get("model", {}),
            "training": config.get("training", {}),
            "evaluation": config.get("evaluation", {}),
        },
        "cohorts": cohort_metadata,
        "files": file_records,
        "publication_note": (
            "Do not publish patient-level labels, probabilities or thresholded predictions. "
            "Deposit these files only through the approved controlled-access process."
        ),
    }
    manifest_path = output_dir / "results_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment() -> dict[str, str]:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for package in ("numpy", "pandas", "torch", "matplotlib", "yaml"):
        try:
            module = __import__(package)
            result[package] = str(getattr(module, "__version__", "installed"))
        except ImportError:
            result[package] = "not installed"
    return result
