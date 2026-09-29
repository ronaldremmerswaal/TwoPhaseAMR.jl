#!/usr/bin/env python3
"""Export a small, reviewable Stage 2 figure/summary snapshot for the docs.

Run after benchmark/scaling/report_final.py has audited the frozen acquisition:

    python3 docs/export_stage2_assets.py \
        benchmark/scaling/results/20260929-stage2-final-baseline110-pinned-spread

The raw machine-local samples stay in the ignored results directory. This
script copies only selected aggregate SVGs and selected aggregate JSON rows.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "docs" / "src" / "assets"
SIZES = {512, 2048, 4096}
WORKERS = {1, 4, 8}
FIGURES = (
    "stage2-final-512.svg",
    "stage2-final-2048.svg",
    "stage2-final-4096.svg",
    "stage2-final-throughput-working-set.svg",
    "stage2-final-memory-allocation.svg",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    raw_path = result_dir / "raw.jsonl"
    report_path = result_dir / "report-corrected.json"
    report = json.loads(report_path.read_text())
    raw_hash = sha256(raw_path)
    if report["raw_jsonl_sha256"] != raw_hash:
        raise ValueError("report does not match the frozen raw samples")
    if report["input_samples"] != 280:
        raise ValueError("expected the completed 280-sample Stage 2 matrix")
    rows = [
        row for row in report["rows"]
        if row["size"] in SIZES and row["requested_workers"] in WORKERS
    ]
    if not rows:
        raise ValueError("no mandatory-size rows in audited report")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    figure_hashes = {}
    for name in FIGURES:
        source = result_dir / name
        target = DESTINATION / name
        shutil.copy2(source, target)
        figure_hashes[name] = sha256(target)
    export = {
        "schema_version": 1,
        "provenance": {
            "machine_local_artifact": str(result_dir.relative_to(ROOT)),
            "raw_jsonl_sha256": raw_hash,
            "report_corrected_sha256": sha256(report_path),
            "report_script_sha256": report["report_script_sha256"],
            "configuration_sha256": report["config_sha256"],
            "figure_sha256": figure_hashes,
            "generation_command": (
                "python3 benchmark/scaling/report_final.py "
                + str(result_dir.relative_to(ROOT))
                + " && python3 docs/export_stage2_assets.py "
                + str(result_dir.relative_to(ROOT))
            ),
            "context": "Ryzen 9 5950X, Julia 1.10.12, 2026-09-29 pinned spread",
        },
        "selection": "Audited aggregate rows at 512/2048/4096 squared and p1/p4/p8; no raw samples",
        "rows": rows,
    }
    output = DESTINATION / "stage2-primary-summary.json"
    output.write_text(json.dumps(export, indent=2) + "\n")
    print(f"Exported {len(rows)} aggregate rows and {len(FIGURES)} SVGs to {DESTINATION}")


if __name__ == "__main__":
    main()
