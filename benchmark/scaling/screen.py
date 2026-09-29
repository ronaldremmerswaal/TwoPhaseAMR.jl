#!/usr/bin/env python3
"""Bounded, sequential Stage 2 toy policy screen at 1024² (exploratory)."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--depot", type=Path, help="optional writable Julia depot used by run.py")
    parser.add_argument("--julia", default=shutil.which("julia"))
    parser.add_argument("--output", type=Path, help="results directory; defaults to a UTC-timestamped local path")
    args = parser.parse_args()
    if args.julia is None:
        parser.error("Julia binary not found; pass --julia")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / "benchmark/scaling/results" / (stamp + "-toy-screen")
    args.output.mkdir(parents=True, exist_ok=False)
    first = ("threads", "recompute", 128)
    policies = [(backend, flux, tile_x) for backend in ("threads", "spawn", "polyester")
                for flux in ("recompute", "stored") for tile_x in (128, 1024)]
    policies.remove(first)
    random.Random(29092026).shuffle(policies)
    policies = [first] + policies + [("serial", flux, 128) for flux in ("recompute", "stored")]
    (args.output / "matrix.json").write_text(json.dumps({
        "purpose": "bounded exploratory screen; no acceptance claim", "size": [1024, 1024],
        "threads_parallel": [1, 4, 8], "threads_serial": [1], "repeats": 2,
        "fastest_worker_seconds_target": 0.25, "placement": "spread",
        "pin_workers": True, "tile_y": 32, "policies_in_run_order": policies,
        "project": str(args.project), "depot": str(args.depot) if args.depot else None, "julia": args.julia,
    }, indent=2) + "\n")
    workload = None
    for number, (backend, flux, tile_x) in enumerate(policies, 1):
        name = f"{backend}-{flux}-{tile_x}x32"
        destination = args.output / name
        command = [sys.executable, str(RUNNER), "--project", str(args.project),
                   "--julia", args.julia]
        if args.depot is not None:
            command += ["--depot", str(args.depot)]
        command += ["--cases", "toy", "--sizes", "1024", "--threads"]
        command += ["1"] if backend == "serial" else ["1", "4", "8"]
        command += ["--repeats", "2", "--seconds", "0.25", "--placement", "spread",
                    "--pin-workers", "--backend", backend, "--flux", flux,
                    "--tile-x", str(tile_x), "--tile", "32", "--output", str(destination)]
        if workload is not None:
            command += ["--workload", str(workload)]
        print(f"SCREEN {number}/{len(policies)} {name}", flush=True)
        subprocess.run(command, check=True)
        if workload is None:
            workload = destination / "workload.json"
    print(f"Completed {len(policies)} sequential exploratory policies: {args.output}", flush=True)


if __name__ == "__main__":
    main()
