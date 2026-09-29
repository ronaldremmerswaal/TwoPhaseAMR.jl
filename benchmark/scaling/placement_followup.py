#!/usr/bin/env python3
"""Bounded p4/p8 ordinary, compact, and spread placement comparison."""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time
import tomllib

from run import ROOT, DRIVER, command_output, placement_mask, project_revision, source_hash, topology, toy_tree_hash


def digest():
    result = hashlib.sha256()
    for path in (Path(__file__), DRIVER, Path(__file__).with_name("run.py")):
        result.update(path.name.encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def ratio_ci(base, current, seed, iterations=20_000):
    rng = random.Random(seed)
    ratios = [statistics.median(rng.choices(base, k=len(base))) /
              statistics.median(rng.choices(current, k=len(current))) for _ in range(iterations)]
    ratios.sort()
    return [ratios[int(.025 * (iterations - 1))], ratios[int(.975 * (iterations - 1))]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default=shutil.which("julia"))
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--depot", type=Path)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=29092026)
    args = parser.parse_args()
    julia = shutil.which(args.julia) if args.julia else None
    if julia is None or args.output.exists():
        parser.error("Julia unavailable or output already exists")
    final_config = json.loads((args.final / "configuration.json").read_text())
    workload = json.loads((args.final / "workload.json").read_text())
    steps = workload["1024"]
    if steps != 2198:
        parser.error("expected frozen 1024² workload of 2198 steps")
    if source_hash(args.project) != final_config["project_sha256"] or toy_tree_hash() != final_config["toy_tree_sha256"]:
        parser.error("project or toy source differs from final matrix")
    runner_hash = digest()
    allowed, nodes = topology()
    masks = {(mode, p): placement_mask(p, mode, allowed, nodes)
             for mode in ("ordinary", "compact", "spread") for p in (4, 8)}
    config = {"schema_version": 1, "kind": "stage2_placement_diagnostic",
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "julia": julia, "version": command_output(julia, "--version"), "project": str(args.project),
              "project_revision": project_revision(args.project), "project_sha256": final_config["project_sha256"],
              "toy_tree_sha256": final_config["toy_tree_sha256"], "runner_sha256": runner_hash,
              "final_artifact": str(args.final), "size": 1024, "steps": steps,
              "backend": "polyester", "flux": "recompute", "tile": [1024, 32], "threshold": 16384,
              "workers": [4, 8], "placements": ["ordinary", "compact", "spread"], "repeats": 5,
              "ordinary_policy": "unrestricted inherited process affinity, no individual worker pinning",
              "compact_spread_policy": "process mask plus one distinct pinned CPU per Julia worker",
              "cpu_masks": {f"{mode}-p{p}": mask for (mode, p), mask in masks.items()}, "seed": args.seed}
    args.output.mkdir(parents=True)
    (args.output / "samples").mkdir()
    (args.output / "source-snapshot").mkdir()
    for path in (Path(__file__), DRIVER, Path(__file__).with_name("run.py")):
        shutil.copy2(path, args.output / "source-snapshot" / path.name)
    toy_root = ROOT / "research" / "UniformLWAdvection"
    for part in ("src", "ext"):
        shutil.copytree(toy_root / part, args.output / "source-snapshot" / "UniformLWAdvection" / part)
    shutil.copy2(toy_root / "Project.toml", args.output / "source-snapshot" / "UniformLWAdvection" / "Project.toml")
    for name in ("Project.toml", "Manifest.toml", "baseline.json"):
        path = args.project / name
        if path.exists():
            shutil.copy2(path, args.output / name)
    receipt = Path("/tmp/twophase-julia110-release.json")
    if receipt.exists():
        shutil.copy2(receipt, args.output / receipt.name)
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    raw_path = args.output / "raw.jsonl"
    raw_path.touch(exist_ok=False)
    rng = random.Random(args.seed)
    for repeat in range(5):
        jobs = [(mode, p) for mode in config["placements"] for p in config["workers"]]
        rng.shuffle(jobs)
        for mode, p in jobs:
            if digest() != runner_hash or source_hash(args.project) != config["project_sha256"] or toy_tree_hash() != config["toy_tree_sha256"]:
                raise RuntimeError("source changed during placement comparison")
            mask = masks[(mode, p)]
            sample_path = args.output / "samples" / f"r{repeat}-{mode}-p{p}.toml"
            env = os.environ.copy()
            env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", JULIA_NUM_GC_THREADS="1")
            if args.depot is not None:
                env["JULIA_DEPOT_PATH"] = str(args.depot) + os.pathsep + str(Path.home() / ".julia")
            command = [julia, "--startup-file=no", f"--project={args.project}", f"--threads={p}",
                       "--gcthreads=1", str(DRIVER), "case=toy", "backend=polyester",
                       "nx=1024", "ny=1024", "tile=32", f"steps={steps}", "seconds=0.5",
                       f"output={sample_path}", "flux=recompute", "tile-x=1024", "threshold=16384"]
            if mode != "ordinary":
                command += ["pin-workers=true", "pin-cpus=" + ",".join(map(str, mask))]
            def set_affinity():
                os.sched_setaffinity(0, mask)
            started = time.perf_counter()
            process = subprocess.run(command, capture_output=True, text=True, env=env,
                                     preexec_fn=None if mode == "ordinary" else set_affinity)
            elapsed = time.perf_counter() - started
            if process.returncode:
                raise RuntimeError(f"placement sample failed: {' '.join(command)}\n{process.stdout}\n{process.stderr}")
            sample = tomllib.loads(sample_path.read_text())
            if sample["steps"] != steps or sample["active_workers"] > p or sample["diagnostics"]["mass_error"] > 1e-10:
                raise RuntimeError("placement sample failed workload, worker, or numerical check")
            sample.update(repeat=repeat, placement=mode, requested_workers=p,
                          process_elapsed_seconds=elapsed, runner_sha256=runner_hash,
                          timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            with raw_path.open("a") as stream:
                stream.write(json.dumps(sample) + "\n")
            print(f"placement {repeat + 1}/5 {mode} p{p}: {sample['time_seconds']:.6f}s; actual {sample['active_workers']}", flush=True)
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    grouped = collections.defaultdict(list)
    for row in raw:
        grouped[(row["placement"], row["requested_workers"])].append(row)
    summary = []
    for (mode, p), rows in sorted(grouped.items()):
        times = [row["time_seconds"] for row in rows]
        spread = [row["time_seconds"] for row in grouped[("spread", p)]]
        ratio = statistics.median(spread) / statistics.median(times)
        ci = [1.0, 1.0] if mode == "spread" else ratio_ci(spread, times, args.seed + p)
        summary.append({"placement": mode, "requested_workers": p, "actual_workers_min": min(row["active_workers"] for row in rows),
                        "samples": len(rows), "time_median_seconds": statistics.median(times),
                        "time_min_seconds": min(times), "speedup_vs_spread": ratio,
                        "speedup_vs_spread_ci95": ci,
                        "pin_workers": mode != "ordinary"})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Completed {len(raw)} placement samples in {args.output}")


if __name__ == "__main__":
    main()
