#!/usr/bin/env python3
"""Separate p24/p32 SMT diagnostic using the frozen Stage 2 workloads."""
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

from run import ROOT, command_output, placement_mask, source_hash, topology, toy_tree_hash
from screen_layouts import SCRIPT, SOURCES, source_digest


def ratio_ci(base, current, seed, iterations=20_000):
    rng = random.Random(seed)
    ratios = []
    for _ in range(iterations):
        ratios.append(statistics.median(rng.choices(base, k=len(base))) /
                      statistics.median(rng.choices(current, k=len(current))))
    ratios.sort()
    return [ratios[int(.025 * (iterations - 1))], ratios[int(.975 * (iterations - 1))]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--julia", default=shutil.which("julia"))
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--depot", type=Path)
    parser.add_argument("--seed", type=int, default=29092026)
    args = parser.parse_args()
    julia = shutil.which(args.julia) if args.julia else None
    if julia is None or args.output.exists():
        parser.error("Julia unavailable or output directory already exists")
    final_config = json.loads((args.final / "configuration.json").read_text())
    workload = json.loads((args.final / "workload.json").read_text())
    final_rows = [json.loads(line) for line in (args.final / "raw.jsonl").read_text().splitlines()]
    if len(final_rows) != 280:
        parser.error("final Stage 2 reference must have all 280 samples")
    if source_digest() != final_config["experiment_sha256"] or source_hash(args.project) != final_config["project_sha256"] or toy_tree_hash() != final_config["toy_tree_sha256"]:
        parser.error("source differs from frozen final matrix")
    allowed, nodes = topology()
    masks = {p: placement_mask(p, "spread", allowed, nodes) for p in (24, 32)}
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = {"schema_version": 1, "kind": "stage2_smt_diagnostic", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "julia": julia, "version": command_output(julia, "--version"), "project": str(args.project),
              "project_sha256": final_config["project_sha256"], "toy_tree_sha256": final_config["toy_tree_sha256"],
              "experiment_sha256": final_config["experiment_sha256"], "runner_sha256": script_hash,
              "final_artifact": str(args.final), "final_config_sha256": final_config["configuration_sha256"],
              "placement": "spread, physical cores first, then SMT siblings", "cpu_masks": masks,
              "repeats": 5, "workers": [24, 32], "seed": args.seed,
              "coverage": "standard1024, standard2048, standard4096 only; affine SMT excluded after intermittent p24 numerical parity failure; 512 excluded because standard has16 tiles",
              "excluded_affine_smt_artifact": "benchmark/scaling/results/20260929-stage2-smt-baseline110-pinned-spread/failure.json"}
    args.output.mkdir(parents=True)
    (args.output / "samples").mkdir()
    for source in (Path(__file__), *SOURCES):
        target = args.output / "source-snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ("Project.toml", "Manifest.toml", "baseline.json"):
        path = args.project / name
        if path.exists():
            shutil.copy2(path, args.output / name)
    receipt = Path("/tmp/twophase-julia110-release.json")
    if receipt.exists():
        shutil.copy2(receipt, args.output / receipt.name)
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    jobs = [(n, "standard", (1024, 32), p) for n in (1024, 2048, 4096) for p in (24, 32)]
    if len(jobs) != 6:
        raise AssertionError("standard-only SMT matrix must have6 jobs per repeat")
    raw_path = args.output / "raw.jsonl"
    raw_path.touch(exist_ok=False)
    rng = random.Random(args.seed)
    for repeat in range(5):
        batch = list(jobs)
        rng.shuffle(batch)
        for n, variant, tile, p in batch:
            if source_digest() != config["experiment_sha256"] or source_hash(args.project) != config["project_sha256"] or toy_tree_hash() != config["toy_tree_sha256"]:
                raise RuntimeError("source changed during SMT sampling")
            mask = masks[p]
            sample_path = args.output / "samples" / f"r{repeat}-{n}-{variant}-{tile[0]}x{tile[1]}-p{p}.toml"
            env = os.environ.copy()
            env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", JULIA_NUM_GC_THREADS="1")
            if args.depot is not None:
                env["JULIA_DEPOT_PATH"] = str(args.depot) + os.pathsep + str(Path.home() / ".julia")
            command = [julia, "--startup-file=no", f"--project={args.project}", f"--threads={p}", "--gcthreads=1", str(SCRIPT),
                       f"variant={variant}", f"nx={n}", f"ny={n}", f"steps={workload[str(n)]}", "backend=polyester",
                       f"tile-x={tile[0]}", f"tile-y={tile[1]}", "threshold=16384",
                       "cpus=" + ",".join(map(str, mask)), f"output={sample_path}"]
            def set_affinity():
                os.sched_setaffinity(0, mask)
            started = time.perf_counter()
            process = subprocess.run(command, capture_output=True, text=True, env=env, preexec_fn=set_affinity)
            elapsed = time.perf_counter() - started
            if process.returncode:
                raise RuntimeError(f"SMT sample failed: {' '.join(command)}\n{process.stdout}\n{process.stderr}")
            sample = tomllib.loads(sample_path.read_text())
            if sample["variant"] != variant or sample["workers"] != p or sample["steps"] != workload[str(n)] or sample["tile"] != list(tile):
                raise RuntimeError("SMT sample differs from frozen job")
            if sample["max_state_delta"] != 0 or sample["probe_state_delta"] != 0 or sample["mass_error"] > 512 * 2.220446049250313e-16:
                raise RuntimeError("SMT numerical parity or conservation failed")
            pass_counts = [len(ids) for ids in sample["compute_pass_worker_ids"]]
            actual = min(pass_counts)
            capacity = min(p, ((n + tile[0] - 1) // tile[0]) * ((n + tile[1] - 1) // tile[1]))
            if not 1 <= actual <= len(sample["active_worker_ids"]) <= capacity:
                raise RuntimeError(f"invalid SMT kernel participation {actual}; requested {p}, tile capacity {capacity}")
            if [entry["cpu"] for entry in sample["affinity"]] != mask:
                raise RuntimeError("SMT pinning mismatch")
            sample.update(repeat=repeat, requested_workers=p, actual_workers_min_pass=actual,
                          tile_worker_capacity=capacity, underoccupied=actual < p,
                          process_elapsed_seconds=elapsed, project_sha256=config["project_sha256"],
                          toy_tree_sha256=config["toy_tree_sha256"], experiment_sha256=config["experiment_sha256"],
                          runner_sha256=script_hash, timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            with raw_path.open("a") as stream:
                stream.write(json.dumps(sample) + "\n")
            print(f"SMT {repeat + 1}/5 {n}² {variant} p{p}: {sample['time_seconds']:.6f}s; actual/pass {actual}", flush=True)
    final_grouped = collections.defaultdict(list)
    for row in final_rows:
        final_grouped[(row["nx"], row["variant"], tuple(row["tile"]), row["workers"])].append(row["time_seconds"])
    smt_rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    smt_grouped = collections.defaultdict(list)
    for row in smt_rows:
        smt_grouped[(row["nx"], row["variant"], tuple(row["tile"]), row["workers"])].append(row["time_seconds"])
    summary = []
    for (n, variant, tile, p), times in sorted(smt_grouped.items()):
        base = final_grouped[(n, variant, tile, 1)]
        sixteen = final_grouped[(n, variant, tile, 16)]
        median = statistics.median(times)
        s = statistics.median(base) / median
        ratio = statistics.median(sixteen) / median
        summary.append({"size": n, "variant": variant, "tile": list(tile), "requested_workers": p,
                        "actual_workers_min_pass": min(row["actual_workers_min_pass"] for row in smt_rows
                                                       if row["nx"] == n and row["variant"] == variant and tuple(row["tile"]) == tile and row["workers"] == p),
                        "samples": len(times), "time_median_seconds": median,
                        "time_min_seconds": min(times), "S_p": s, "S_p_ci95": ratio_ci(base, times, args.seed + n + p),
                        "E_p_requested": s / p, "physical_core_normalized_S_p_over_16": s / 16,
                        "T16_over_Tp": ratio, "T16_over_Tp_ci95": ratio_ci(sixteen, times, args.seed + n + p + 16),
                        "comparison_window": "p1/p16 from earlier final matrix; p24/p32 from separate SMT acquisition"})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Completed {len(smt_rows)} SMT samples in {args.output}")


if __name__ == "__main__":
    main()
