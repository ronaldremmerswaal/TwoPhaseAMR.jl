#!/usr/bin/env python3
"""Five-repeat strong-scaling evidence for frozen standard and affine LW scopes."""
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time
import tomllib

from run import ROOT, command_output, machine_metadata, placement_mask, project_revision, source_hash, topology, toy_tree_hash
from screen_layouts import EXPERIMENTS, SCRIPT, SOURCES, source_digest


INITIAL_STEPS = {128: 4096, 256: 4096, 512: 1024, 1024: 256, 2048: 64, 4096: 16}
SIZES = [128, 256, 512, 1024, 2048, 4096]
THREADS = [1, 2, 4, 8, 16]


def runner_digest():
    digest = hashlib.sha256()
    for path in (Path(__file__), *SOURCES):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def bootstrap_ratio(base, current, seed, count=20_000):
    if base is current:
        return [1.0, 1.0]
    rng = random.Random(seed)
    ratios = []
    for _ in range(count):
        a = statistics.median(rng.choices(base, k=len(base)))
        b = statistics.median(rng.choices(current, k=len(current)))
        ratios.append(a / b)
    ratios.sort()
    return [ratios[int(0.025 * (count - 1))], ratios[int(0.975 * (count - 1))]]


def jobs_for_final():
    jobs = []
    for n in SIZES:
        jobs.extend((n, "standard", (1024, 32), p) for p in THREADS)
        if n >= 2048:
            for variant in ("fused_affine", "fused_amortized"):
                jobs.extend((n, variant, (512, 8), p) for p in THREADS)
        if n in (512, 2048, 4096):
            jobs.append((n, "halo", (1024, 32), 1))
            jobs.append((n, "fused_amortized", (1024, 32), 1))
    return jobs


def summarize(records, seed):
    grouped = collections.defaultdict(list)
    for row in records:
        grouped[(row["nx"], row["variant"], tuple(row["tile"]), row["workers"])].append(row)
    result = []
    for (n, variant, tile, p), rows in sorted(grouped.items()):
        times = [r["time_seconds"] for r in rows]
        median = statistics.median(times)
        base = grouped.get((n, variant, tile, 1), [])
        base_times = [r["time_seconds"] for r in base]
        speedup = statistics.median(base_times) / median if base_times else None
        ci = ([1.0, 1.0] if p == 1 else bootstrap_ratio(base_times, times, seed + n + p + sum(map(ord, variant)))) if base_times else None
        tile_capacity = min(p, math.ceil(n / tile[0]) * math.ceil(n / tile[1]))
        result.append({"nx": n, "ny": n, "variant": variant, "tile": list(tile), "workers": p,
                       "api_scope": "amortized_trajectory" if variant == "fused_amortized" else "observable_step",
                       "samples": len(rows), "steps": rows[0]["steps"], "seconds_median": median,
                       "seconds_min": min(times), "seconds_max": max(times),
                       "seconds_mad": statistics.median(abs(t - median) for t in times),
                       "self_speedup": speedup, "self_speedup_ci95": ci,
                       "self_efficiency": speedup / p if speedup is not None else None,
                       "self_efficiency_ci95": [bound / p for bound in ci] if ci else None,
                       "active_workers_min": min(r["active_workers"] for r in rows),
                       "active_workers_min_pass": min(r["active_workers_min_pass"] for r in rows),
                       "copyback_workers_min": min(r["copyback_active_workers"] for r in rows),
                       "active_worker_scope": rows[0]["active_worker_scope"],
                       "tile_worker_capacity": tile_capacity,
                       "mass_error_max": max(r["mass_error"] for r in rows),
                       "state_delta_max": max(r["max_state_delta"] for r in rows),
                       "translation_l2_max": max(r["translation_l2"] for r in rows),
                       "allocation_median_bytes": statistics.median(r["allocation_bytes"] for r in rows),
                       "peak_rss_median_bytes": statistics.median(r["peak_rss_bytes"] for r in rows),
                       "numerical_array_bytes": rows[0]["numerical_array_bytes"]})
    for row in result:
        n, scope = row["nx"], row["api_scope"]
        references = [item for item in result if item["nx"] == n and item["api_scope"] == scope and item["workers"] == 1]
        if not references:
            continue
        fastest = min(references, key=lambda item: item["seconds_median"])
        key = (n, fastest["variant"], tuple(fastest["tile"]), 1)
        current_key = (n, row["variant"], tuple(row["tile"]), row["workers"])
        base_times = [sample["time_seconds"] for sample in grouped[key]]
        current_times = [sample["time_seconds"] for sample in grouped[current_key]]
        ratio = fastest["seconds_median"] / row["seconds_median"]
        ci = [1.0, 1.0] if key == current_key else bootstrap_ratio(base_times, current_times, seed + n + row["workers"] + 314159)
        row.update(fastest_serial_variant=fastest["variant"], fastest_serial_tile=fastest["tile"],
                   fastest_serial_seconds=fastest["seconds_median"],
                   speedup_vs_fastest_serial=ratio, speedup_vs_fastest_serial_ci95=ci,
                   efficiency_vs_fastest_serial=ratio / row["workers"],
                   efficiency_vs_fastest_serial_ci95=[bound / row["workers"] for bound in ci])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default=shutil.which("julia"))
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--depot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0.5, help="target shortest finalist trajectory")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=29092026)
    args = parser.parse_args()
    julia = shutil.which(args.julia) if args.julia else None
    if julia is None:
        parser.error("Julia executable unavailable; pass --julia")
    args.julia = julia
    if args.seconds <= 0 or args.repeats < 1:
        parser.error("seconds and repeats must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    allowed, nodes = topology()
    masks = {p: placement_mask(p, "spread", allowed, nodes) for p in THREADS}
    source_before = source_digest()
    runner_before = runner_digest()
    project_before = source_hash(args.project)
    toy_before = toy_tree_hash()
    version = command_output(julia, "--version")
    if not version.lower().startswith("julia version 1.10."):
        parser.error(f"frozen final script expects Julia 1.10.x; got {version!r}")
    destination = args.output
    destination.mkdir(parents=True)
    (destination / "samples").mkdir()
    for source in (Path(__file__), *SOURCES):
        target = destination / "source-snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ("Project.toml", "Manifest.toml", "baseline.json"):
        path = args.project / name
        if path.exists():
            shutil.copy2(path, destination / name)
    receipt = Path("/tmp/twophase-julia110-release.json")
    if receipt.exists():
        shutil.copy2(receipt, destination / receipt.name)
    config = {"schema_version": 1, "kind": "final_lw_scaling", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "julia": julia, "julia_version": version, "project": str(args.project),
              "project_revision": project_revision(args.project), "project_sha256": project_before,
              "toy_tree_sha256": toy_before, "experiment_sha256": source_before, "runner_sha256": runner_before,
              "depot": str(args.depot) if args.depot else None, "placement": "spread", "pin_workers": True,
              "cpu_masks": masks, "blas_threads": 1, "gc_threads": 1,
              "sizes": SIZES, "threads": THREADS, "repeats": args.repeats, "seconds_target": args.seconds,
              "general_step_policy": ["standard", [1024, 32]],
              "large_grid_candidates": [["fused_affine", [512, 8]], ["fused_amortized", [512, 8]]],
              "crossover_cells": 2048 * 2048, "serial_references": [["halo", [1024, 32]], ["fused_amortized", [1024, 32]]],
              "caveat": "amortized trajectory is not an individually observable step API",
              "machine": machine_metadata(allowed, nodes), "seed": args.seed}
    config["configuration_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (destination / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    raw = destination / "raw.jsonl"
    raw.touch(exist_ok=False)
    pilot_raw = destination / "pilots.jsonl"
    pilot_raw.touch(exist_ok=False)
    workload_hash = None

    def check_source():
        if source_digest() != source_before or runner_digest() != runner_before or source_hash(args.project) != project_before or toy_tree_hash() != toy_before:
            raise RuntimeError("project, toy, experiment, or runner source changed during frozen measurements")

    def invoke(n, variant, tile, p, steps, kind, repeat):
        check_source()
        mask = masks[p]
        sample_path = destination / "samples" / f"{kind}-r{repeat}-{variant}-{tile[0]}x{tile[1]}-{n}-p{p}.toml"
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", JULIA_NUM_GC_THREADS="1")
        if args.depot is not None:
            env["JULIA_DEPOT_PATH"] = str(args.depot) + os.pathsep + str(Path.home() / ".julia")
        command = [julia, "--startup-file=no", f"--project={args.project}", f"--threads={p}",
                   "--gcthreads=1", str(SCRIPT), f"variant={variant}", f"nx={n}", f"ny={n}",
                   f"steps={steps}", "backend=polyester", f"tile-x={tile[0]}", f"tile-y={tile[1]}",
                   "threshold=16384", "cpus=" + ",".join(map(str, mask)), f"output={sample_path}"]
        def set_affinity():
            os.sched_setaffinity(0, mask)
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, env=env, preexec_fn=set_affinity)
        elapsed = time.perf_counter() - started
        if process.returncode:
            raise RuntimeError(f"sample failed: {' '.join(command)}\n{process.stdout}\n{process.stderr}")
        check_source()
        sample = tomllib.loads(sample_path.read_text())
        required = ("time_seconds", "max_state_delta", "mass_error", "translation_l2",
                    "checksum", "dt", "final_time", "observed_final_time", "active_worker_ids",
                    "compute_pass_worker_ids", "copyback_worker_ids", "active_worker_scope",
                    "allocation_bytes", "peak_rss_bytes", "numerical_array_bytes")
        missing = [name for name in required if name not in sample]
        if missing:
            raise RuntimeError(f"sample missing required fields: {missing}")
        for name in ("time_seconds", "max_state_delta", "mass_error", "translation_l2",
                     "checksum", "dt", "final_time", "observed_final_time"):
            if not math.isfinite(sample[name]):
                raise RuntimeError(f"nonfinite sample field {name}")
        if sample["variant"] != variant or sample["nx"] != n or sample["ny"] != n or sample["steps"] != steps or sample["workers"] != p or sample["tile"] != list(tile):
            raise RuntimeError("sample does not match frozen job")
        if sample["max_state_delta"] > 1e-12 or sample["mass_error"] > 1e-12:
            raise RuntimeError("sample numerical parity/conservation failed")
        if len(sample["affinity"]) != p or [a["cpu"] for a in sample["affinity"]] != mask:
            raise RuntimeError("sample worker pinning mismatch")
        actual = len(sample["active_worker_ids"])
        pass_counts = [len(ids) for ids in sample["compute_pass_worker_ids"]]
        min_pass = min(pass_counts) if pass_counts else 0
        copy_count = len(sample["copyback_worker_ids"])
        sample["active_workers"] = actual
        sample["active_workers_min_pass"] = min_pass
        sample["copyback_active_workers"] = copy_count
        capacity = min(p, math.ceil(n / tile[0]) * math.ceil(n / tile[1]))
        sample["requested_workers"] = p
        sample["tile_worker_capacity"] = capacity
        sample["underoccupied_by_tile_count"] = capacity < p
        if variant == "halo" and p == 1 and sample["active_worker_scope"] == "halo prototype not instrumented":
            pass # Explicitly uninstrumented serial reference, never a scaling gate.
        elif not 1 <= min_pass <= actual <= capacity:
            raise RuntimeError(f"invalid kernel participation {actual}; requested {p}, tile capacity {capacity}")
        if n in (512, 1024, 2048, 4096) and p in (4, 8) and min_pass != p:
            raise RuntimeError(f"primary-gate per-pass kernel participation {min_pass} differs from requested {p}")
        if sample["time_seconds"] <= 0 or not math.isfinite(sample["time_seconds"]):
            raise RuntimeError("invalid sample time")
        sample.update(kind=kind, repeat=repeat, process_elapsed_seconds=elapsed,
                      project_sha256=project_before, toy_tree_sha256=toy_before,
                      experiment_sha256=source_before, runner_sha256=runner_before,
                      configuration_sha256=config["configuration_sha256"],
                      workload_sha256=workload_hash,
                      timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        with (pilot_raw if kind == "pilot" else raw).open("a") as stream:
            stream.write(json.dumps(sample) + "\n")
        return sample

    workloads = {}
    for n in SIZES:
        pilots = [("standard", (1024, 32), 1), ("standard", (1024, 32), 8),
                  ("standard", (1024, 32), 16)]
        if n >= 2048:
            pilots += [("fused_amortized", (512, 8), 8), ("fused_amortized", (512, 8), 16)]
        estimates = []
        for variant, tile, p in pilots:
            sample = invoke(n, variant, tile, p, INITIAL_STEPS[n], "pilot", 0)
            estimates.append((sample["time_seconds"] / INITIAL_STEPS[n], variant, tile, p))
            print(f"pilot {n}² {variant} p{p}: {sample['time_seconds']:.6f}s/{INITIAL_STEPS[n]} steps", flush=True)
        fastest = min(estimates)
        workloads[str(n)] = math.ceil(args.seconds / fastest[0])
        for margin in range(3):
            check = invoke(n, fastest[1], fastest[2], fastest[3], workloads[str(n)], "pilot", margin + 1)
            if check["time_seconds"] >= 0.35:
                break
            workloads[str(n)] = math.ceil(workloads[str(n)] * (0.55 / check["time_seconds"]))
        else:
            raise RuntimeError(f"unable to freeze a meaningful {n}² trajectory after three margin checks")
        print(f"frozen {n}²: {workloads[str(n)]} steps; fastest pilot estimate {fastest[0]:.4g}s/step", flush=True)
    (destination / "workload.json").write_text(json.dumps(workloads, indent=2) + "\n")
    workload_hash = hashlib.sha256((destination / "workload.json").read_bytes()).hexdigest()

    all_jobs = jobs_for_final()
    rng = random.Random(args.seed)
    records = []
    for repeat in range(args.repeats):
        jobs = list(all_jobs)
        rng.shuffle(jobs)
        for n, variant, tile, p in jobs:
            sample = invoke(n, variant, tile, p, workloads[str(n)], "final", repeat)
            records.append(sample)
            print(f"repeat {repeat + 1}/{args.repeats} {n}² {variant} {tile} p{p}: {sample['time_seconds']:.6f}s; active {sample['active_workers']}", flush=True)
            (destination / "summary.json").write_text(json.dumps(summarize(records, args.seed), indent=2) + "\n")
    print(f"Completed {len(records)} final samples in {destination}")


if __name__ == "__main__":
    main()
