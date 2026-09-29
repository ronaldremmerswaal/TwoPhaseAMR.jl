#!/usr/bin/env python3
"""Bounded, fresh-process screen of experimental LW storage layouts."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import tomllib

from run import ROOT, command_output, placement_mask, project_revision, source_hash, topology, toy_tree_hash


EXPERIMENTS = ROOT / "research" / "UniformLWAdvection" / "experiments"
SCRIPT = EXPERIMENTS / "screen_layouts.jl"
SOURCES = [Path(__file__), Path(__file__).with_name("run.py"), SCRIPT,
           EXPERIMENTS / "halo.jl", EXPERIMENTS / "fused.jl", EXPERIMENTS / "fused_affine.jl",
           EXPERIMENTS / "fused_affine_polyester.jl",
           ROOT / "research" / "UniformLWAdvection" / "src" / "UniformLWAdvection.jl",
           ROOT / "research" / "UniformLWAdvection" / "ext" / "UniformLWAdvectionPolyesterExt.jl",
           ROOT / "research" / "UniformLWAdvection" / "Project.toml"]


def source_digest():
    digest = hashlib.sha256()
    for source in SOURCES:
        digest.update(str(source.relative_to(ROOT)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default=shutil.which("julia"))
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--depot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=["standard", "halo", "fused", "fused_affine", "fused_amortized"],
                        default=["standard", "halo", "fused"])
    parser.add_argument("--tiles", nargs="+", default=["1024x32"], help="fixed tile shapes, e.g. 128x8 512x8")
    parser.add_argument("--seed", type=int, default=29092026)
    args = parser.parse_args()
    try:
        tiles = [tuple(map(int, value.lower().split("x"))) for value in args.tiles]
    except ValueError:
        parser.error("tiles must be positive NxM shapes")
    if any(len(tile) != 2 or min(tile) < 1 for tile in tiles):
        parser.error("tiles must be positive NxM shapes")
    julia = shutil.which(args.julia) if args.julia else None
    if julia is None:
        parser.error("Julia executable unavailable; pass --julia")
    args.julia = julia
    version = command_output(julia, "--version")
    if not version.lower().startswith("julia version 1.10."):
        parser.error(f"screen currently freezes Julia 1.10 syntax and baseline; got {version!r}")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    allowed, nodes = topology()
    masks = {p: placement_mask(p, "spread", allowed, nodes) for p in (1, 4, 8)}
    baseline_hash = source_hash(args.project)
    experiment_hash = source_digest()
    args.output.mkdir(parents=True)
    (args.output / "samples").mkdir()
    (args.output / "source-snapshot").mkdir()
    for source in SOURCES:
        shutil.copy2(source, args.output / "source-snapshot" / source.name)
    for name in ("Project.toml", "Manifest.toml", "baseline.json"):
        source = args.project / name
        if source.exists():
            shutil.copy2(source, args.output / name)
    receipt = Path("/tmp/twophase-julia110-release.json")
    if receipt.exists():
        shutil.copy2(receipt, args.output / receipt.name)
    config = {"schema_version": 1, "kind": "bounded_layout_screen",
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "project": str(args.project), "project_revision": project_revision(args.project),
              "project_sha256": baseline_hash, "toy_tree_sha256": toy_tree_hash(), "experiment_sha256": experiment_hash,
              "julia": julia, "julia_version": version, "depot": str(args.depot) if args.depot else None,
              "variants": args.variants, "sizes": [512, 4096],
              "steps": {"512": 1024, "4096": 16}, "workers": [1, 4, 8], "repeats": 2,
              "backend": "polyester", "flux": "recompute", "tiles": tiles,
              "threshold": 16384, "placement": "spread", "pinning": "per-worker Linux affinity",
              "cpu_masks": masks, "seed": args.seed}
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    jobs = [(variant, tile, n, p, repeat) for repeat in range(2)
            for variant in config["variants"] for tile in tiles for n in config["sizes"] for p in config["workers"]]
    rng = random.Random(args.seed)
    jobs_per_repeat = len(args.variants) * len(tiles) * len(config["sizes"]) * len(config["workers"])
    for repeat in range(2):
        subset = jobs[repeat * jobs_per_repeat:(repeat + 1) * jobs_per_repeat]
        rng.shuffle(subset)
        jobs[repeat * jobs_per_repeat:(repeat + 1) * jobs_per_repeat] = subset
    output = args.output / "raw.jsonl"
    output.touch(exist_ok=False)
    for index, (variant, tile, n, p, repeat) in enumerate(jobs, start=1):
        if source_digest() != experiment_hash or source_hash(args.project) != baseline_hash:
            raise RuntimeError("source changed during the frozen layout screen")
        sample_path = args.output / "samples" / f"r{repeat}-{variant}-{tile[0]}x{tile[1]}-{n}-p{p}.toml"
        mask = masks[p]
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", JULIA_NUM_GC_THREADS="1")
        if args.depot is not None:
            env["JULIA_DEPOT_PATH"] = str(args.depot) + os.pathsep + str(Path.home() / ".julia")
        command = [julia, "--startup-file=no", f"--project={args.project}", f"--threads={p}",
                   "--gcthreads=1", str(SCRIPT), f"variant={variant}", f"nx={n}", f"ny={n}",
                   f"steps={config['steps'][str(n)]}", "backend=polyester", f"tile-x={tile[0]}", f"tile-y={tile[1]}",
                   "threshold=16384", "cpus=" + ",".join(map(str, mask)), f"output={sample_path}"]
        def set_affinity():
            os.sched_setaffinity(0, mask)
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, env=env, preexec_fn=set_affinity)
        process_elapsed = time.perf_counter() - started
        if process.returncode:
            raise RuntimeError(f"sample {index}/{len(jobs)} failed: {command}\n{process.stdout}\n{process.stderr}")
        sample = tomllib.loads(sample_path.read_text())
        if sample["variant"] != variant or sample["nx"] != n or sample["steps"] != config["steps"][str(n)] or sample["workers"] != p or sample["tile"] != list(tile):
            raise RuntimeError("screen sample does not match frozen job")
        if sample["max_state_delta"] > 1e-12 or sample["mass_error"] > 1e-12:
            raise RuntimeError("layout screen numerical check failed")
        if len(sample["affinity"]) != p or [a["cpu"] for a in sample["affinity"]] != mask:
            raise RuntimeError("screen sample worker pinning mismatch")
        record = {**sample, "repeat": repeat, "process_elapsed_seconds": process_elapsed,
                  "project_sha256": baseline_hash, "experiment_sha256": experiment_hash,
                  "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        with output.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"{index}/{len(jobs)} {variant} tile{tile[0]}x{tile[1]} {n}² p{p} r{repeat + 1}: {sample['time_seconds']:.6f}s, "
              f"delta={sample['max_state_delta']:.3g}, mass={sample['mass_error']:.3g}", flush=True)
    print(f"Completed {len(jobs)} independent exploratory layout samples in {args.output}")


if __name__ == "__main__":
    main()
