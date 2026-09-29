#!/usr/bin/env python3
"""Fresh-process strong-scaling runner. Python standard library only."""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).with_name("driver.jl")


def source_hash(project):
    digest = hashlib.sha256()
    files = sorted((project / "src").rglob("*.jl")) if (project / "src").exists() else []
    if (project / "ext").exists():
        files += sorted((project / "ext").rglob("*.jl"))
    files += [project / name for name in ("Project.toml", "Manifest.toml") if (project / name).exists()]
    for path in files:
        digest.update(str(path.relative_to(project)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def toy_tree_hash():
    return source_hash(ROOT / "research" / "UniformLWAdvection")


def runner_hash():
    digest = hashlib.sha256()
    for path in (DRIVER, Path(__file__), Path(__file__).with_name("plot.py"), Path(__file__).with_name("compare.py"), Path(__file__).with_name("screen.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def project_revision(project):
    snapshot = project / "baseline.json"
    if snapshot.exists():
        return json.loads(snapshot.read_text()).get("git_commit", "unavailable")
    return command_output("git", "-C", str(project), "rev-parse", "HEAD")


def command_output(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def topology():
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    nodes = {}
    for cpu in allowed:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        def read(relative, fallback):
            try:
                return (path / relative).read_text().strip()
            except OSError:
                return fallback
        core = read("topology/core_id", str(cpu))
        package = read("topology/physical_package_id", "0")
        l3 = read("cache/index3/id", "unknown")
        nodes[cpu] = {"package": package, "core": core, "l3": l3}
    return allowed, nodes


def placement_mask(count, mode, allowed, nodes):
    if mode == "ordinary":
        return allowed
    cores = collections.defaultdict(list)
    for cpu in allowed:
        value = nodes[cpu]
        cores[(value["package"], value["core"])].append(cpu)
    first = {key: min(cpus) for key, cpus in cores.items()}
    groups = collections.defaultdict(list)
    for key, cpu in first.items():
        groups[nodes[cpu]["l3"]].append(cpu)
    for group in groups.values():
        group.sort()
    if mode == "compact":
        order = [cpu for group in sorted(groups) for cpu in groups[group]]
    else:
        order = []
        maximum = max(map(len, groups.values()))
        for index in range(maximum):
            order.extend(groups[group][index] for group in sorted(groups) if index < len(groups[group]))
    siblings = [cpu for cpu in allowed if cpu not in set(first.values())]
    order.extend(siblings)
    if count > len(order):
        raise ValueError(f"requested {count} workers but affinity permits {len(order)} CPUs")
    return order[:count]


def memory_metadata():
    try:
        meminfo = Path("/proc/meminfo").read_text().splitlines()
        return {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in meminfo if line.startswith(("MemTotal:", "MemAvailable:"))}
    except OSError:
        return {}


def machine_metadata(allowed, nodes):
    governors = {}
    boost = {}
    for cpu in allowed:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
        if path.exists():
            governors[str(cpu)] = path.read_text().strip()
    for name in ("/sys/devices/system/cpu/cpufreq/boost", "/sys/devices/system/cpu/amd_pstate/status"):
        if Path(name).exists():
            boost[name] = Path(name).read_text().strip()
    return {"os": platform.platform(), "python": sys.version.split()[0],
            "allowed_cpus": allowed, "cpu_topology": nodes,
            "governors": governors, "boost": boost, "memory": memory_metadata()}


def invoke(args, case, nx, ny, count, backend, steps, seconds, mask, output, more, profile_output=None):
    source_before = source_hash(args.project)
    runner_before = runner_hash()
    toy_before = toy_tree_hash() if case == "toy" else None
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", JULIA_NUM_GC_THREADS="1")
    if args.depot is not None:
        env["JULIA_DEPOT_PATH"] = str(args.depot) + os.pathsep + str(Path.home() / ".julia")
    command = [args.julia, "--startup-file=no", f"--project={args.project}",
               f"--threads={count},0" if args.julia_minor >= 12 else f"--threads={count}", str(DRIVER), f"case={case}",
               f"backend={backend}", f"nx={nx}", f"ny={ny}",
               f"tile={args.tile}", f"steps={steps}", f"seconds={seconds}",
               f"output={output}"]
    command += [f"{key}={value}" for key, value in more.items()]
    if args.pin_workers:
        command.extend(["pin-workers=true", "pin-cpus=" + ",".join(map(str, mask))])
    if profile_output is not None:
        command.extend(["profile=true", f"profile-output={profile_output}"])
    def set_affinity():
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, mask)
    launched = time.perf_counter()
    proc = subprocess.run(command, env=env, text=True, capture_output=True,
                          preexec_fn=set_affinity if args.placement != "ordinary" else None)
    if proc.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{proc.stdout}\n{proc.stderr}")
    with open(output, "rb") as stream:
        record = tomllib.load(stream)
    record["process_elapsed_seconds"] = time.perf_counter() - launched
    if record["runtime"]["threads_default"] != count:
        raise RuntimeError(f"requested {count} default Julia threads but runtime used {record['runtime']['threads_default']}")
    record["stdout"] = proc.stdout.strip()
    record["requested_workers"] = count
    record["requested_affinity"] = mask
    record["project"] = str(args.project)
    record["project_revision"] = project_revision(args.project)
    record["source_tree_sha256"] = source_before
    if source_hash(args.project) != source_before or runner_hash() != runner_before:
        raise RuntimeError("project or runner source changed during a benchmark process")
    if case == "toy" and toy_tree_hash() != toy_before:
        raise RuntimeError("toy source changed during a benchmark process")
    record["runner_sha256"] = runner_before
    record["toy_tree_sha256"] = toy_before
    record["placement"] = args.placement
    record["pin_workers"] = args.pin_workers
    record["profile_output"] = str(profile_output) if profile_output else None
    return record


def summary(records, seed):
    grouped = collections.defaultdict(list)
    for record in records:
        key = (record["case"], record["backend"], record["nx"], record["ny"], record["requested_workers"], record["placement"])
        grouped[key].append(record)
    rng = random.Random(seed)
    output = []
    for key, samples in sorted(grouped.items()):
        one_key = key[:4] + (1,) + key[5:]
        one = grouped.get(one_key)
        times = [item["time_seconds"] for item in samples]
        t = statistics.median(times)
        baseline = statistics.median(item["time_seconds"] for item in one) if one else float("nan")
        speedup = baseline / t if one else None
        replicas = []
        if key[4] == 1:
            interval = [1.0, 1.0]
        elif one and len(one) >= 2 and len(samples) >= 2:
            for _ in range(2000):
                p_times = [rng.choice(times) for _ in times]
                one_times = [rng.choice(one)["time_seconds"] for _ in one]
                replicas.append(statistics.median(one_times) / statistics.median(p_times))
            replicas.sort()
            interval = [replicas[50], replicas[1949]]
        elif key[4] != 1:
            interval = [None, None]
        gate = "screen_only"
        if key[4] in (4, 8) and len(times) >= 5:
            if interval[0] is None:
                gate = "inconclusive"
            elif interval[0] >= 0.5 * key[4]:
                gate = "timing_target_supported"
            elif interval[1] < 0.5 * key[4]:
                gate = "timing_target_missed"
            else:
                gate = "inconclusive"
        active_gate = all(item["active_workers"] == key[4] for item in samples)
        if gate == "timing_target_supported" and not active_gate:
            gate = "invalid_inactive_workers"
        output.append({"case": key[0], "backend": key[1], "nx": key[2], "ny": key[3],
                       "workers": key[4], "placement": key[5], "samples": len(times),
                       "time_median_seconds": t,
                       "time_mad_seconds": statistics.median(abs(value-t) for value in times),
                       "speedup": speedup, "efficiency": speedup/key[4] if speedup is not None else None,
                       "speedup_ci95_bootstrap": interval,
                       "efficiency_gate_50pct": gate, "all_requested_workers_active_in_probe": active_gate,
                       "active_workers_min": min(item["active_workers"] for item in samples),
                       "active_workers_max": max(item["active_workers"] for item in samples),
                       "worker_probe_imbalance_median": statistics.median(item["worker_probe_imbalance_ratio"] for item in samples),
                       "allocation_median_bytes": statistics.median(item["allocation_bytes"] for item in samples),
                       "gc_median_seconds": statistics.median(item["gc_seconds"] for item in samples),
                       "peak_rss_median_bytes": statistics.median(item["peak_rss_bytes_linux"] for item in samples),
                       "mass_error_max": max(item["diagnostics"].get("mass_error", 0) for item in samples),
                       "l2_error_max": max(item["diagnostics"].get("l2_error", 0) for item in samples),
                       "working_set_bytes": samples[0]["fixture"]["working_set_bytes"],
                       "steps": samples[0]["steps"],
                       "throughput_cells_per_second": samples[0]["fixture"]["cells"]*samples[0]["steps"]/t})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--julia", default=command_output("which", "julia"))
    parser.add_argument("--depot", type=Path, help="optional Julia depot; otherwise inherit JULIA_DEPOT_PATH")
    parser.add_argument("--profile", action="store_true", help="write separate untimed flat Profile trace per process")
    parser.add_argument("--cases", nargs="+", default=["stream", "stencil", "gather", "launch", "toy"])
    parser.add_argument("--sizes", nargs="*", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--shapes", nargs="*", default=[], help="additional NxM rectangles, e.g. 257x193")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=0.25,
                        help="target per sample at maximum physical worker count")
    parser.add_argument("--placement", choices=["ordinary", "compact", "spread"], default="ordinary")
    parser.add_argument("--pin-workers", action="store_true", help="bind each Julia default worker to one distinct selected Linux CPU")
    parser.add_argument("--backend", default="auto", help="controls: threads; upwind: package; toy: threads")
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--tile-x", type=int, default=128)
    parser.add_argument("--threshold", type=int, default=16384)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--launch-items", type=int)
    parser.add_argument("--flux", choices=["recompute", "stored"], default="recompute")
    parser.add_argument("--steps", type=int, default=0, help="fixed trajectory steps for every case and worker count")
    parser.add_argument("--workload", type=Path, help="reuse frozen workload.json from a prior run")
    parser.add_argument("--allow-large-upwind", action="store_true", help="allow main-package upwind meshes above 1024² after memory preflight")
    parser.add_argument("--allow-large-wy", action="store_true", help="allow WY reversal above 256² after memory preflight")
    parser.add_argument("--diagnostics", choices=["summary", "sampled"], default="summary")
    parser.add_argument("--diagnostic-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=29092026)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="one process/count, 128² and very short samples")
    args = parser.parse_args()
    if args.smoke:
        args.cases, args.sizes, args.threads, args.repeats, args.seconds = ["stream", "stencil", "gather", "launch", "upwind", "upwind_complete"], [128], [1, 2], 1, 0.01
    try:
        shapes = [(n, n) for n in args.sizes] + [tuple(map(int, shape.lower().split("x"))) for shape in args.shapes]
    except ValueError:
        parser.error("shapes must use NxM with positive integers")
    if not shapes or any(len(shape) != 2 or any(n < 1 for n in shape) for shape in shapes):
        parser.error("provide at least one positive square size or rectangular shape")
    if args.repeats < 1 or any(p < 1 for p in args.threads) or any(n < 1 for n in args.sizes):
        parser.error("repeats, threads, and sizes must be positive")
    if args.tile < 1 or args.tile_x < 1 or args.threshold < 0 or args.block < 1 or args.diagnostic_interval < 1:
        parser.error("tile dimensions and block must be positive; threshold must be nonnegative")
    if any(case in ("upwind", "upwind_complete", "wy_complete", "wy_kernel") for case in args.cases) and args.backend not in ("auto", "package"):
        parser.error("upwind uses the package backend; run alternative backends as separate cases")
    if args.pin_workers and args.placement == "ordinary":
        parser.error("--pin-workers requires compact or spread placement")
    if args.steps < 0:
        parser.error("steps must be nonnegative")
    if any(case in ("upwind", "upwind_complete") for case in args.cases) and max(max(shape) for shape in shapes) > 1024 and not args.allow_large_upwind:
        parser.error("main-package upwind above 1024² requires --allow-large-upwind; choose sizes after memory preflight")
    if any(case in ("wy_complete", "wy_kernel") for case in args.cases) and max(max(shape) for shape in shapes) > 256 and not args.allow_large_wy:
        parser.error("WY above 256² requires --allow-large-wy; full T=1 trajectory is costly")
    if args.allow_large_wy and any(case in ("wy_complete", "wy_kernel") for case in args.cases):
        available = memory_metadata().get("MemAvailable", "0 kB").split()[0]
        estimated_bytes = 700 * max(nx*ny for nx,ny in shapes)
        if estimated_bytes > 0.5 * int(available) * 1024:
            parser.error(f"estimated WY working set {estimated_bytes/2**30:.1f} GiB exceeds half available memory")
    if args.allow_large_upwind and any(case in ("upwind", "upwind_complete") for case in args.cases):
        available = memory_metadata().get("MemAvailable", "0 kB").split()[0]
        estimated_bytes = 500 * max(nx*ny for nx,ny in shapes)
        if estimated_bytes > 0.5 * int(available) * 1024:
            parser.error(f"estimated upwind working set {estimated_bytes/2**30:.1f} GiB exceeds half available memory")
    if not args.julia:
        parser.error("Julia executable unavailable; pass --julia")
    resolved_julia = shutil.which(args.julia)
    if resolved_julia is None:
        parser.error(f"Julia executable unavailable: {args.julia}")
    args.julia = resolved_julia
    version_text = command_output(args.julia, "--version")
    match = re.search(r"julia version (\d+)\.(\d+)", version_text, re.IGNORECASE)
    if match is None or int(match.group(1)) != 1:
        parser.error(f"cannot determine supported Julia 1.x version from {version_text!r}")
    args.julia_minor = int(match.group(2))
    if args.depot is not None:
        args.depot.mkdir(parents=True, exist_ok=True)
    allowed, nodes = topology()
    for count in args.threads:
        placement_mask(count, args.placement, allowed, nodes)
    date = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output or (ROOT / "benchmark" / "scaling" / "results" / date)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "raw.jsonl").touch(exist_ok=False)
    config = {"schema_version": 1, "created_utc": date, "cases": args.cases,
              "sizes": args.sizes, "shapes": args.shapes, "threads": args.threads, "repeats": args.repeats,
              "seconds_target": args.seconds, "placement": args.placement, "pin_workers": args.pin_workers, "backend": args.backend,
              "tile": args.tile, "tile_x": args.tile_x, "threshold": args.threshold,
              "block": args.block, "launch_items": args.launch_items,
              "diagnostics": args.diagnostics, "diagnostic_interval": args.diagnostic_interval,
              "flux": args.flux, "seed": args.seed, "project": str(args.project),
              "steps_override": args.steps, "workload_source": str(args.workload) if args.workload else None,
              "profile": args.profile, "depot": str(args.depot) if args.depot else None,
              "project_revision": project_revision(args.project),
              "workspace_revision": command_output("git", "-C", str(ROOT), "rev-parse", "HEAD"),
              "source_tree_sha256": source_hash(args.project),
              "runner_sha256": runner_hash(),
              "toy_tree_sha256": toy_tree_hash() if "toy" in args.cases else None,
              "workspace_dirty": command_output("git", "-C", str(ROOT), "status", "--porcelain"),
              "julia": args.julia, "julia_version_text": version_text,
              "threads_flag_policy": "p,0" if args.julia_minor >= 12 else "p",
              "machine": machine_metadata(allowed, nodes)}
    config["configuration_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (destination / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    # Calibrate at the largest requested worker count, then freeze its trajectory
    # across all worker counts and repeats. This avoids an ill-defined scaling curve.
    work = []
    frozen = json.loads(args.workload.read_text()) if args.workload else {}
    with tempfile.TemporaryDirectory(prefix="twophase-scaling-") as temporary:
        toml_path = Path(temporary) / "sample.toml"
        for case in args.cases:
            backend = args.backend if args.backend != "auto" else ("package" if case in ("upwind", "upwind_complete", "wy_complete", "wy_kernel") else "threads")
            for nx, ny in shapes:
                count = max(args.threads)
                mask = placement_mask(count, args.placement, allowed, nodes)
                extra = {"flux": args.flux, "tile-x": args.tile_x, "threshold": args.threshold} if case == "toy" else {}
                if case in ("upwind", "upwind_complete", "wy_complete", "wy_kernel"):
                    extra["block"] = args.block
                if case in ("wy_complete", "wy_kernel"):
                    extra["diagnostics"] = args.diagnostics
                    extra["diagnostic-interval"] = args.diagnostic_interval
                if case == "launch" and args.launch_items:
                    extra["launch-items"] = args.launch_items
                workload_key = f"{case}:{nx}:{ny}"
                if args.steps:
                    steps = args.steps
                elif workload_key in frozen:
                    steps = int(frozen[workload_key])
                else:
                    record = invoke(args, case, nx, ny, count, backend, 0, args.seconds, mask, toml_path, extra)
                    steps = record["steps"]
                frozen[workload_key] = steps
                work.extend((case, nx, ny, count0, backend, steps, extra, repeat)
                            for repeat in range(args.repeats) for count0 in args.threads)
                print(f"frozen {case}/{backend} {nx}×{ny}: {steps} steps", flush=True)
        (destination / "workload.json").write_text(json.dumps(frozen, indent=2) + "\n")
        rng = random.Random(args.seed)
        # Every repeat visits all counts once, in independently randomized order.
        by_repeat = collections.defaultdict(list)
        for job in work:
            by_repeat[job[-1]].append(job)
        records = []
        for repeat in range(args.repeats):
            rng.shuffle(by_repeat[repeat])
            for case, nx, ny, count, backend, steps, extra, _ in by_repeat[repeat]:
                mask = placement_mask(count, args.placement, allowed, nodes)
                profile_output = destination / f"{case}-{nx}x{ny}-p{count}-repeat{repeat}.profile.txt" if args.profile else None
                record = invoke(args, case, nx, ny, count, backend, steps, args.seconds,
                                mask, toml_path, extra, profile_output)
                if record["source_tree_sha256"] != config["source_tree_sha256"] or record["runner_sha256"] != config["runner_sha256"]:
                    raise RuntimeError("project or runner source differs from the frozen configuration")
                if case == "toy" and record["toy_tree_sha256"] != config["toy_tree_sha256"]:
                    raise RuntimeError("toy source differs from the frozen configuration")
                record["configuration_sha256"] = config["configuration_sha256"]
                record["repeat"] = repeat
                record["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                with (destination / "raw.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                records.append(record)
                print(f"repeat {repeat+1}/{args.repeats} {record['stdout']}", flush=True)
                (destination / "summary.json").write_text(json.dumps(summary(records, args.seed), indent=2) + "\n")
    print(f"Saved {len(records)} independent process samples to {destination}")


if __name__ == "__main__":
    main()
