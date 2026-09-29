#!/usr/bin/env python3
"""Compare independent scaling runs at a frozen numerical workload."""
import argparse
import collections
import json
from pathlib import Path


def load(directory):
    config = json.loads((directory / "configuration.json").read_text())
    raw = [json.loads(line) for line in (directory / "raw.jsonl").read_text().splitlines()]
    summary = json.loads((directory / "summary.json").read_text())
    signatures = collections.defaultdict(set)
    for row in raw:
        diagnostics = row["diagnostics"]
        fixture = row["fixture"]
        key = (row["case"], row["nx"], row["ny"])
        signatures[key].add((row["steps"], diagnostics.get("dt"), diagnostics.get("final_time"),
                             diagnostics["numerical_contract"], fixture["cells"],
                             fixture.get("block_nx"), fixture.get("block_ny"),
                             fixture.get("diagnostics_mode"), fixture.get("diagnostic_interval"),
                             row["runtime"]["precision"], row["runtime"]["blas_threads"],
                             row["runtime"]["julia_version"], row["placement"],
                             row.get("pin_workers",False)))
    if any(len(values) != 1 for values in signatures.values()):
        raise ValueError(f"inconsistent workload within {directory}")
    return config, raw, summary, {key: next(iter(value)) for key, value in signatures.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="first is starting-revision baseline")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    loaded = [load(path) for path in args.runs]
    runner_hashes = {config["runner_sha256"] for config, _, _, _ in loaded}
    if len(runner_hashes) > 1:
        raise ValueError("benchmark runner changed across runs; use one harness revision")
    reference = loaded[0][3]
    for config, raw, rows, signatures in loaded[1:]:
        for key in reference.keys() & signatures.keys():
            if reference[key] != signatures[key]:
                raise ValueError(f"different physical workload for {key}: {reference[key]} versus {signatures[key]}")
    one_times = collections.defaultdict(list)
    for index, (_, _, rows, _) in enumerate(loaded):
        for row in rows:
            if row["workers"] == 1:
                one_times[(row["case"], row["nx"], row["ny"], row["placement"])].append((row["time_median_seconds"], index, row["backend"]))
    original_one = {}
    for row in loaded[0][2]:
        if row["workers"] == 1:
            original_one[(row["case"], row["nx"], row["ny"], row["placement"])] = row["time_median_seconds"]
    comparisons = []
    for index, (config, raw, rows, _) in enumerate(loaded):
        for row in rows:
            key = (row["case"], row["nx"], row["ny"], row["placement"])
            fastest, fastest_run, fastest_backend = min(one_times[key])
            baseline = original_one.get(key)
            comparisons.append({"run": str(args.runs[index]), "case": row["case"],
                                "nx": row["nx"], "ny": row["ny"], "workers": row["workers"],
                                "backend": row["backend"], "placement": row["placement"],
                                "steps": row["steps"], "time_median_seconds": row["time_median_seconds"],
                                "self_speedup": row["speedup"], "self_efficiency": row["efficiency"],
                                "self_speedup_ci95": row["speedup_ci95_bootstrap"],
                                "absolute_speedup_vs_fastest_serial": fastest/row["time_median_seconds"],
                                "fastest_serial_run": str(args.runs[fastest_run]),
                                "fastest_serial_backend": fastest_backend,
                                "speedup_vs_starting_revision_serial": baseline/row["time_median_seconds"] if baseline else None,
                                "mass_error_max": row["mass_error_max"], "l2_error_max": row["l2_error_max"]})
    output = args.output or (args.runs[0] / "comparison.json")
    output.write_text(json.dumps({"schema_version": 1, "runs": [str(path) for path in args.runs],
                                  "comparisons": comparisons}, indent=2) + "\n")
    print(f"Compared {len(args.runs)} runs with matched physical workloads: {output}")


if __name__ == "__main__":
    main()
