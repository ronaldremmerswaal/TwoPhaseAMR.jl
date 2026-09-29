#!/usr/bin/env python3
"""Summarize bounded, instrumented Stage 2 component profiles.

The split clocks perturb cache state and scheduling. They are diagnostic
fractions of an instrumented trajectory, never acceptance timings.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tomllib


def pct(value):
    return f"{100 * value:.1f}%"


def collect(standard, affine):
    rows = []
    for size in (2048, 4096):
        for workers in (1, 4, 8):
            path = standard / f"n{size}-p{workers}.toml"
            sample = tomllib.loads(path.read_text())
            if (sample["nx"], sample["ny"], sample["default_threads"]) != (size, size, workers):
                raise ValueError(f"standard profile metadata mismatch: {path}")
            if sample["full_diagnostics"]["max_state_delta"] != 0 or sample["pass_diagnostics"]["max_state_delta"] != 0:
                raise ValueError(f"standard profile numerical parity failed: {path}")
            passes = sample["passes"]
            if [item["name"] for item in passes] != ["x_half_1", "y_full", "x_half_2"]:
                raise ValueError(f"unexpected standard sweep names: {path}")
            split_total = sum(item["median_seconds"] for item in passes)
            if split_total <= 0:
                raise ValueError(f"nonpositive split time: {path}")
            rows.append({
                "implementation": "standard", "mode": "observable_step", "size": size,
                "workers": workers, "steps": sample["steps"],
                "complete_diagnostic_median_seconds": sample["full_median_seconds"],
                "split_median_seconds": split_total,
                "component_fractions": {item["name"]: item["median_seconds"] / split_total for item in passes},
                "complete_allocated_bytes": sample["full_allocated_bytes"],
                "active_worker_ids": sample["observed_worker_ids_untimed"],
                "working_array_bytes": sample["working_array_bytes"],
                "numerical_delta": sample["pass_diagnostics"]["max_state_delta"],
                "source": str(path), "placement_scope": "process affinity mask; no per-worker pin verification",
            })
    for workers in (1, 4, 8):
        path = affine / f"n2048-p{workers}" / "profile.toml"
        sample = tomllib.loads(path.read_text())
        if (sample["nx"], sample["ny"], sample["default_workers"]) != (2048, 2048, workers):
            raise ValueError(f"affine profile metadata mismatch: {path}")
        for name, item in sample["modes"].items():
            if item["complete_checks"]["max_state_delta"] != 0 or item["split_checks"]["max_state_delta"] != 0:
                raise ValueError(f"affine profile numerical parity failed: {path}")
            if item["worker_probe"]["active_worker_ids"] != list(range(1, workers + 1)):
                raise ValueError(f"affine profile worker participation failed: {path}")
            fractions = {key: item[f"diagnostic_{key}_fraction"] for key in ("compute", "copyback", "validation")}
            rows.append({
                "implementation": "affine", "mode": name, "size": 2048,
                "workers": workers, "steps": sample["steps"],
                "complete_diagnostic_median_seconds": item["complete_median_seconds"],
                "split_median_seconds": item["split_clock_median"]["elapsed_seconds"],
                "component_fractions": fractions,
                "complete_allocated_bytes": statistics.median(item["complete_allocated_bytes"]),
                "active_worker_ids": item["worker_probe"]["active_worker_ids"],
                "working_array_bytes": item["numeric_array_bytes"],
                "numerical_delta": item["split_checks"]["max_state_delta"],
                "source": str(path), "placement_scope": "verified per-worker Linux pinning",
                "diagnostic_halo_duplicate_update_fraction": item["analytic_halo_duplicate_update_fraction"],
                "complete_gc_seconds": item["complete_gc_seconds"],
            })
    return rows


def plot(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    panels = [
        ("Standard 2048²", [r for r in rows if r["implementation"] == "standard" and r["size"] == 2048],
         [("x_half_1", "#5684c6"), ("y_full", "#63a27a"), ("x_half_2", "#d88c52")]),
        ("Standard 4096²", [r for r in rows if r["implementation"] == "standard" and r["size"] == 4096],
         [("x_half_1", "#5684c6"), ("y_full", "#63a27a"), ("x_half_2", "#d88c52")]),
        ("Affine 2048²", [r for r in rows if r["implementation"] == "affine"],
         [("compute", "#5684c6"), ("copyback", "#d88c52"), ("validation", "#777777")]),
    ]
    for ax, (title, panel, components) in zip(axes, panels):
        labels = [f"p{r['workers']}" if r["implementation"] == "standard" else
                  f"p{r['workers']} {r['mode'][0]}" for r in panel]
        left = [0.0] * len(panel)
        for name, color in components:
            widths = [row["component_fractions"].get(name, 0.0) for row in panel]
            ax.barh(labels, widths, left=left, label=name, color=color)
            left = [a + b for a, b in zip(left, widths)]
        ax.set_xlim(0, 1.03)
        ax.set_title(title)
        ax.set_xlabel("Fraction of instrumented split trajectory")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=.25)
    figure.suptitle("Diagnostic component clocks; not acceptance timing or worker busy time")
    for extension in ("svg", "png"):
        figure.savefig(destination / f"stage2-component-fractions.{extension}", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--affine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = collect(args.standard, args.affine)
    failed_stderr = args.affine / "n4096-p1.stderr.txt"
    if "tracked affine path changed state or clock" not in failed_stderr.read_text():
        raise ValueError("expected preserved affine 4096 p1 probe failure")
    report = {
        "kind": "stage2_diagnostic_components_not_acceptance",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "standard_profiles_completed": 6,
        "affine_2048_profiles_completed": 3,
        "affine_4096_profiles_completed": 0,
        "affine_4096_failure": str(failed_stderr),
        "scope": "split clocks and CPU sampling perturb scheduling/cache; no hardware counters or worker busy-time measure",
        "rows": rows,
    }
    (args.output / "component-report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Stage 2 diagnostic component profiles", "",
        "The six standard profiles cover p1/4/8 at 2048² and 4096². Three affine profiles cover p1/4/8 at 2048². An affine 4096² p1 run failed the untimed tracked/untracked full-state-or-clock comparison; no later affine profiles ran. The failure remains unresolved.", "",
        "Each percentage below uses instrumented split-clock time, not the acceptance matrix. The standard profiler constrained the process to the selected CPU set but did not pin each worker. The affine profiler verified each worker's CPU mask. Neither profiler measured physical DRAM traffic or worker busy-time imbalance.", "",
        "| Implementation | Grid | Workers | Diagnostic full (s) | Split components | Allocated / full run |",
        "|:--|--:|--:|--:|:--|--:|",
    ]
    for row in rows:
        labels = ", ".join(f"{name} {pct(value)}" for name, value in row["component_fractions"].items())
        lines.append(f"| {row['implementation']} {row['mode']} | {row['size']}² | {row['workers']} | {row['complete_diagnostic_median_seconds']:.3f} | {labels} | {row['complete_allocated_bytes'] / 1024:.1f} KiB |")
    lines += ["", "The three standard Strang sweeps occupy similar shares at both mandatory sizes; no single pass explains the p4/p8 scaling miss. At 2048², affine observable copy-back consumes about 16–24% of the instrumented path at p4/p8, while amortized trajectory copy-back is below 1%. These fractions do not establish a DRAM-bandwidth cause. Julia CPU samples may include compilation/runtime frames; GC time was captured for affine complete trials only, and no causal hardware counter was available.", ""]
    (args.output / "component-report.md").write_text("\n".join(lines))
    plot(rows, args.output)
    print(f"Wrote {len(rows)} diagnostic profile rows to {args.output}")


if __name__ == "__main__":
    main()
