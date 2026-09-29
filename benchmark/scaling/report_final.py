#!/usr/bin/env python3
"""Audit and plot a completed, frozen Stage 2 final scaling artifact."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys


COLORS = {"standard": "#1f77b4", "fused_affine": "#d95f02",
          "fused_amortized": "#7570b3", "halo": "#666666"}


def scope(variant):
    return "amortized_trajectory" if variant == "fused_amortized" else "observable_step"


def serial_eligible(variant, target_scope):
    return target_scope == "amortized_trajectory" or variant != "fused_amortized"


def percentile(sorted_values, fraction):
    return sorted_values[int(fraction * (len(sorted_values) - 1))]


def ratio_ci(baseline, candidate, seed, iterations=20_000):
    if baseline is candidate:
        return [1.0, 1.0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        a = statistics.median(rng.choices(baseline, k=len(baseline)))
        b = statistics.median(rng.choices(candidate, k=len(candidate)))
        estimates.append(a / b)
    estimates.sort()
    return [percentile(estimates, .025), percentile(estimates, .975)]


def key(row):
    return row["nx"], row["variant"], tuple(row["tile"]), row["workers"]


def audit(raw, config, workload):
    expected = config["repeats"]
    groups = collections.defaultdict(list)
    for row in raw:
        groups[key(row)].append(row)
    if len(raw) != 56 * expected:
        raise ValueError(f"expected {56 * expected} final samples, got {len(raw)}")
    if any(len(rows) != expected for rows in groups.values()):
        raise ValueError("every final configuration must have five independent samples")
    hash_keys = ("project_sha256", "toy_tree_sha256", "experiment_sha256", "runner_sha256", "configuration_sha256", "workload_sha256")
    for name in hash_keys:
        values = {row[name] for row in raw}
        if len(values) != 1:
            raise ValueError(f"source/configuration/workload changed: {name}")
    if {row["project_sha256"] for row in raw} != {config["project_sha256"]}:
        raise ValueError("project hash differs from configuration")
    for n in config["sizes"]:
        rows = [row for row in raw if row["nx"] == n]
        if {row["steps"] for row in rows} != {workload[str(n)]}:
            raise ValueError(f"workload steps differ at {n}²")
        if len({row["dt"] for row in rows}) != 1 or len({row["final_time"] for row in rows}) != 1:
            raise ValueError(f"physical trajectory differs at {n}²")
        if len({row["translation_l2"] for row in rows}) != 1:
            raise ValueError(f"analytic translation error differs at {n}²")
    for row in raw:
        if row["variant"] != "halo" and "probe_state_delta" not in row:
            raise ValueError("instrumented kernel lacks probe parity field")
        if row["max_state_delta"] != 0 or row.get("probe_state_delta", 0) != 0 or row["mass_error"] > 512 * sys.float_info.epsilon:
            raise ValueError("numerical parity, probe parity, or conservation failed")
        if not math.isfinite(row["translation_l2"]):
            raise ValueError("nonfinite analytic error")
        if row["requested_workers"] != row["workers"] or len(row["affinity"]) != row["workers"]:
            raise ValueError("requested worker or affinity metadata inconsistent")
        if row["variant"] == "halo":
            if row["workers"] != 1 or row["active_workers"] != 0 or "not instrumented" not in row["active_worker_scope"]:
                raise ValueError("halo must be an explicitly uninstrumented serial-only reference")
        elif not 1 <= row["active_workers_min_pass"] <= row["active_workers"] <= row["tile_worker_capacity"]:
            raise ValueError("invalid actual compute worker participation")
        if row["nx"] in (512, 1024, 2048, 4096) and row["workers"] in (4, 8) and row["active_workers_min_pass"] != row["workers"]:
            raise ValueError("mandatory primary point failed actual per-pass participation")
    return groups


def build_report(groups, raw, config, workload):
    samples = {}
    for group_key, rows in groups.items():
        samples[group_key] = [row["time_seconds"] for row in rows]
    by_size_scope = collections.defaultdict(list)
    for group_key, rows in groups.items():
        n, variant, tile, p = group_key
        if p != 1:
            continue
        for target_scope in ("observable_step", "amortized_trajectory"):
            if serial_eligible(variant, target_scope):
                by_size_scope[(n, target_scope)].append(group_key)
    fastest = {scope_key: min(keys, key=lambda item: statistics.median(samples[item]))
               for scope_key, keys in by_size_scope.items()}
    result = []
    for group_key, rows in sorted(groups.items()):
        n, variant, tile, p = group_key
        own = (n, variant, tile, 1)
        target_scope = scope(variant)
        reference = fastest[(n, target_scope)]
        times = samples[group_key]
        median = statistics.median(times)
        own_median = statistics.median(samples[own])
        ref_median = statistics.median(samples[reference])
        seed = config["seed"] + n + p + sum(map(ord, variant)) + tile[0] + tile[1]
        self_ci = ratio_ci(samples[own], times, seed) if own != group_key else [1.0, 1.0]
        abs_ci = ratio_ci(samples[reference], times, seed + 10101) if reference != group_key else [1.0, 1.0]
        serial_regression_ci = ratio_ci(samples[own], samples[reference], seed + 20202) if own != reference else [1.0, 1.0]
        self_efficiency_ci = [value / p for value in self_ci]
        target_classification = ("pass" if self_efficiency_ci[0] >= .5 else
                                 "fail" if self_efficiency_ci[1] < .5 else "inconclusive") if p in (4, 8) else "diagnostic"
        med = lambda field: statistics.median(row[field] for row in rows)
        result.append({"size": n, "variant": variant, "tile": list(tile), "api_scope": target_scope,
                       "requested_workers": p, "actual_workers_min": min(row["active_workers"] for row in rows),
                       "actual_workers_min_pass": min(row["active_workers_min_pass"] for row in rows),
                       "tile_worker_capacity": rows[0]["tile_worker_capacity"],
                       "worker_evidence_scope": rows[0]["active_worker_scope"],
                       "samples": len(rows), "steps": rows[0]["steps"],
                       "time_median_seconds": median, "time_min_seconds": min(times),
                       "time_max_seconds": max(times),
                       "self_speedup": own_median / median, "self_speedup_ci95": self_ci,
                       "self_efficiency": own_median / median / p,
                       "self_efficiency_ci95": self_efficiency_ci,
                       "fastest_serial_variant": reference[1], "fastest_serial_tile": list(reference[2]),
                       "fastest_serial_median_seconds": ref_median,
                       "absolute_speedup": ref_median / median, "absolute_speedup_ci95": abs_ci,
                       "absolute_efficiency": ref_median / median / p,
                       "absolute_efficiency_ci95": [value / p for value in abs_ci],
                       "serial_regression_fraction": own_median / ref_median - 1,
                       "serial_regression_ci95": [value - 1 for value in serial_regression_ci],
                       "initial_serial_regression_budget": 0.05,
                       "self_median_above_target_at_this_count": p in (4, 8) and own_median / median / p >= .5,
                       "self_target_ci_classification": target_classification,
                       "translation_l2": rows[0]["translation_l2"],
                       "mass_error_max": max(row["mass_error"] for row in rows),
                       "peak_rss_median_bytes": med("peak_rss_bytes"),
                       "numerical_array_bytes": rows[0]["numerical_array_bytes"],
                       "throughput_cell_steps_per_second": n * n * rows[0]["steps"] / median,
                       "allocation_median_bytes": med("allocation_bytes"),
                       "allocation_median_bytes_per_step": med("allocation_bytes") / rows[0]["steps"],
                       "valid_sample_duration": min(times) >= 0.2})
    report = {"schema_version": 1,
              "scope_rules": {"observable_step": "fastest p1 among standard, halo, and fused_affine at identical size/trajectory; excludes amortized-only implementation",
                              "amortized_trajectory": "fastest p1 among all complete final-state trajectories, including standard and halo",
                              "halo_evidence": "p1 serial time and numerical parity only; kernel worker IDs not instrumented",
                              "efficiency_denominator": "requested Julia compute workers, even when tile count underoccupies them"},
              "input_samples": len(raw), "workload": workload, "rows": result}
    return report


def write_markdown(report, path):
    rows = report["rows"]
    lines = ["# Stage 2 final scaling audit", "",
             "Five independent fresh-process samples per configuration. Speedups use the same fixed steps per size. The absolute reference is the fastest measured one-worker complete trajectory within the indicated API scope. The observable-step scope excludes amortized-only implementations; the amortized scope includes standard and halo serial trajectories. Halo's p1 worker-ID count is uninstrumented, not zero computation.", "",
             "| Size | Scope/configuration | p | Active min/pass | Time median (s) | Self E | Self 95% CI | CI gate | Absolute E | Absolute 95% CI | Serial regression | Reference | Min sample (s) |", "|---:|---|---:|---:|---:|---:|---|---|---:|---|---:|---|---:|"]
    for row in rows:
        if row["size"] not in (512, 1024, 2048, 4096):
            continue
        label = f"{row['variant']} {row['tile'][0]}×{row['tile'][1]}"
        reference = f"{row['fastest_serial_variant']} {row['fastest_serial_tile'][0]}×{row['fastest_serial_tile'][1]}"
        ci = row["absolute_efficiency_ci95"]
        self_ci = row["self_efficiency_ci95"]
        workers = "unobserved (serial)" if row["variant"] == "halo" else str(row["actual_workers_min_pass"])
        lines.append(f"| {row['size']} | {label} | {row['requested_workers']} | {workers} | {row['time_median_seconds']:.4f} | {row['self_efficiency']:.3f} | [{self_ci[0]:.3f}, {self_ci[1]:.3f}] | {row['self_target_ci_classification']} | {row['absolute_efficiency']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {100*row['serial_regression_fraction']:+.1f}% | {reference} | {row['time_min_seconds']:.3f} |")
    lines += ["", "The primary target is **fixed-configuration self-efficiency** ≥0.50 at p4 or p8 on each mandatory 512², 2048², and 4096² complete-step fixture. Speedup against the fastest validated serial trajectory and the initial ≤5% one-worker regression budget are separate safeguards. Inspect confidence intervals and numerical gates before assigning a stage outcome. Small-grid underoccupancy and SMT are separate diagnostics. Worker IDs show participation; worker busy-time fractions were not measured.", ""]
    path.write_text("\n".join(lines))


def plots(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["rows"]
    for n in (512, 1024, 2048, 4096):
        items = [row for row in rows if row["size"] == n and row["requested_workers"] in (1, 2, 4, 8, 16)]
        policies = sorted({(row["variant"], tuple(row["tile"])) for row in items if row["variant"] != "halo"})
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), layout="constrained")
        for variant, tile in policies:
            curve = sorted((row for row in items if row["variant"] == variant and tuple(row["tile"]) == tile), key=lambda row: row["requested_workers"])
            if len(curve) < 2:
                continue
            p = [row["requested_workers"] for row in curve]
            label = f"{variant} {tile[0]}×{tile[1]}"
            color = COLORS.get(variant, "#444444")
            axes[0].plot(p, [row["time_median_seconds"] for row in curve], marker="o", label=label, color=color)
            self_values = [row["self_efficiency"] for row in curve]
            abs_values = [row["absolute_speedup"] for row in curve]
            axes[1].plot(p, self_values, marker="o", label=label, color=color)
            axes[1].fill_between(p, [row["self_efficiency_ci95"][0] for row in curve],
                                 [row["self_efficiency_ci95"][1] for row in curve], color=color, alpha=.15)
            axes[2].plot(p, abs_values, marker="o", label=label, color=color)
            axes[2].fill_between(p, [row["absolute_speedup_ci95"][0] for row in curve],
                                 [row["absolute_speedup_ci95"][1] for row in curve], color=color, alpha=.15)
            axes[3].plot(p, [row["self_speedup"] for row in curve], marker="o", label=label, color=color)
            axes[3].fill_between(p, [row["self_speedup_ci95"][0] for row in curve],
                                 [row["self_speedup_ci95"][1] for row in curve], color=color, alpha=.15)
            for row in curve:
                if row["actual_workers_min_pass"] < row["requested_workers"]:
                    axes[1].annotate(f"{row['actual_workers_min_pass']} active", (row["requested_workers"], row["self_efficiency"]), xytext=(4, 5), textcoords="offset points", fontsize=7)
        axes[0].set(ylabel="Wall time (s)", title=f"{n}² complete trajectory")
        axes[1].axhline(.5, color="black", linestyle="--", linewidth=1, label="50% target")
        axes[1].set(ylabel="Self efficiency T₁/(pTₚ)", title="Fixed-configuration scaling")
        axes[2].set(ylabel="Speedup vs fastest serial", title="Absolute benefit by API scope")
        axes[3].set(ylabel="Self speedup T₁/Tₚ", title="Fixed-configuration speedup")
        for ax in axes:
            ax.set(xlabel="Requested Julia workers", xticks=[1, 2, 4, 8, 16])
            ax.grid(alpha=.25)
        axes[0].legend(fontsize=8)
        for extension in ("svg", "png"):
            fig.savefig(destination / f"stage2-final-{n}.{extension}", dpi=160)
        plt.close(fig)
    standard = [row for row in rows if row["variant"] == "standard" and row["requested_workers"] in (1, 4, 8, 16)]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    for p in (1, 4, 8, 16):
        curve = sorted((row for row in standard if row["requested_workers"] == p), key=lambda row: row["size"])
        ax.plot([row["numerical_array_bytes"] / 2**20 for row in curve],
                [row["throughput_cell_steps_per_second"] / 1e6 for row in curve],
                marker="o", label=f"{p} workers")
    ax.set_xscale("log")
    ax.set(xlabel="Numerical-array working set (MiB)", ylabel="Million cell-steps/s",
           title="Standard LW throughput; frozen steps differ by size")
    ax.grid(alpha=.25)
    ax.legend()
    for extension in ("svg", "png"):
        fig.savefig(destination / f"stage2-final-throughput-working-set.{extension}", dpi=160)
    plt.close(fig)
    selected = [row for row in rows if row["size"] in (512, 2048, 4096) and row["requested_workers"] in (1, 4, 8)]
    labels = [f"{row['size']}² {row['variant']} {row['tile'][0]}×{row['tile'][1]} p{row['requested_workers']}" for row in selected]
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, .28 * len(selected))), layout="constrained")
    y = range(len(selected))
    axes[0].barh(y, [row["numerical_array_bytes"] / 2**20 for row in selected], color="#4c78a8")
    axes[0].set(xlabel="Numerical-array storage (MiB)", yticks=list(y), yticklabels=labels)
    axes[1].barh(y, [row["allocation_median_bytes_per_step"] for row in selected], color="#f58518")
    axes[1].set(xlabel="Timed allocation per step (bytes)", yticks=list(y), yticklabels=[])
    for extension in ("svg", "png"):
        fig.savefig(destination / f"stage2-final-memory-allocation.{extension}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    path = args.result_dir
    raw_path = path / "raw.jsonl"
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    config = json.loads((path / "configuration.json").read_text())
    workload = json.loads((path / "workload.json").read_text())
    groups = audit(raw, config, workload)
    report = build_report(groups, raw, config, workload)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["report_script_sha256"] = script_hash
    report["raw_jsonl_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    report["original_summary_sha256"] = hashlib.sha256((path / "summary.json").read_bytes()).hexdigest()
    report["config_sha256"] = config["configuration_sha256"]
    (path / "report-corrected.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, path / "report-corrected.md")
    plots(report, path)
    print(f"Audited {len(raw)} samples and wrote corrected scope report/figures to {path}")


if __name__ == "__main__":
    main()
