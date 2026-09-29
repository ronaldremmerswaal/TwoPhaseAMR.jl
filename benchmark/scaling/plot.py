#!/usr/bin/env python3
"""Generate standalone SVG/PNG scaling figures from run.py summary.json."""
import argparse
import collections
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "twophase-scaling-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory holding summary.json and raw.jsonl")
    args = parser.parse_args()
    rows = json.loads((args.results / "summary.json").read_text())
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["case"], row["backend"], row["placement"])].append(row)
    for (case, backend, placement), values in groups.items():
        by_size = collections.defaultdict(list)
        for row in values:
            by_size[(row["nx"], row["ny"])].append(row)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for size, series in sorted(by_size.items()):
            series.sort(key=lambda row: row["workers"])
            label = f"{size[0]}×{size[1]}"
            workers = [row["workers"] for row in series]
            axes[0, 0].plot(workers, [row["time_median_seconds"] for row in series], marker="o", label=label)
            axes[0, 1].plot(workers, [row["speedup"] for row in series], marker="o", label=label)
            axes[1, 0].plot(workers, [row["efficiency"] for row in series], marker="o", label=label)
            axes[1, 1].plot(workers, [row["worker_probe_imbalance_median"] for row in series], marker="o", label=label)
        for axis, ylabel in zip(axes.flat, ("Wall time (s)", "Speedup T₁/Tₚ", "Efficiency Sₚ/p", "Probe item imbalance: max/mean")):
            axis.set(xlabel="Requested Julia workers", ylabel=ylabel)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        axes[0, 0].set_yscale("log")
        axes[1, 0].axhline(0.5, color="gray", ls="--", lw=1)
        fig.suptitle(f"{case} / {backend} / {placement}; fixed trajectory within each size")
        stem = f"{case}-{backend}-{placement}-scaling"
        fig.savefig(args.results / f"{stem}.svg")
        fig.savefig(args.results / f"{stem}.png", dpi=160)
        plt.close(fig)

        # Working-set transition and error are separate from scaling panels.
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        by_worker = collections.defaultdict(list)
        for row in values:
            by_worker[row["workers"]].append(row)
        for worker, series in sorted(by_worker.items()):
            series.sort(key=lambda row: row["working_set_bytes"])
            axes[0].plot([row["working_set_bytes"] / 2**20 for row in series],
                         [row["throughput_cells_per_second"] / 1e6 for row in series],
                         marker="o", label=f"p={worker}")
        ordered = sorted(values, key=lambda row: (row["nx"], row["workers"]))
        axes[1].scatter([row["time_median_seconds"] for row in ordered],
                        [row["l2_error_max"] for row in ordered],
                        c=[row["workers"] for row in ordered], cmap="viridis")
        axes[0].set(xlabel="Working set (MiB)", ylabel="Cell steps/s (million)")
        axes[1].set(xlabel="Wall time (s)", ylabel="Max RMS solution error across runs")
        for axis in axes:
            axis.grid(alpha=0.2)
        axes[0].legend()
        fig.suptitle(f"{case} / {backend} / {placement}; error is trajectory-dependent")
        stem = f"{case}-{backend}-{placement}-working-set"
        fig.savefig(args.results / f"{stem}.svg")
        fig.savefig(args.results / f"{stem}.png", dpi=160)
        plt.close(fig)
    print(f"Wrote figures for {len(groups)} case/backend/placement groups to {args.results}")


if __name__ == "__main__":
    main()
