# Cache-cliff diagnostics

These separate diagnostics leave the frozen Fortran/OpenMP toy and its release
binary untouched. `lw_omp_padded.f90` is a copy of the strict three-sweep
kernel with an optional padded leading dimension. `--pad 0|8|32` adds unused
cells after each logical column, while input/output files still contain the
same logical `nx×ny` Float64 field. `test_padded.jl` compares the complete
field with the frozen implementation on one-cell, odd, and nondivisible grids.

Build using `make`; `make test` runs the Julia full-field check with the
`benchmark/scaling` tooling project. The binary accepts the original toy's command line plus
`--pad` and `--counters`. `--counters 0` keeps hardware counters off and is the
setting for padding timing. `--counters 1` counts user-mode cycles,
instructions, Linux generic cache misses, and Zen 3 demand fills from local
memory/IO versus cache within the CCX. `--counters 2` replaces the last two
with Zen 3 hardware-prefetch fills from those origins. The event codes are
0x0843, 0x0243, 0x085a, and 0x025a respectively. Linux generic cache misses
on this CPU correspond to L2 request misses, so they cannot be called L3
misses. Local memory/IO fills are not total DRAM bytes; writebacks and other
traffic are outside these counts.

The optional counters open for each already-created OpenMP worker after
warmup and worker probes. They start immediately before the complete timed
trajectory and stop immediately after its wall timer. The counts include
worker runtime waiting and the small start/stop region boundaries; the wall
timer excludes those boundaries. JSON exposes every worker's raw count,
`time_enabled_ns`, and `time_running_ns`, plus aggregate raw and scaled counts.
Compare running/enabled ratios before using scaled values. These are
instrumented diagnostic runs, separate from the primary timing curves.

`perf_permission.c` checks which own-thread user-mode event types the host
permits without changing any system setting. Hardware counters are optional:
the numerical padding diagnostic runs with `--counters 0` even when PMU
access is denied.

The raw event encodings and source descriptions come from the Linux kernel's
[Zen 3 memory event table](https://github.com/torvalds/linux/blob/master/tools/perf/pmu-events/arch/x86/amdzen3/memory.json).
The kernel [maps generic cache misses to event 0x0964 on Zen 2/3](https://github.com/torvalds/linux/blob/master/arch/x86/events/amd/core.c),
whose [Zen 3 definition](https://github.com/torvalds/linux/blob/master/tools/perf/pmu-events/arch/x86/amdzen3/cache.json)
counts instruction and data requests that missed L2, excluding L2 prefetch.
The two raw source-origin cohorts measure demand and hardware-prefetch fills
separately; neither includes all writes or every source of memory traffic.
Do not convert these fills into measured DRAM bandwidth.

For each event, use its own `time_enabled_ns` and `time_running_ns` fields to
check multiplexing. `aggregate_scaled` is the sum of each worker's
`value * time_enabled_ns / time_running_ns`, calculated before aggregation.
Events are opened independently, so their start and stop instants differ by
the few calls needed to program them. The wall timer excludes this setup,
while counter values can include runtime waiting and brief region boundaries.
Counter runs therefore form a separate diagnostic cohort from timings with
`--counters 0`.

The padded binary has a separate compiled code path even at `--pad 0`.
Compare padding values within this binary at fixed logical grid, thread
placement, and step count; use the frozen implementation only for full-field
numerical parity. Tile sweeps receive whole contiguous allocated arrays with
explicit leading dimension, so there is no logical-field packing in the
timed sweeps. Packing `q(1:nx,:)` for binary output, copying the input logical
field into padded storage, and the logical-field checksum occur outside the
wall timer. The small tests cover one-cell, odd, and nondivisible grids;
large-grid parity must also be checked by the measurement harness before
interpreting performance results.
