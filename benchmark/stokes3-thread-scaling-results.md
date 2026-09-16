# Stokes-3 shared-memory scaling

## Configuration and controls

- Date: 2026-09-16
- Julia: 1.12.7
- CPU: AMD Ryzen 9 5950X, 16 physical cores / 32 hardware threads
- Julia compute threads: 1, 2, 4, and 8 in separate processes
- BLAS threads: 1
- Physical interval: `t=0` through `t=1`
- Default example mesh: 8×8-cell blocks and maximum tree level 5
- Pressure and diffusion preconditioners: tree-based geometric multigrid

Every run took 448 time steps and changed topology 71 times. The scaling
driver compared every row of the traces and established exact equality of the
physical-time sequence, time-step sizes, AMR decisions, complete ordered block
keys, cell and face counts, pressure and diffusion iteration counts, and
multigrid-level reuse counts. The results therefore compare the same numerical
and mesh work. Aggregate data are in
[`stokes3-thread-results/scaling.csv`](stokes3-thread-results/scaling.csv) and
[`stokes3-thread-results/stage-scaling.csv`](stokes3-thread-results/stage-scaling.csv).

The table uses the sum of measured stages. The profiler's enclosing wall time
also contains diagnostic work such as recursive retained-size measurement,
block-key serialization, record construction, and printing. Those operations
are deliberately outside the physical stage totals and should not be used to
judge solver scaling.

## End-to-end physical stages

| Threads | Accounted time | Speedup | Parallel efficiency | Median step | Mean fixed-topology step | Mean topology-changing step | Allocated |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.829 s | 1.00× | 100.0% | 42.286 ms | 53.537 ms | 234.450 ms | 18.720 GiB |
| 2 | 32.449 s | 1.13× | 56.7% | 31.909 ms | 44.714 ms | 219.608 ms | 18.824 GiB |
| 4 | 27.682 s | 1.33× | 33.3% | 23.470 ms | 35.360 ms | 202.135 ms | 18.824 GiB |
| 8 | 27.204 s | 1.35× | 16.9% | 21.876 ms | 32.188 ms | 212.236 ms | 18.824 GiB |

The median step is 1.93× faster with eight threads. Fixed-topology steps are
1.66× faster on average. Topology-changing steps improve by only 1.10× at eight
threads and are slightly faster with four threads. Consequently, doubling
from four to eight threads improves the complete run by only 1.8%.

Threading adds 0.56% to measured allocation. Allocation is essentially flat
from two through eight threads, so task creation and parallel scratch state do
not introduce a growing memory cost.

## Stage scaling

| Stage | One-thread time | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| Advection | 6.577 s | 1.21× | 1.99× | 2.75× |
| Viscosity coefficient update | 0.300 s | 1.64× | 1.79× | 1.81× |
| Diffusion solver refresh | 1.295 s | 1.18× | 1.21× | 1.14× |
| Diffusion solve | 5.255 s | 1.19× | 1.43× | 1.51× |
| Pressure operator update | 0.927 s | 0.94× | 1.09× | 0.92× |
| Pressure preconditioner refresh | 0.536 s | 1.07× | 1.18× | 1.10× |
| Surface tension | 0.960 s | 1.18× | 1.31× | 1.02× |
| Gravity | 1.798 s | 1.21× | 1.32× | 1.45× |
| Pressure projection | 3.853 s | 1.46× | 1.82× | 1.89× |
| AMR indicators | 2.257 s | 1.11× | 1.55× | 2.14× |
| AMR field transfer | 1.759 s | 1.07× | 1.16× | 1.19× |
| Topology-dependent rebuild | 11.202 s | 1.01× | 1.03× | 0.96× |

The face/cell numerical work scales: advection, pressure projection, gravity,
and AMR indicator evaluation all become materially faster. The elliptic solves
improve less because coarse multigrid levels and global Krylov reductions have
too little independent work to amortize scheduling and synchronization.
Surface tension is small and saturates at four threads.

The topology-dependent rebuild is the limiting stage. It accounts for 30.4%
of one-thread stage time and 43.0% at eight threads because its absolute time
does not decrease. It retains the serial symbolic tasks identified by the core
benchmarks: geometry schedule discovery outside the locally cached patches,
coordinate assembly and grouping, sparse matrix construction, and hierarchy
bookkeeping. With its measured eight-thread value held fixed, even infinitely
fast remaining stages could not exceed about 3.15× overall speedup.

## Load-balancing decision

These results do not support adding a persistent cost-weighted interface load
balancer yet:

1. The isolated mixed-level interface kernels scale at least as well as their
   uniform counterparts.
2. In the complete adaptive case, advection reaches 2.75× and AMR indicators
   2.14× at eight threads while following exactly the same moving interface
   and tree.
3. The lost scaling is concentrated in topology reconstruction, operator
   refresh, coarse-grid work, and reductions. Reassigning interface blocks
   cannot accelerate those serial or globally synchronized stages.

Fine dynamic batches are therefore adequate for the current shared-memory
implementation. Four threads give nearly all of the measured whole-simulation
benefit on this mesh. Eight threads remain useful for transport-heavy runs or
long intervals between mesh adaptations. A weighted block scheduler should be
reconsidered only after topology construction is reduced or parallelized and
per-worker timing shows a persistent imbalance in interface kernels.

## Reproduction

```bash
STOKES3_THREAD_COUNTS=1,2,4,8 \
  julia --project=. benchmark/stokes3_thread_scaling.jl
```

The driver writes full traces under `/tmp/stokes3-thread-scaling` by default.
It rejects a comparison if any numerical-work or mesh-evolution field differs.
To regenerate the aggregate tables from existing traces, use:

```bash
STOKES3_THREAD_REUSE_RESULTS=true \
  julia --project=. benchmark/stokes3_thread_scaling.jl
```
