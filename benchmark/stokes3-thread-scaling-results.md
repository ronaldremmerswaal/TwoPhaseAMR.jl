# Stokes-3 shared-memory scaling

## Configuration and controls

- Date: 2026-09-16
- Julia: 1.12.7
- CPU: AMD Ryzen 9 5950X, 16 physical cores / 32 hardware threads
- Julia compute threads: 1, 2, 4, and 8 in separate processes
- BLAS threads: 1
- Physical interval: `t=0` through `t=1`
- Default example mesh: 8x8-cell blocks and maximum tree level 5
- Pressure and diffusion preconditioners: tree-based geometric multigrid

Every run took 448 time steps and changed topology 71 times. The scaling
driver established exact equality of time steps, AMR decisions, ordered block
keys, cell and face counts, pressure and diffusion iteration counts, and
multigrid-level reuse counts. A second eight-thread run passed the same check.
The results therefore compare identical numerical and mesh work.

Aggregate data and complete per-step traces are in
[`stokes3-thread-results-block-local`](stokes3-thread-results-block-local).
The [repeat stage table](stokes3-thread-results-block-local/stage-scaling-8-repeat.csv)
records run-to-run variation.

## End-to-end physical stages

| Threads | Accounted time | Speedup | Efficiency | Median step | Fixed topology | Changing topology | Allocated |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.276 s | 1.00x | 100.0% | 41.108 ms | 51.595 ms | 236.976 ms | 18.645 GiB |
| 2 | 30.843 s | 1.18x | 58.8% | 29.262 ms | 42.398 ms | 209.280 ms | 18.747 GiB |
| 4 | 27.979 s | 1.30x | 32.4% | 22.833 ms | 35.842 ms | 203.753 ms | 18.748 GiB |
| 8 | 27.157 s | 1.34x | 16.7% | 20.949 ms | 32.498 ms | 209.926 ms | 18.748 GiB |

The independent eight-thread run takes 27.144 s, a 0.05% difference. Its
median, mean fixed-topology, and mean changing-topology steps are 21.215,
33.239, and 205.814 ms. The complete result is stable even when individual
stages shift with garbage-collection placement.

The current one-thread total is 1.5% lower than the pre-locality result
(36.829 s), and the eight-thread total is 0.2% lower (27.204 s). Overall
scaling therefore remains essentially unchanged at 1.34x. Eight threads make
the median step 1.96x faster, fixed-topology steps 1.59x faster, and
topology-changing steps 1.13x faster. Moving from four to eight threads lowers
the complete measured stage time by 2.9%.

Threading adds 0.55% to measured allocation. Total allocation falls by about
0.4% relative to the previous study at both one and eight threads.

## Stage scaling

| Stage | One-thread time | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| Advection | 5.908 s | 1.21x | 1.78x | 2.16x |
| Viscosity coefficient update | 0.287 s | 1.25x | 1.33x | 1.35x |
| Diffusion solver refresh | 1.306 s | 1.17x | 1.21x | 1.19x |
| Diffusion solve | 5.541 s | 1.30x | 1.59x | 1.65x |
| Pressure operator update | 0.906 s | 0.85x | 0.84x | 0.99x |
| Pressure preconditioner refresh | 0.534 s | 1.08x | 1.17x | 1.11x |
| Surface tension | 0.815 s | 0.94x | 0.93x | 0.96x |
| Gravity | 1.444 s | 1.02x | 0.99x | 1.17x |
| Pressure projection | 3.823 s | 1.51x | 1.82x | 1.97x |
| AMR indicators | 2.207 s | 1.41x | 1.80x | 1.84x |
| AMR field transfer | 2.220 s | 1.61x | 1.51x | 1.55x |
| Topology-dependent rebuild | 11.186 s | 1.02x | 1.01x | 0.96x |

No complete stage reaches 4x. Topology rebuild remains the largest cost: it is
30.8% of one-thread stage time and 42.7% at eight threads. Its absolute time is
unchanged, within 1%, from the preceding study. Removing the duplicate WY
cell-reference vector reduces its one-thread allocation by 77 MiB over the
run, but global sparse construction and hierarchy bookkeeping still dominate
both time and memory.

The block-local loops reduce the one-thread advection stage by 10.2%, from
6.577 to 5.908 s. The current eight-thread stage is slower than the preceding
measurement: 2.734 s, and 2.836 s on repeat, versus 2.393 s. This is not an
interface load-balance signal. Isolated WY interface, scalar, and momentum
kernels preserve or improve their absolute eight-thread time, and the exact
mesh workload is unchanged. The aggregate stage contains many short scheduled
passes, so parallel granularity, barriers, and run context are the remaining
suspects; the current measurements do not identify which one dominates. A
follow-up should time each WY directional subpass and count scheduled regions
before fusing block-backed work or changing block ownership.

AMR transfer illustrates why stage timing must retain GC metrics. Its
one-thread time rises from 1.759 to 2.220 s even though the isolated dense
prolongation kernel is substantially faster. Garbage collection accounts for
0.746 s of the current stage, and the eight-thread repeat moves transfer time
from 1.436 to 1.564 s as its GC time moves from 0.126 to 0.275 s. The numerical
work and allocation volume agree. This run does not establish a transfer
regression.

## Load-balancing decision

The measurements still do not support an interface-weighted load balancer:

1. Mixed-level interface and compatible transport kernels scale as well as or
   better than their uniform counterparts at the same work size.
2. The complete adaptive run loses most of its scaling in topology rebuild,
   elliptic solves, operator refresh, and short synchronized passes.
3. Reassigning interface blocks cannot accelerate global sparse assembly,
   Krylov reductions, or coarse multigrid levels.

Fine block batches remain adequate for the current workload. Four threads are
the conservative operating point for this small adaptive case; eight threads
still reduce fixed-topology time, but topology changes limit the end-to-end
benefit.

## Reproduction

```bash
STOKES3_THREAD_COUNTS=1,2,4,8 \
STOKES3_THREAD_OUTPUT=benchmark/stokes3-thread-results-block-local \
  julia --project=. benchmark/stokes3_thread_scaling.jl
```

The driver rejects a comparison if any numerical-work or mesh-evolution field
differs. To regenerate aggregate tables from existing traces, set
`STOKES3_THREAD_REUSE_RESULTS=true`.
