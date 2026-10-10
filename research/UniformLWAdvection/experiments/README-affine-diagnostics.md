# Affine partition and code-generation diagnostics

`check_affine_partition.jl` is a standalone numerical check for the frozen
affine fused kernel. Its default 1031×257 grid spans three x tiles and a
partial y tile at tile size 512×8. It verifies tile/chunk ownership, compares
20 complete 29-step trajectories to the serial reference, poisons reusable
scratch and the full-grid output with `NaN` before each checked trajectory,
compares every compute pass, and checks the caller-owned worker probe. It
exits nonzero at the first detected discrepancy and reports its cell, tile,
and chunk. The optional `cpus=` list pins each Julia worker; omit it for a
portable numerical check. Use Julia 1.10.12 and an environment with
Polyester installed:

```sh
JULIA_DEPOT_PATH=/tmp/twophase-julia110-depot:/home/ronald/.julia \
  /tmp/julia-1.10.12/bin/julia --startup-file=no \
  --project=/tmp/twophase-baseline110 --threads=7 \
  research/UniformLWAdvection/experiments/check_affine_partition.jl
```

For the reported large-grid case, pass
`nx=4096 ny=4096 steps=29 reps=5` and use `--threads=24`. The original
spread placement can be requested with
`cpus=0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15,16,17,18,19,20,21,22,23`.
Options `poison=0`, `trace=0`, or `track=0` disable individual diagnostic
passes when isolating an issue. The script is not a performance benchmark.

The original pinned SMT screen at 4096²/p24/29 steps failed serial parity
once by `3.125383729596365e-9`, above its `512eps` budget. Its original
artifacts are in
`benchmark/scaling/results/20260929-stage2-smt-baseline110-pinned-spread/`.
Six exact fresh-process reruns and bounded large/odd partition stress checks
were exact; their logs and source hashes are in
`/tmp/twophase-affine-p24-diagnosis/`. **Clean stress does not resolve the
recorded intermittent failure.** Keep the affected SMT curve out of accepted
scaling claims until the cause is found and a fresh numerical gate passes.

`inspect_affine_codegen.jl` compares optimized LLVM and typed IR for the
same tile/copyback functions with `Matrix` arguments (p1 serial branch) and
`PtrArray` arguments (captured by Polyester at p≥2). Run it only outside a
timing window. By default it archives `*.llvm`, `*.typed`, and source/version
metadata under the ignored SMT result directory's `codegen/` subdirectory;
set `TWOPHASE_CODEGEN_DIR` to override. Any IR difference is a code-generation
hypothesis, not proof of a runtime slowdown.
