# Benchmarks

Run from the package root:

```sh
julia --project -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add("BenchmarkTools"); include("benchmark/benchmarks.jl")'
```

Setup/regridding and time-stepping kernels are measured separately. Cell
advection, one-component staggered advection, and simultaneous two-component
staggered advection are reported independently. Pressure-gradient setup and
application are reported for uniform and mixed-level meshes. Results for the
current implementation and machine are recorded in `results.md`. The
mixed-level case separately measures AMR schedule construction and allocation-
free one- and two-component upwind updates.

The milestone-3 pressure comparison additionally requires
`IncompressibleNavierStokes.jl`. Run it in a temporary environment so the
reference package does not become a runtime dependency:

```sh
julia --project -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add(["BenchmarkTools", "IncompressibleNavierStokes"]); include("benchmark/poisson_comparison.jl")'
```

It compares a 64×64 periodic Float64 single block with the same deterministic
broadband right-hand side, zero initial guess, relative tolerance, Julia
process, CPU thread count, and warm-median protocol. Setup is excluded. The
iterative `psolver_cg` reference is used because, unlike the package's FFT
direct solver, it exposes the tolerance/iteration comparison required by
`DEVELOPMENT.md`.

The package's standalone V-cycle and V-cycle-preconditioned-CG paths can be
compared without extra dependencies:

```sh
julia --project=. benchmark/pcg_multigrid.jl
```

The graded-tree methodology and recorded results are in
[pcg-multigrid-results.md](pcg-multigrid-results.md).

The nonsymmetric staggered implicit-diffusion comparison is reproduced with:

```sh
julia --project=. benchmark/bicgstab_multigrid.jl
```

Recorded standalone-multigrid and V-cycle-preconditioned-BiCGStab results are
in [bicgstab-multigrid-results.md](bicgstab-multigrid-results.md).
