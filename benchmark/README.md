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

Unpreconditioned and geometric-MG-preconditioned Krylov.jl CG can be compared
without extra dependencies:

```sh
julia --project=. benchmark/pcg_multigrid.jl
```

The old standalone-multigrid measurements in
[pcg-multigrid-results.md](pcg-multigrid-results.md) predate the separation of
Krylov iteration and preconditioning and are retained as historical results.

The nonsymmetric staggered implicit-diffusion comparison is reproduced with:

```sh
julia --project=. benchmark/bicgstab_multigrid.jl
```

The old standalone-multigrid results in
[bicgstab-multigrid-results.md](bicgstab-multigrid-results.md) are historical.

Compare all applicable preconditioners on the same graded pressure and
staggered-diffusion systems in a temporary environment:

```sh
julia --project=. -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add(["AlgebraicMultigrid", "IncompleteLU", "Krylov", "LimitedLDLFactorizations"]); include("benchmark/preconditioner_comparison.jl")'
```

The table reports setup time, warm median solve time, Krylov iterations, final
strong residual, and retained preconditioner bytes. Pressure uses CG except
for ILU, which is paired with BiCGSTAB because ILU need not be SPD. Staggered
diffusion uses BiCGSTAB throughout; incomplete Cholesky is inapplicable to its
nonsymmetric AMR operator. One recorded run is in
[preconditioner-results.md](preconditioner-results.md).
