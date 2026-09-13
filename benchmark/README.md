# Benchmarks

Run from the package root:

```sh
julia --project -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path="."); Pkg.add("BenchmarkTools"); include("benchmark/benchmarks.jl")'
```

Setup/regridding and time-stepping kernels are measured separately. Cell
advection, one-component staggered advection, and simultaneous two-component
staggered advection are reported independently. Results for the current
implementation and machine are recorded in `results.md`.
