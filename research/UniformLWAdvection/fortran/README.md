# Fortran/OpenMP uniform Lax–Wendroff toy

`lw_omp.f90` implements the constant-carrier `:recompute` method of the Julia
`UniformLWAdvection` toy. Both use Float64 periodic cell averages, dense
right/top face speed arrays, the same oriented Lax–Wendroff face flux, and
three complete synchronized Strang sweeps `x(dt/2), y(dt), x(dt/2)`. The
observable `q` array is updated at the end of every step. This implementation
uses static OpenMP tile scheduling; Julia's reference backend is Polyester.

Build with `make`; run bounded independent scalar-oracle tests with
`make test`. The test harness uses Julia and the
`benchmark/scaling/Project.toml` tooling environment. The release flags are
printed by `make -n`, and vectorization
diagnostics are saved in ignored `build/vectorization.txt`. The default
layout is 512×16 cells per tile. `-ffp-contract=off` avoids FMA contraction
for more transparent cross-language numerical comparison.

Example:

```sh
./build/lw_omp --nx 512 --ny 512 --steps 20 --threads 4 \
  --tile-x 512 --tile-y 16 --cpus 0,8,1,9 \
  --input /tmp/q0.bin --output /tmp/qfinal.bin
```

The optional input and output are headerless Float64 arrays in Fortran/Julia
column-major order (`write(io, vec(q))` on the Julia side). With no input,
the program initializes the same smooth cell-average translation shape
independently using standard Fortran trigonometric functions. The benchmark
harness supplies Julia's exact initial field to remove trigonometric library
differences. Default speeds are 0.7 and −0.3, and default
`dt = 0.35/(0.7nx + 0.3ny)`; `--u`, `--v`, and `--dt` may override them.

The program prints one JSON record. `time_seconds` covers only the complete
step trajectory after warmup, an untimed tracked/untracked equality probe,
and restoration of the initial field. `seconds_per_unknown_step` divides by
`nx*ny*steps`. The final array is written after timing. The JSON includes
actual worker IDs for all three untimed sweep loops and an exact CPU affinity
verification for each OpenMP worker if `--cpus` is supplied. One unique CPU
per worker is required. `--cpus` binds threads with Linux
`sched_setaffinity`, then checks each mask with `sched_getaffinity`. An absent
`--cpus` is reported as unpinned.
