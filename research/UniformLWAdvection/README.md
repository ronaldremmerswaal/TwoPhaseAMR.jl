# UniformLWAdvection

This is a small, independent periodic transport experiment for the threading
plan's Stage 2. It evolves Float64 **cell averages** on `[0,1)²`; `u[i,j]`
belongs to the face to the right of cell `(i,j)` and `v[i,j]` to its upper
face. Only one copy of each periodic face is stored. The primary performance
fixture uses a spatially constant prescribed carrier, held fixed during each
step. `set_velocity!` changes both dense carrier arrays between steps. A
separately validated `carrier=:variable` option uses the conservative
variable-speed predictor described below.

The step is the conservative Strang sequence `x(dt/2)`, `y(dt)`, `x(dt/2)`.
For each face of a directional substep of duration `τ`, spacing `h`, and
constant face speed `a`, the flux is

```math
F_{i+1/2}=a(q_i+q_{i+1})/2-a^2\tau(q_{i+1}-q_i)/(2h).
```

An output cell gathers the difference of its positive and negative face
fluxes. `:stored` computes each unique face once in a separate array, then
gathers; `:recompute` evaluates each face for each adjacent cell with the
same orientation and arithmetic. Each sweep reads an immutable input and
writes a distinct output. These choices give the same arithmetic answer in
the current tests. No clipping or `@fastmath` is used: unlimited LW can
overshoot a discontinuity, and it is not a VOF method.

```julia
using UniformLWAdvection

s = State(512, 512; u=0.7, v=-0.3)
initialize!(s)                  # exact averages of a smooth periodic fixture
w = Workspace(s; backend=:threads, flux=:recompute, tile=(128,32))
dt = stable_timestep(s)         # starting CFL 0.4
simulate!(s, w, dt, 20)
@show mass(s) translation_error(s)
```

`step!(s,w,dt)` accepts a finite nonnegative timestep with directional
Courant numbers `|u|dt/(2dx) ≤ 1` and `|v|dt/dy ≤ 1`. `stable_timestep`
chooses `0.4/max(|u|/dx, |v|/dy)`; it returns `Inf` for zero flow, for which
a finite caller-specified timestep is appropriate. `initialize!` resets the
field and clock; benchmark drivers should restore an initial state before
each independent trajectory. `translation_error` measures RMS error against
the analytic translated *cell averages*. `mass` is the unit-domain
volume-weighted sum.

`Workspace` supports `backend=:serial`, `:threads`, or `:spawn` without
nonstandard dependencies. The threaded backend uses Julia's dynamic scheduler
over nonoverlapping spatial tiles. The spawned backend creates a bounded
number of tasks, each owning a contiguous group of tiles. `tile=(nx,k)`
implements column slabs; `(ti,tj)` makes two-dimensional tiles. All tile
loops preserve contiguous traversal of the first Julia array index. A
`threshold` (default 16384 cells) triggers the serial path for small work.
For worker-participation evidence, call `track_workers!(w,true)`, run an
**untimed** step, and inspect `active_workers(w)`; disable it before timing.
Each workspace is exclusive to one in-flight step or simulation. Separate
state/workspace pairs can run concurrently.

`backend=:polyester` is an optional weak-dependency extension. Install and
load Polyester in the caller's benchmark environment, then use the same
workspace constructor. The extension requests `per=thread` explicitly so
SMT participation can be measured. The main package does not depend on this
toy or on Polyester through this toy. The extension has no implicit fallback
when Polyester is absent; an unavailable backend throws a clear error.

The numerical-array footprint is `5 × 8nxny` bytes for recomputation and
`6 × 8nxny` bytes with stored flux; `memory_bytes(s,w)` returns this count.
At 4096² these are 640 and 768 MiB, before tile metadata, Julia overhead,
and benchmark copies. Activating a varying carrier allocates two reusable
velocity-refresh arrays, increasing these counts to seven and eight arrays.
Recomputed flux has three sweep barriers per timestep;
stored flux has a flux and gather barrier for each of three sweeps. Actual
DRAM traffic depends on cache reuse and should be measured, not inferred
from array counts alone. Scheduler allocations are bounded by tiles/workers,
and the serial warmed step allocates zero bytes in the current Julia test.

Run numerical tests with the direct Julia binary or your configured launcher:

```sh
julia --startup-file=no --project=research/UniformLWAdvection --threads=1 research/UniformLWAdvection/test/runtests.jl
julia --startup-file=no --project=research/UniformLWAdvection --threads=4 research/UniformLWAdvection/test/runtests.jl
```

The tests compare against a separately written scalar face-flux reference,
including odd periodic seams, one-cell dimensions, positive/negative/zero
speed, repeated steps, flux policies, tile layouts, concurrent independent
workspaces, conservation, constant states, and analytic translation order.
With Polyester available, run `test/polyester.jl` from an environment where
both packages are visible. The driver and Stage 2 acceptance record live
separately in `benchmark/`.

For a variable carrier frozen at the global step midpoint, the predictor is

```math
q^*_{i+1/2}=(q_i+q_{i+1})/2-\tau[(a q)_{i+1}-(a q)_i]/(2h),
\qquad F_{i+1/2}=a_{i+1/2}q^*_{i+1/2},
```

where each cell speed is the arithmetic mean of its two adjacent face
speeds. `Workspace(s; carrier=:variable)` chooses this formula. Call
`set_velocity!(s,uface,vface,s.t+dt/2)` before every time-varying step;
the functions receive physical face coordinates and the midpoint time.
Alternatively pass dense face matrices to `set_velocity!`. The constant
carrier kernel remains a separate specialization, including the unchanged
LW arithmetic. Both setters fill and validate reusable scratch before copying
either face field; a failed callback leaves the current carrier unchanged.
`stable_timestep` for varying carriers uses the largest
sampled face speed. A speed-based CFL alone is not a general stability
theorem for arbitrary coefficients, so the variable path restricts each
directional substep to 0.4 and is validated on smooth manufactured flows.
The tests include a time-dependent shear with analytic cell averages and a
discrete streamfunction with roundoff-level divergence. Directional splitting
does not preserve a constant scalar exactly for the latter; its defect
converges at second order. No unsplit free-stream-preserving variant is claimed.
