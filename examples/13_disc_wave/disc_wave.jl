using TwoPhaseAMR
using LinearAlgebra: dot, mul!
using SparseArrays: SparseMatrixCSC
using SpecialFunctions: besselj

"""
    bessel_zero(m, k)

The `k`-th positive zero `j_{m,k}` of the Bessel function `J_m`. Sign changes
are bracketed on a grid of spacing `1/20`, starting where `J_m` is still
positive (consecutive zeros are more than two apart), and bisected to
round-off.
"""
function bessel_zero(m::Integer, k::Integer)
    m >= 0 && k >= 1 || throw(ArgumentError("bessel_zero needs m >= 0 and k >= 1"))
    found = 0
    left = max(float(m), 1 / 20)
    left_value = besselj(m, left)
    while true
        right = left + 1 / 20
        right_value = besselj(m, right)
        if iszero(right_value) || signbit(right_value) != signbit(left_value)
            found += 1
            if found == k
                iszero(right_value) && return right
                while true
                    middle = (left + right) / 2
                    left < middle < right || return middle
                    middle_value = besselj(m, middle)
                    iszero(middle_value) && return middle
                    if signbit(middle_value) == signbit(left_value)
                        left, left_value = middle, middle_value
                    else
                        right = middle
                    end
                end
            end
        end
        left, left_value = right, right_value
    end
    return
end

"""
    DiscMode(axis, m, k; sine=false, amplitude=1)

A standing Dirichlet eigenmode of the displacement component `axis` (`1` for
`x`, `2` for `y`) on the disc `r < R`, in polar coordinates about its centre:

    amplitude * J_m(j_{m,k} r/R) * cos(mθ) * cos(ωt),    ω = c j_{m,k}/R,

with `sin(mθ)` for `sine=true`. It vanishes on the circle and solves the
wave equation `u_tt = c²Δu` for all time.
"""
struct DiscMode
    axis::Int
    m::Int
    k::Int
    sine::Bool
    amplitude::Float64
    root::Float64
end

function DiscMode(axis, m, k; sine = false, amplitude = 1.0)
    axis in (1, 2) || throw(ArgumentError("a disc mode belongs to axis 1 or 2"))
    return DiscMode(axis, m, k, sine, amplitude, bessel_zero(m, k))
end

# u_x: the axisymmetric fundamental (0,1) and the (2,1) mode; u_y: the (1,1)
# mode and the (3,2) mode, which has a nodal circle.
const DISC_WAVE_MODES = (
    DiscMode(1, 0, 1), DiscMode(1, 2, 1; amplitude = 0.6),
    DiscMode(2, 1, 1; sine = true), DiscMode(2, 3, 2; amplitude = 0.8),
)

# The square box is [-1.25R, 1.25R]^2; the disc is centred at the origin.
const DISC_WAVE_BOX = 1.25

"""Angular frequency `c j_{m,k}/R` of `mode`."""
mode_frequency(mode::DiscMode; radius = 1.0, speed = 1.0) = speed * mode.root / radius

function mode_shape(mode::DiscMode, x, y, radius)
    angle = atan(y, x)
    angular = mode.sine ? sin(mode.m * angle) : cos(mode.m * angle)
    return mode.amplitude * besselj(mode.m, mode.root * hypot(x, y) / radius) * angular
end

"""
    disc_wave_exact(modes, x, y, time, axis; radius=1, speed=1)

Exact displacement of component `axis` and its rate `∂u/∂t` at `(x, y)`.
"""
function disc_wave_exact(modes, x, y, time, axis; radius = 1.0, speed = 1.0)
    displacement = 0.0
    rate = 0.0
    for mode in modes
        mode.axis == axis || continue
        omega = mode_frequency(mode; radius, speed)
        shape = mode_shape(mode, x, y, radius)
        displacement += shape * cos(omega * time)
        rate -= omega * shape * sin(omega * time)
    end
    return displacement, rate
end

"""
    disc_wave_mesh(resolution; radius=1, composite=false)

`resolution × resolution` cells in `8 × 8` blocks on the box
`[-1.25R, 1.25R]²`. With `composite=true` every base block that the circle
`r = R` crosses is refined once, a static band of half-width cells along the
wall (the tree's 2:1 balance is automatic).
"""
function disc_wave_mesh(resolution; radius = 1.0, composite = false)
    resolution isa Integer && resolution >= 16 && iszero(resolution % 8) ||
        throw(ArgumentError("resolution must be a multiple of 8, at least 16"))
    half = DISC_WAVE_BOX * radius
    function crosses_wall(origin, extent)
        lower, upper = origin, origin .+ extent
        nearest = hypot((max(lower[axis], 0.0, -upper[axis]) for axis in 1:2)...)
        farthest = hypot((max(abs(lower[axis]), abs(upper[axis])) for axis in 1:2)...)
        return nearest <= radius <= farthest
    end
    return BlockTreeMesh(;
        base_mesh = (resolution, resolution), block_size = (8, 8),
        origin = (-half, -half), extent = (2half, 2half), periodic = (false, false),
        max_level = composite ? 1 : 0,
        refine = (key, origin, extent) ->
        composite && key.level == 0 && crosses_wall(origin, extent),
    )
end

"""
    disc_wave_value_points(geometry, component, closure)

Where the unknowns of one velocity component of the viscous operator sit, and
the open (fluid) dual volume of each, zero for a covered face.
`LSStagClosure()` places the value of a cut face at the midpoint of its open
part, which lies in the fluid; `WallDistanceClosure()` keeps it at the face
centre, which may lie in the solid.
"""
function disc_wave_value_points(geometry, component, closure)
    mesh = geometry.mesh
    points = Vector{NTuple{2, Float64}}(undef, length(component.local_to_global))
    weights = zeros(length(component.local_to_global))
    for (index, face_index) in pairs(component.local_to_global)
        face = mesh.faces[face_index]
        aperture = geometry.aperture[face_index]
        points[index] = face.center
        aperture > 0 || continue
        weights[index] = geometry.face_dual_volumes[face_index]
        if closure isa LSStagClosure && aperture < 1
            _, _, lower, upper, _, _ =
                TwoPhaseAMR._ls_stag_open_part(geometry, face, aperture)
            middle = (lower + upper) / 2
            points[index] = face.axis == 1 ? (face.center[1], middle) :
                (middle, face.center[2])
        end
    end
    return points, weights
end

# One velocity component: its unknowns, the operator's mass `V` and symmetric
# positive semidefinite stiffness `K` (`V⁻¹K = -Δ_h`), the value points and
# the open dual volumes that weigh the error norms and the modal projections.
struct DiscWaveComponent
    faces::Vector{Int}
    volumes::Vector{Float64}
    stiffness::SparseMatrixCSC{Float64, Int}
    points::Vector{NTuple{2, Float64}}
    weights::Vector{Float64}
    values::Vector{Float64}
    product::Vector{Float64}
end

function DiscWaveComponent(geometry, component, closure)
    points, weights = disc_wave_value_points(geometry, component, closure)
    unknowns = length(component.local_to_global)
    return DiscWaveComponent(
        component.local_to_global, component.volumes, component.stiffness,
        points, weights, zeros(unknowns), zeros(unknowns),
    )
end

"""Discrete energy `½ Σ (vᵀVv + c² uᵀKu)` over both components."""
function disc_wave_energy(components, displacement, rate, speed)
    total = 0.0
    for component in components
        kinetic = 0.0
        for (index, face) in pairs(component.faces)
            component.values[index] = displacement.interior[face]
            kinetic += component.volumes[index] * rate.interior[face]^2
        end
        mul!(component.product, component.stiffness, component.values)
        total += kinetic + speed^2 * dot(component.values, component.product)
    end
    return total / 2
end

"""
Volume-weighted RMS (`l2`) and maximum (`max`) errors of the displacement and
its rate over the fluid unknowns, at their value points. `norms` holds the same
norms of the exact solution, for relative errors.
"""
function disc_wave_errors(components, displacement, rate, modes, time; radius, speed)
    squares = zeros(4)
    maxima = zeros(4)
    volume = 0.0
    for (axis, component) in pairs(components)
        for (index, face) in pairs(component.faces)
            weight = component.weights[index]
            iszero(weight) && continue
            x, y = component.points[index]
            exact = disc_wave_exact(modes, x, y, time, axis; radius, speed)
            values = (
                displacement.interior[face] - exact[1], rate.interior[face] - exact[2],
                exact[1], exact[2],
            )
            for (slot, value) in pairs(values)
                squares[slot] += weight * value^2
                maxima[slot] = max(maxima[slot], abs(value))
            end
            volume += weight
        end
    end
    l2 = sqrt.(squares ./ volume)
    return (;
        displacement_l2 = l2[1], displacement_max = maxima[1],
        rate_l2 = l2[2], rate_max = maxima[2],
        norms = (;
            displacement_l2 = l2[3], displacement_max = maxima[3],
            rate_l2 = l2[4], rate_max = maxima[4],
        ),
    )
end

"""Error `name` of `errors` relative to the same norm of the exact solution."""
relative_error(errors, name::Symbol) = getproperty(errors, name) / getproperty(errors.norms, name)

# Sample the exact initial state, or (`:projected`) solve the discrete
# Helmholtz problem `(I - h²Δ_h) u₀ = (I - h²Δ) u(0)` whose exact solution is
# the initial displacement: exact modes are eigenfunctions, so its source is
# `Σ (1 + h²k²) φ_k`. Samples excite the stiff modes of small cut cells,
# which an energy-conserving scheme never damps.
function _disc_wave_initial_state!(
        displacement, rate, components, modes, operator, geometry;
        initial_data, radius, speed, tolerance,
    )
    initial_data in (:projected, :sampled) ||
        throw(ArgumentError("initial_data must be :projected or :sampled"))
    shift = (geometry.mesh.extent[1] / geometry.mesh.base_mesh[1])^2
    source = initial_data === :sampled ? displacement : FaceVelocity(geometry.mesh)
    fill!(displacement.interior, 0.0)
    fill!(rate.interior, 0.0)
    for (axis, component) in pairs(components)
        for (index, face) in pairs(component.faces)
            iszero(component.weights[index]) && continue
            x, y = component.points[index]
            displacement_value, rate_value =
                disc_wave_exact(modes, x, y, 0.0, axis; radius, speed)
            rate.interior[face] = rate_value
            initial_data === :sampled && (displacement.interior[face] = displacement_value)
            initial_data === :projected && (
                source.interior[face] = sum(
                    mode -> mode.axis == axis ?
                        (1 + shift * (mode.root / radius)^2) *
                        mode_shape(mode, x, y, radius) : 0.0,
                    modes,
                )
            )
        end
    end
    initial_data === :sampled && return nothing
    projector = StaggeredEllipticSolver(
        StaggeredEllipticOperator(operator; reaction = 1, diffusion = shift);
        preconditioner_options = (; geometry),
    )
    copyto!(displacement.interior, source.interior)
    stats = solve!(displacement, projector, source; tolerance, max_cycles = 500)
    stats.converged || error("the initial Helmholtz projection did not converge")
    return nothing
end

# Phase of each mode: the projections a, b of the displacement and its rate
# on the exact shape φ give `atan(-b/ω, a) = ωt` for the exact solution.
function _disc_wave_phases!(phases, components, displacement, rate, shapes, modes, omegas)
    for (index, mode) in pairs(modes)
        component = components[mode.axis]
        shape = shapes[index]
        projected_displacement = 0.0
        projected_rate = 0.0
        for (local_index, face) in pairs(component.faces)
            weight = component.weights[local_index] * shape[local_index]
            projected_displacement += weight * displacement.interior[face]
            projected_rate += weight * rate.interior[face]
        end
        phase = atan(-projected_rate / omegas[index], projected_displacement)
        # Unwrap: steps advance each phase by less than π.
        phases[index] += rem2pi(phase - phases[index], RoundNearest)
    end
    return phases
end

# One implicit-midpoint step of u_t = v, v_t = c²Δ_h u: the midpoint
# displacement ū solves (I - (c dt/2)² Δ_h) ū = u + (dt/2) v, then
# u ← 2ū - u and v ← (4/dt)(ū - u) - v.
function _disc_wave_step!(displacement, rate, midpoint, source, solver, dt, tolerance)
    for index in eachindex(source.interior)
        @inbounds source.interior[index] =
            displacement.interior[index] + dt / 2 * rate.interior[index]
        @inbounds midpoint.interior[index] = source.interior[index]
    end
    stats = solve!(midpoint, solver, source; tolerance, max_cycles = 200)
    stats.converged || error("the implicit-midpoint solve did not converge: $stats")
    for index in eachindex(displacement.interior)
        @inbounds begin
            old = displacement.interior[index]
            displacement.interior[index] = 2midpoint.interior[index] - old
            rate.interior[index] = 4 / dt * (midpoint.interior[index] - old) -
                rate.interior[index]
        end
    end
    return stats.x.cycles + stats.y.cycles
end

"""
    run_example(; resolution=128, composite=false, closure=LSStagClosure(),
                modes=DISC_WAVE_MODES, radius=1, speed=1, final_time=nothing,
                steps=128, tolerance=1e-10, initial_data=:projected,
                keyframes=EquispacedTimes(65), recorder_options=(;))

Solve the staggered wave equation `u_tt = c²Δu` for both components of a
`FaceVelocity` `u` on the disc `r < radius` (fluid) inside the box
`[-1.25R, 1.25R]²` (solid outside the disc, `NoSlip()` on the circle), with
`StaggeredDiffusionOperator(geometry; closure)` as the Laplacian. Each
component then solves the scalar Dirichlet problem, and `modes` (by default
two per component) are exact standing waves for all time.

The time integrator is the implicit midpoint rule for `u_t = v`,
`v_t = c²Δ_h u` (equivalently Newmark's average acceleration): one shifted
solve `(I - (c dt/2)² Δ_h) ū = u + dt v/2` per step with the
geometry-aware multigrid-preconditioned conjugate gradients of
[`StaggeredEllipticSolver`](@ref), `dt = final_time/steps`. It conserves the
discrete energy `½(vᵀVv + c² uᵀKu)` up to the solver `tolerance` for any `dt`.
`final_time=nothing` runs one period of the slowest mode.

`initial_data=:projected` starts from the discrete Helmholtz solution with
the exact initial displacement as solution (shift `h²`); `:sampled` samples
the modes at the value points. `keyframes` and `recorder_options` configure
the [`Recorder`](@ref) of the displacement.

Returns `(; geometry, operator, displacement, rate, modes, dt, steps, time,
errors, initial_errors, times, energy, frequencies, explicit_limit, cycles,
maximum_cycles, setup_seconds, run_seconds, recorder)`. `frequencies` holds
for each mode the exact `ω`, the measured `discrete` frequency (slope of the
unwrapped modal phase) and its `spatial` part `(2/dt) tan(Ω dt/2)` without
the implicit-midpoint dispersion. `explicit_limit` is leapfrog's stability
limit `2/(c√ρ)`, `ρ` the largest absolute row sum of `Δ_h`.
"""
function run_example(;
        resolution = 128, composite = false, closure = LSStagClosure(),
        modes = DISC_WAVE_MODES, radius = 1.0, speed = 1.0, final_time = nothing,
        steps = 128, tolerance = 1.0e-10, initial_data = :projected,
        keyframes = EquispacedTimes(65), recorder_options = (;),
    )
    steps isa Integer && steps >= 1 || throw(ArgumentError("steps must be a positive integer"))
    omegas = [mode_frequency(mode; radius, speed) for mode in modes]
    final_time = something(final_time, 2pi / minimum(omegas))
    dt = final_time / steps
    started = time()
    mesh = disc_wave_mesh(resolution; radius, composite)
    geometry = GeometryField(mesh, (x, y) -> radius - hypot(x, y); lipschitz = 1)
    operator = StaggeredDiffusionOperator(geometry; boundary = NoSlip(), closure)
    solver = StaggeredEllipticSolver(
        StaggeredEllipticOperator(operator; reaction = 1, diffusion = (speed * dt / 2)^2);
        preconditioner_options = (; geometry),
    )
    components = (
        DiscWaveComponent(geometry, operator.x, closure),
        DiscWaveComponent(geometry, operator.y, closure),
    )
    displacement = FaceVelocity(mesh)
    rate = FaceVelocity(mesh)
    midpoint = FaceVelocity(mesh)
    source = FaceVelocity(mesh)
    _disc_wave_initial_state!(
        displacement, rate, components, modes, operator, geometry;
        initial_data, radius, speed, tolerance,
    )
    initial_errors = disc_wave_errors(components, displacement, rate, modes, 0.0; radius, speed)
    shapes = [
        [mode_shape(mode, point..., radius) for point in components[mode.axis].points]
            for mode in modes
    ]
    phases = zeros(length(modes))
    _disc_wave_phases!(phases, components, displacement, rate, shapes, modes, omegas)
    phase_history = zeros(length(modes), steps + 1)
    phase_history[:, 1] .= phases
    energy = zeros(steps + 1)
    energy[1] = disc_wave_energy(components, displacement, rate, speed)

    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, :displacement, displacement)
    register_geometry!(fields, geometry)
    recorder = Recorder(fields; merge((; keyframes, preview = nothing), recorder_options)...)
    state(time, step) = SimulationState(displacement, nothing, nothing, time, step)
    setup_seconds = time() - started

    started = time()
    cycles = 0
    maximum_cycles = 0
    observe!(recorder, :initial, state(0.0, 0), StepInfo(; dt, final_time))
    for step in 1:steps
        step_cycles = _disc_wave_step!(displacement, rate, midpoint, source, solver, dt, tolerance)
        cycles += step_cycles
        maximum_cycles = max(maximum_cycles, step_cycles)
        _disc_wave_phases!(phases, components, displacement, rate, shapes, modes, omegas)
        phase_history[:, step + 1] .= phases
        energy[step + 1] = disc_wave_energy(components, displacement, rate, speed)
        observe!(recorder, :step, state(step * dt, step), StepInfo(; dt, final_time))
    end
    observe!(recorder, :completed, state(final_time, steps), StepInfo(; dt, final_time))
    run_seconds = time() - started

    times = collect(range(0.0, final_time; length = steps + 1))
    centred = times .- sum(times) / length(times)
    # Least-squares slope of each unwrapped phase.
    frequencies = ntuple(length(modes)) do index
        discrete = sum(centred .* phase_history[index, :]) / sum(abs2, centred)
        spatial = 2 / dt * tan(discrete * dt / 2)
        (; mode = modes[index], exact = omegas[index], discrete, spatial)
    end
    return (;
        geometry, operator, displacement, rate, modes, dt, steps, time = final_time,
        errors = disc_wave_errors(components, displacement, rate, modes, final_time; radius, speed),
        initial_errors, times, energy, frequencies,
        explicit_limit = 2sqrt(stable_timestep(operator; cfl = 1)) / speed,
        cycles, maximum_cycles, setup_seconds, run_seconds, recorder,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run = run_example()
    drift = maximum(abs, run.energy ./ first(run.energy) .- 1)
    println(
        "dt = ", run.dt, " (", run.dt / run.explicit_limit,
        " times the explicit leapfrog limit ", run.explicit_limit, "), ",
        run.steps, " steps to t = ", run.time,
    )
    println("errors at t = ", run.time, ": ", run.errors)
    println(
        "relative errors (u L2, u max, ∂u/∂t L2, ∂u/∂t max): ",
        relative_error.(Ref(run.errors), (:displacement_l2, :displacement_max, :rate_l2, :rate_max)),
    )
    println("largest relative energy change: ", drift)
    for entry in run.frequencies
        println(
            "mode (", entry.mode.m, ",", entry.mode.k, ") of u_", "xy"[entry.mode.axis],
            ": relative frequency error ", entry.discrete / entry.exact - 1,
            " (spatial part ", entry.spatial / entry.exact - 1, ")",
        )
    end
    println(
        "CG iterations: ", run.cycles, " (at most ", run.maximum_cycles,
        " per step, both components); setup ", run.setup_seconds,
        " s, time stepping ", run.run_seconds, " s",
    )
end
