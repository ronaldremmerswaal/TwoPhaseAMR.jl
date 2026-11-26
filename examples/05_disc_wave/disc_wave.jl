using TwoPhaseAMR
using LinearAlgebra: dot

Base.include(@__MODULE__, joinpath(@__DIR__, "disc_modes.jl"))

# The square box is [-1.25R, 1.25R]^2; the disc is centred at the origin.
const DISC_WAVE_BOX = 1.25

"""
`resolution²` cells in `8 × 8` blocks on the box `[-1.25R, 1.25R]²`; with
`composite=true` the base blocks that the circle `r = R` crosses are refined once.
"""
function disc_wave_mesh(resolution; radius = 1.0, composite = false)
    resolution isa Integer && resolution >= 16 && iszero(resolution % 8) ||
        throw(ArgumentError("resolution must be a multiple of 8, at least 16"))
    half = DISC_WAVE_BOX * radius
    # The nearest and farthest points of a block from the centre straddle r = R.
    crosses_wall(lower, upper) =
        hypot(max.(lower, 0.0, .-upper)...) <= radius <= hypot(max.(abs.(lower), abs.(upper))...)
    return BlockTreeMesh(;
        base_mesh = (resolution, resolution), block_size = (8, 8),
        origin = (-half, -half), extent = (2half, 2half), periodic = (false, false),
        max_level = composite ? 1 : 0,
        refine = (key, origin, extent) ->
        composite && key.level == 0 && crosses_wall(origin, origin .+ extent),
    )
end

"""
Initial displacement from the discrete Helmholtz problem `(I - h²Δ_h) u₀ =
(I - h²Δ) u(·, 0)`; exact modes are eigenfunctions, so the source is
`Σ (1 + h²k²) φ_k` at the `value_points` of the operator. Sampled values would
excite the stiff modes of small cut cells, which the scheme never damps.
"""
function project_initial_displacement!(displacement, operator, geometry, modes; radius, tolerance)
    shift = (geometry.mesh.extent[1] / geometry.mesh.base_mesh[1])^2
    source = FaceVelocity(geometry.mesh)
    for axis in 1:2
        nodes = value_points(operator, axis)
        for (index, face) in pairs(nodes.faces)
            iszero(nodes.volumes[index]) && continue
            x, y = nodes.points[index]
            source.interior[face] = sum(
                mode -> mode.axis == axis ?
                    (1 + shift * (mode.root / radius)^2) * mode_shape(mode, x, y, radius) : 0.0,
                modes,
            )
        end
    end
    projector = StaggeredEllipticSolver(
        StaggeredEllipticOperator(operator; reaction = 1, diffusion = shift);
        preconditioner_options = (; geometry),
    )
    copyto!(displacement.interior, source.interior)
    solve!(displacement, projector, source; tolerance, max_cycles = 500).converged ||
        error("the initial Helmholtz projection did not converge")
    return displacement
end

"""
One implicit-midpoint step of `u_t = v`, `v_t = c²Δ_h u`: the midpoint
displacement ū solves `(I - (cΔt/2)² Δ_h) ū = u + (Δt/2) v`, then `u ← 2ū - u`
and `v ← (4/Δt)(ū - u) - v`. Returns the conjugate-gradient iterations.
"""
function implicit_midpoint_step!(displacement, rate, midpoint, source, solver, dt, tolerance)
    @. source.interior = displacement.interior + dt / 2 * rate.interior
    copyto!(midpoint.interior, source.interior)
    stats = solve!(midpoint, solver, source; tolerance, max_cycles = 200)
    stats.converged || error("the implicit-midpoint solve did not converge: $stats")
    @. rate.interior = 4 / dt * (midpoint.interior - displacement.interior) - rate.interior
    @. displacement.interior = 2midpoint.interior - displacement.interior
    return stats.x.cycles + stats.y.cycles
end

"""Discrete energy `½ Σ (vᵀVv + c² uᵀKu)` of both components, `V⁻¹K = -Δ_h`."""
function wave_energy(operator, displacement, rate, speed)
    total = 0.0
    for component in (operator.x, operator.y)
        values = displacement.interior[component.local_to_global]
        kinetic = 0.0
        for (index, face) in pairs(component.local_to_global)
            kinetic += component.volumes[index] * rate.interior[face]^2
        end
        total += kinetic + speed^2 * dot(values, component.stiffness * values)
    end
    return total / 2
end

"""
    run_example(; resolution=128, composite=false, closure=LSStagClosure(), modes=DISC_WAVE_MODES,
                radius=1, speed=1, final_time=nothing, steps=128, tolerance=1e-10, kwargs...)

Solve `u_tt = c²Δu` for both components of a `FaceVelocity` on the disc `r < radius`
(solid outside, `NoSlip()` on the circle) from `modes` at rest: `StaggeredDiffusionOperator`
as `Δ_h`, implicit-midpoint steps `dt = final_time/steps` (default: one period of the
slowest mode). `on_step(state)` sees the initial state and every step; `start_studio=studio`
(after loading TwoPhaseAMRStudio) follows the run live.
"""
function run_example(;
        resolution = 128, composite = false, closure = LSStagClosure(),
        modes = DISC_WAVE_MODES, radius = 1.0, speed = 1.0, final_time = nothing,
        steps = 128, tolerance = 1.0e-10, initial_displacement! = project_initial_displacement!,
        on_step = nothing, keyframes = EquispacedTimes(65), recorder_options = (;),
        start_studio = nothing,
    )
    steps isa Integer && steps >= 1 || throw(ArgumentError("steps must be a positive integer"))
    final_time = something(final_time, 2pi / minimum(mode -> mode_frequency(mode; radius, speed), modes))
    dt = final_time / steps
    started = time()
    mesh = disc_wave_mesh(resolution; radius, composite)
    geometry = GeometryField(mesh, (x, y) -> radius - hypot(x, y); lipschitz = 1)
    operator = StaggeredDiffusionOperator(geometry; boundary = NoSlip(), closure)
    solver = StaggeredEllipticSolver(
        StaggeredEllipticOperator(operator; reaction = 1, diffusion = (speed * dt / 2)^2);
        preconditioner_options = (; geometry),
    )
    displacement, rate, midpoint, source = (FaceVelocity(mesh) for _ in 1:4)
    initial_displacement!(displacement, operator, geometry, modes; radius, tolerance)
    energy = [wave_energy(operator, displacement, rate, speed)]
    report(step, time) = isnothing(on_step) ||
        on_step((; step, time, displacement, rate, operator, modes, radius, speed))
    report(0, 0.0)
    fields = VisualizationFieldRegistry()
    register_geometry!(register_default_fields!(fields, :displacement, displacement), geometry)
    preview = isnothing(start_studio) ? nothing : EveryWallInterval(0.2)
    recorder = Recorder(fields; merge((; keyframes, preview), recorder_options)...)
    setup_seconds = time() - started
    progress = Ref((; step = 0, time = 0.0, iterations = 0))
    function advance!(target, observer)
        (; step, time, iterations) = progress[]
        state() = SimulationState(displacement, nothing, nothing, time, step)
        info(exception = nothing) = StepInfo(; dt, final_time = target, exception)
        observe!(observer, :initial, state(), info())
        try
            while (step + 1) * dt <= target + dt / 2 && !stop_requested(observer)
                iterations += implicit_midpoint_step!(displacement, rate, midpoint, source, solver, dt, tolerance)
                step += 1
                # The last step lands on the target up to round-off.
                time = abs(step * dt - target) <= dt / 2 ? target : step * dt
                push!(energy, wave_energy(operator, displacement, rate, speed))
                report(step, time)
                observe!(observer, :step, state(), info())
            end
        catch exception
            failure = exception isa Exception ? exception : ErrorException(sprint(showerror, exception))
            observe!(observer, :failed, state(), info(failure))
            rethrow()
        end
        observe!(observer, :completed, state(), info())
        progress[] = (; step, time, iterations)
        return (; time, steps = step, stopped = stop_requested(observer))
    end
    driver = isnothing(start_studio) ? nothing : TwoPhaseAMR.StudioDriver(advance!, recorder)
    app = isnothing(start_studio) ? nothing : start_studio(recorder)
    run_seconds = @elapsed result = isnothing(driver) ? advance!(final_time, recorder) : simulate!(driver, final_time)
    return (;
        mesh, geometry, operator, displacement, rate, modes, radius, speed, dt, result.steps,
        result.time, energy, progress[].iterations, setup_seconds, run_seconds, recorder, app,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--studio" in ARGS
        using TwoPhaseAMRStudio
        run = run_example(; start_studio = studio)
    else
        run = run_example()
    end
    println(
        run.steps, " steps of ", run.dt, " to t = ", run.time, "; largest relative energy change ",
        maximum(abs, run.energy ./ first(run.energy) .- 1), "; ", run.iterations,
        " CG iterations; setup ", run.setup_seconds, " s, stepping ", run.run_seconds, " s",
    )
    isnothing(run.app) || try
        wait(run.app)
    finally
        close(run.app)
    end
end
