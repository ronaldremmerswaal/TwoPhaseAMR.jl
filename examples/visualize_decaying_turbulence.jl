using Random
using TwoPhaseAMR
using GLMakie

GLMakie.activate!()

# A real Fourier basis makes a reproducible, periodic, divergence-free
# continuum velocity. The discrete projection below removes the small
# divergence introduced when it is sampled on a nonconforming staggered mesh.
struct TurbulenceMode
    kx::Int
    ky::Int
    amplitude::Float64
    phase::Float64
end

function turbulence_modes(; seed=20250308, minimum_wavenumber=3,
                          maximum_wavenumber=9, peak_wavenumber=6.0)
    rng = MersenneTwister(seed)
    bandwidth = max(1.0, (maximum_wavenumber - minimum_wavenumber) / 3)
    modes = TurbulenceMode[]
    for kx in 0:maximum_wavenumber,
        ky in -maximum_wavenumber:maximum_wavenumber
        (kx == 0 && ky <= 0) && continue
        wavenumber = hypot(kx, ky)
        minimum_wavenumber <= wavenumber <= maximum_wavenumber || continue
        envelope = exp(-0.5 * ((wavenumber - peak_wavenumber) / bandwidth)^2)
        amplitude = envelope * randn(rng) / wavenumber
        push!(modes, TurbulenceMode(
            kx, ky, amplitude, 2pi * rand(rng)))
    end
    return modes
end

function mode_velocity(modes, x, y, axis)
    value = 0.0
    for mode in modes
        sine = sin(mode.kx * x + mode.ky * y + mode.phase)
        if axis == 1
            value -= mode.amplitude * mode.ky * sine
        else
            value += mode.amplitude * mode.kx * sine
        end
    end
    return value
end

mutable struct TurbulenceDiagnosticCache{T,M}
    step::Int
    xvelocity::FaceField{1,T,M}
    yvelocity::FaceField{2,T,M}
    xworkspace::StaggeredAdvectionWorkspace{1,T,M}
    yworkspace::StaggeredAdvectionWorkspace{2,T,M}
    divergence::CellField{T,M}
    initial_energy::Float64
    initial_enstrophy::Float64
    values::Dict{Symbol,Float64}
end

function TurbulenceDiagnosticCache(velocity)
    xvelocity = FaceField{1}(velocity.mesh)
    yvelocity = FaceField{2}(velocity.mesh)
    return TurbulenceDiagnosticCache{
        eltype(velocity),typeof(velocity.mesh)}(
        typemin(Int), xvelocity, yvelocity,
        StaggeredAdvectionWorkspace(xvelocity),
        StaggeredAdvectionWorkspace(yvelocity),
        CellField(velocity.mesh), NaN, NaN, Dict{Symbol,Float64}())
end

function refresh_diagnostic_cache!(cache, velocity)
    number_of_xfaces = count(face -> face.axis == 1, velocity.mesh.faces)
    number_of_yfaces = count(face -> face.axis == 2, velocity.mesh.faces)
    if length(cache.xvelocity.interior) != number_of_xfaces ||
            length(cache.yvelocity.interior) != number_of_yfaces
        cache.xvelocity = FaceField{1}(velocity.mesh)
        cache.yvelocity = FaceField{2}(velocity.mesh)
        cache.xworkspace = StaggeredAdvectionWorkspace(cache.xvelocity)
        cache.yworkspace = StaggeredAdvectionWorkspace(cache.yvelocity)
        cache.divergence = CellField(velocity.mesh)
    end
    return cache
end

function cell_mean(mesh, values, transform=identity)
    length(values) == ncells(mesh) ||
        throw(DimensionMismatch("one value per owned cell is required"))
    value_index = 0
    integral = 0.0
    for block in mesh.blocks
        for _ in 1:prod(mesh.block_size)
            value_index += 1
            integral += transform(values[value_index]) * block.volume
        end
    end
    return integral / prod(mesh.extent)
end

function kinetic_energy!(cache, velocity)
    copyto!(cache.xvelocity, velocity)
    copyto!(cache.yvelocity, velocity)
    twice_energy =
        staggered_inner_product(
            cache.xvelocity, cache.xvelocity, cache.xworkspace) +
        staggered_inner_product(
            cache.yvelocity, cache.yvelocity, cache.yworkspace)
    return 0.5 * twice_energy / prod(velocity.mesh.extent)
end

function update_diagnostics!(cache, velocity, step)
    step == cache.step && return cache
    refresh_diagnostic_cache!(cache, velocity)
    energy = kinetic_energy!(cache, velocity)
    speed = visualization_values(velocity, CentredMagnitude())
    vorticity = visualization_values(velocity, Vorticity())
    enstrophy = 0.5 * cell_mean(velocity.mesh, vorticity, abs2)
    divergence!(cache.divergence, velocity)
    maximum_divergence = maximum(
        maximum(abs, interior(cache.divergence, block))
        for block in eachindex(velocity.mesh.blocks))

    if !isfinite(cache.initial_energy)
        cache.initial_energy = energy
        cache.initial_enstrophy = enstrophy
    end
    cache.values[:energy_fraction] = energy / cache.initial_energy
    cache.values[:enstrophy_fraction] = enstrophy / cache.initial_enstrophy
    cache.values[:peak_speed] = maximum(speed)
    cache.values[:maximum_divergence] = maximum_divergence
    cache.step = step
    return cache
end

mutable struct TurbulenceAMRHistory
    last_blocks::Int
    finest_uniform_cells::Int
    samples::Vector{NamedTuple}
    verbose::Bool
end

function record_adaptivity!(history, velocity, time, step)
    mesh = velocity.mesh
    blocks = nblocks(mesh)
    active_cell_fraction = ncells(mesh) / history.finest_uniform_cells
    if blocks != history.last_blocks
        levels = sort!(unique(block.key.level for block in mesh.blocks))
        sample = (; time, step, blocks, cells=ncells(mesh), levels,
                  active_cell_fraction)
        push!(history.samples, sample)
        history.verbose && @info(
            "Adaptive turbulence mesh changed", time, step, blocks,
            cells=sample.cells, levels,
            finest_grid_fraction=round(active_cell_fraction; digits=3))
        history.last_blocks = blocks
    end
    return active_cell_fraction
end

function cached_diagnostic(cache, name)
    return function (velocity, pressure, time, step, solver)
        update_diagnostics!(cache, velocity, step)
        return cache.values[name]
    end
end

"""
    visualize_decaying_turbulence(; kwargs...)

Start an interactive, freely decaying 2-D homogeneous-isotropic-turbulence
demonstration with dynamic adaptive refinement. The default mesh starts at a
uniform 128-by-128 equivalent resolution and coarsens where the live pressure
field does not require that resolution. A protected 32-by-32 base level,
update cadence, and cooldown prevent over-coarsening and mesh chatter.
The bulk convection scheme is Lax--Wendroff, with a three-point MC-limited
flux on non-regular refinement-interface connections.

The returned named tuple owns the numerical state, viewer, visualization
session, and running `simulation` task. Use the field menu to compare velocity
components, speed, vorticity, and pressure; click blocks/faces to inspect them;
use **Pause**/**Resume**, checkpoint markers, and **Live** while the task runs.
"""
function visualize_decaying_turbulence(;
        seed=20250308, block_size=8, base_level=2, max_level=4,
        viscosity=2.5e-3, final_time=10, dt=nothing,
        scheme=LW(),
        E_refine=0.6, E_coarsen=0.2,
        update_interval=5, coarsen_delay=25,
        checkpoints=25, live_dt=final_time / 1000,
        diagnostic_every=1, display=true, asynchronous=true, verbose=true)
    0 <= base_level <= max_level ||
        throw(ArgumentError("require 0 <= base_level <= max_level"))
    final_time > 0 || throw(ArgumentError("final_time must be positive"))
    Threads.nthreads() > 1 || @warn(
        "Run this interactive example with julia --threads=auto for a responsive GUI")

    modes = turbulence_modes(; seed)
    mesh = BlockTreeMesh(
        block_size=(block_size, block_size), extent=(2pi, 2pi),
        periodic=(true, true), max_level=max_level,
        refine=(key, origin, extent) -> key.level < max_level,
        adaptation=MeshAdaptationOptions(;
            update_interval, coarsen_delay, minimum_level=base_level))
    initial_blocks = nblocks(mesh)
    initial_cells = ncells(mesh)
    finest_uniform_cells = block_size^2 * 4^max_level
    verbose && @info(
        "Built uniform finest turbulence mesh",
        blocks=initial_blocks, cells=initial_cells,
        levels=sort(unique(block.key.level for block in mesh.blocks)))

    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> mode_velocity(modes, x, y, axis))
    pressure = CellField(mesh)
    refinement = InterpolationErrorIndicator(
        pressure; E_refine, E_coarsen)
    solver = NavierStokesSolver(
        mesh; viscosity, boundary=FreeSlip(), scheme,
        coarse_size=64,
        refinement_indicators=(refinement,))
    workspace = NavierStokesWorkspace(solver)

    verbose && @info "Projecting initial Fourier velocity onto the discrete divergence-free space"
    projection = project!(
        velocity, pressure, solver.pressure_solver, 1.0,
        workspace.projection; rtol=1e-10, max_cycles=120)
    projection.converged ||
        error("initial turbulence projection did not converge")

    diagnostic_cache = TurbulenceDiagnosticCache(velocity)
    initial_energy = kinetic_energy!(diagnostic_cache, velocity)
    rms_velocity = sqrt(2initial_energy)
    scale = 1. / rms_velocity
    velocity.interior .*= scale
    velocity.boundary .*= scale
    initialize!(pressure, (x, y) -> 0.0)

    fields = VisualizationFieldRegistry()
    register_default_fields!(
        fields, :velocity, velocity, :pressure, pressure)
    register_diagnostic!(
        fields, :energy_fraction,
        cached_diagnostic(diagnostic_cache, :energy_fraction);
        label="Kinetic energy / initial", color=:dodgerblue)
    register_diagnostic!(
        fields, :enstrophy_fraction,
        cached_diagnostic(diagnostic_cache, :enstrophy_fraction);
        label="Enstrophy / initial", color=:darkorange)
    register_diagnostic!(
        fields, :peak_speed,
        cached_diagnostic(diagnostic_cache, :peak_speed);
        label="Peak speed", color=:seagreen)
    register_diagnostic!(
        fields, :maximum_divergence,
        cached_diagnostic(diagnostic_cache, :maximum_divergence);
        label="Maximum |div u|", color=:purple)
    adaptivity_history = TurbulenceAMRHistory(
        -1, finest_uniform_cells, NamedTuple[], verbose)
    register_diagnostic!(
        fields, :active_cell_fraction,
        (velocity, pressure, time, step, solver) ->
            record_adaptivity!(adaptivity_history, velocity, time, step);
        label="Active cells / uniform finest", color=:gray35)

    viewer = visualize(
        fields; field=:velocity_vorticity, show_blocks=true,
        show_cells=false, display, figure=(; size=(1200, 900)))
    session = VisualizationSession(
        fields; viewer, checkpoints, live_dt, diagnostic_every,
        asynchronous)
    verbose && @info(
        "Starting decaying-turbulence simulation",
        final_time, checkpoints, live_dt)
    simulation = Threads.@spawn simulate!(
        velocity, pressure, solver, final_time, workspace;
        dt, visualization=session, cfl=0.7, rtol=2e-9, max_cycles=100)

    return (; mesh, modes, velocity, pressure, solver, workspace, fields,
            viewer, session, simulation, initial_projection=projection,
            refinement, adaptivity_history, initial_blocks, initial_cells,
            finest_uniform_cells)
end

function wait_for_turbulence!(run)
    result = fetch(run.simulation)
    wait_visualization!(run.session)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run = visualize_decaying_turbulence()
    println("Started adaptive 2-D decaying turbulence on " *
            "$(run.initial_blocks) blocks and $(run.initial_cells) owned cells.")
    result = wait_for_turbulence!(run)
    fraction = ncells(run.mesh) / run.finest_uniform_cells
    println("Completed $(result.steps) steps and $(result.mesh_updates) mesh " *
            "updates at t=$(result.time) on $(nblocks(run.mesh)) blocks. " *
            "The final mesh uses $(round(100fraction; digits=1))% of the " *
            "uniform finest-grid cells. Close the window to exit.")
    scene = visualization_figure(run.viewer).scene
    scene.events.window_open[] && wait(scene)
end
