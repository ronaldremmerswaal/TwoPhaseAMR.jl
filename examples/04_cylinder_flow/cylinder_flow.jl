using TwoPhaseAMR

"""
    CutCellPressureForce(geometry)

Pressure force on the fixed solid's wall segments, for the drag and lift
diagnostics. `cache(state, axis)` returns `2F_axis` (the coefficient `C = 2F` for
`rho = U = D = 1`) at the state's `pressure`; `refresh!(cache)` collects the cut
cells again. The wall segments are collected once per
`TwoPhaseAMR._interface_geometry_revision(geometry)` and reused while it is
unchanged, so a step visits the cut cells only instead of searching every cell;
a regrid refreshes the geometry and an `initialize!` or `reconstruct!` of
`geometry.solid` rebuilds its planes, and both change that revision. The first,
impulsive-start sample (`time == 0`) is `NaN`: it is not a developed force.
"""
struct CutCellPressureForce{G, T}
    geometry::G
    # (cell, per-axis normal, its norm, wall segment length in physical units)
    cut_cells::Vector{Tuple{CellRef, NTuple{2, T}, T, T}}
    revision::Base.RefValue{UInt}
end

function CutCellPressureForce(geometry)
    T = eltype(geometry.solid)
    return refresh!(
        CutCellPressureForce(
            geometry, Tuple{CellRef, NTuple{2, T}, T, T}[], Ref(typemax(UInt)),
        ),
    )
end

"""Collect the cut cells' wall segments of `force.geometry`, in cell order."""
function refresh!(force::CutCellPressureForce{G, T}) where {G, T}
    mesh, solid, cut_cells = force.geometry.mesh, force.geometry.solid, force.cut_cells
    empty!(cut_cells)
    nx, ny = mesh.block_size
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(block, i, j)
        segment = TwoPhaseAMR._local_plic_segment(solid, ref)
        isnothing(segment) && continue
        (first, last), spacing = segment, mesh.blocks[block].spacing
        segment_length = hypot(
            (last[1] - first[1]) * spacing[1],
            (last[2] - first[2]) * spacing[2],
        )
        normal = (
            TwoPhaseAMR._get(solid.normal_x, ref) / spacing[1],
            TwoPhaseAMR._get(solid.normal_y, ref) / spacing[2],
        )
        scale = hypot(normal...)
        iszero(scale) || push!(cut_cells, (ref, normal, scale, segment_length))
    end
    force.revision[] = TwoPhaseAMR._interface_geometry_revision(force.geometry)
    return force
end

function (force::CutCellPressureForce{G, T})(state, axis) where {G, T}
    iszero(state.time) && return T(NaN)
    force.revision[] ==
        TwoPhaseAMR._interface_geometry_revision(force.geometry) || refresh!(force)
    total = zero(T)
    for (ref, normal, scale, segment_length) in force.cut_cells
        total -= TwoPhaseAMR._get(state.pressure, ref) * normal[axis] / scale *
            segment_length
    end
    return 2total
end

"""
    run_example(; reynolds=100, max_level=3, final_time=40, perturbation=0,
                start_studio=nothing, recorder_options=(;), kwargs...)

Run nondimensional flow around a unit cylinder in a unit free stream.
`domain_origin`, `domain_extent`, and `base_mesh` control the computational
domain and its base grid; `vorticity_tolerance` controls wake refinement.
`perturbation` sets the strength of a localized, divergence-free initial
vortex that breaks wake symmetry without changing the boundary inflow.
Pass `start_studio=studio` after loading TwoPhaseAMRStudio to follow the run
live. `recorder_options` overrides the recording schedule; other keywords are
passed to [`FlowSimulation`](@ref).
"""
function run_example(;
        reynolds = 100.0,
        max_level = 3,
        final_time = 40.0,
        perturbation = 0.0,
        domain_origin = (-4.0, -4.0),
        domain_extent = (14.0, 8.0),
        base_mesh = (28, 16),
        vorticity_tolerance = 0.2,
        start_studio = nothing,
        recorder_options = (;),
        kwargs...,
    )
    base_level = min(1, max_level)
    mesh = BlockTreeMesh(
        base_mesh = base_mesh,
        block_size = (4, 4), origin = domain_origin,
        extent = domain_extent, periodic = (false, false),
        max_level = max_level,
        refine = (key, origin, extent) -> key.level < base_level,
        adaptation = MeshAdaptationOptions(
            minimum_level = base_level, update_interval = 5,
            coarsen_delay = 5,
        ),
    )
    cylinder = (x, y) -> hypot(x, y) - 0.5
    geometry = GeometryField(
        mesh, cylinder; lipschitz = 1, atol = 1.0e-7, maxdepth = 24,
    )
    inflow = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0
    velocity = FaceVelocity(mesh, inflow)
    refinement = (
        GeometryRefinementIndicator(geometry; level = max_level),
        VorticityRefinementIndicator(
            velocity; refine_tolerance = vorticity_tolerance,
            coarsen_tolerance = vorticity_tolerance / 3, maximum_level = max_level,
        ),
    )
    adaptation = AMRWorkspace(mesh, geometry, velocity)
    for pass in 1:(max_level - base_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(geometry.solid, cylinder; lipschitz = 1)
        refresh_geometry!(geometry)
        set_velocities!(velocity, inflow)
    end
    if !iszero(perturbation)
        # Localized divergence-free initial vortex breaks the symmetric wake.
        # The boundary remains the unperturbed unit inflow.
        vortex(x, y, time, axis) = begin
            dx, dy = x - 1.0, y - 0.5
            radius2 = 0.5^2
            factor = perturbation / radius2 * exp(-(dx^2 + dy^2) / (2radius2))
            axis == 1 ? 1.0 - dy * factor : dx * factor
        end
        set_velocities!(velocity, vortex)
    end


    boundary = VelocityBoundaryConditions(
        left = Inflow(inflow), right = Outflow(),
        bottom = ZeroGradientVelocity(), top = ZeroGradientVelocity(),
    )
    physics = SinglePhase(
        density = 1.0, dynamic_viscosity = inv(reynolds),
    )
    flow_options = (;
        velocity = velocity, geometry, boundary,
        diagnostics = (
            KineticEnergy(), LinearMomentum(), DivergenceNorm(),
            LeafBlockCount(),
        ),
        adaptivity = Adaptivity(
            criteria = max_level > base_level ? refinement : (),
        ),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    pressure_force = CutCellPressureForce(geometry)
    add_diagnostic!(flow, :pressure_drag, state -> pressure_force(state, 1))
    add_diagnostic!(flow, :pressure_lift, state -> pressure_force(state, 2))
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
    register_geometry!(fields, geometry)
    recorder = Recorder(
        fields; diagnostics = flow,
        merge(
            (;
                keyframes = EquispacedTimes(21),
                preview = isnothing(start_studio) ? nothing : EveryWallInterval(0.2),
            ),
            recorder_options,
        )...,
    )
    driver = isnothing(start_studio) ? nothing : TwoPhaseAMR.StudioDriver(flow, recorder)
    app = isnothing(start_studio) ? nothing : start_studio(recorder)
    result = try
        isnothing(driver) ? simulate!(flow, final_time; observer = recorder) :
            simulate!(driver, final_time)
    catch
        isnothing(app) || close(app)
        rethrow()
    end
    return (; flow, result, recorder, app)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--studio" in ARGS
        using TwoPhaseAMRStudio
        run = run_example(; start_studio = studio)
    else
        run = run_example()
    end
    print_simulation_summary(run.flow, run.result)
    println(snapshot_store(run.recorder))
    if !isnothing(run.app)
        try
            wait(run.app)
        finally
            close(run.app)
        end
    end
end
