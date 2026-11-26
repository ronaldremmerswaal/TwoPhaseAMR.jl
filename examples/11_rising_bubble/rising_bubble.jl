using TwoPhaseAMR

"""
    hysing_parameters(case)

Material parameters of test case `1` or `2` of Hysing et al. (2009): the
surrounding liquid comes first in `density` and `dynamic_viscosity`, the
bubble second. Both cases have Reynolds number 35; the Eötvös number is 10 for
case 1 and 125 for case 2.
"""
function hysing_parameters(case)
    case == 1 && return (;
        density = (1000.0, 100.0), dynamic_viscosity = (10.0, 1.0),
        surface_tension = 24.5,
    )
    case == 2 && return (;
        density = (1000.0, 1.0), dynamic_viscosity = (10.0, 0.1),
        surface_tension = 1.96,
    )
    throw(ArgumentError("case must be 1 or 2"))
end

# The staggered liquid fraction of an interior face: the spacing-weighted mean
# of its two cells, which also decides where a two-velocity phase value exists.
function _face_liquid_fraction(interface, face)
    mesh = interface.mesh
    minus, plus = face.minus, face.plus
    hm = mesh.blocks[minus.block].spacing[face.axis]
    hp = mesh.blocks[plus.block].spacing[face.axis]
    fraction = (
        hm * interior(interface, minus.block)[minus.i, minus.j] +
            hp * interior(interface, plus.block)[plus.i, plus.j]
    ) / (hm + hp)
    return clamp(fraction, zero(fraction), one(fraction))
end

"""
    bubble_centroid(interface)

Height of the bubble's centre of mass, `∫ y dx / ∫ 1 dx` over the gas phase,
with each cell's gas fraction at its centre.
"""
function bubble_centroid(interface)
    mesh = interface.mesh
    nx, ny = mesh.block_size
    area = moment = zero(eltype(interface))
    for block in eachindex(mesh.blocks)
        fraction = interior(interface, block)
        volume = mesh.blocks[block].volume
        for j in 1:ny, i in 1:nx
            gas = (1 - fraction[i, j]) * volume
            area += gas
            moment += gas * cellcenter(mesh, block, i, j)[2]
        end
    end
    return moment / area
end

"""
    bubble_rise_velocity(state)

Mean vertical bubble velocity, `∫ v dx / ∫ 1 dx` over the gas phase: the
vertical face velocities weighted by the gas fraction of their staggered
volumes. With two velocities it averages the gas velocity, which exists on
exactly the faces with a positive gas fraction.
"""
function bubble_rise_velocity(state)
    interface = state.interface
    mesh = interface.mesh
    velocity = state.velocity isa TwoPhaseVelocity ? state.velocity[2] :
        state.velocity
    volumes = mesh.face_dual_volumes
    momentum = volume = zero(eltype(interface))
    for (index, face) in pairs(mesh.faces)
        face.axis == 2 || continue
        gas = (1 - _face_liquid_fraction(interface, face)) * volumes[index]
        iszero(gas) && continue
        momentum += gas * velocity.interior[index]
        volume += gas
    end
    return momentum / volume
end

"""
    bubble_circularity(state)

Circularity `2√(πA)/P` of the bubble: the perimeter of the circle with the
bubble's area `A` over the length `P` of the PLIC interface.
"""
function bubble_circularity(state)
    area = diagnostic_value(PhaseVolume(), state)[2]
    return 2sqrt(pi * area) / diagnostic_value(InterfaceMeasure(), state)
end

"""
    run_example(; case=1, max_level=4, far_field_level=max_level-2,
                final_time=3.0, velocity_model=OneVelocityModel(),
                boundary=VelocityBoundaryConditions(left=FreeSlip(), right=FreeSlip(),
                                                    bottom=NoSlip(), top=NoSlip()),
                start_studio=nothing, recorder_options=(;), kwargs...)

Run test case `case` (1 or 2) of the rising-bubble benchmark of Hysing et al.
(2009): a bubble of radius 0.25 starts at rest at `(0.5, 0.5)` in
`[0,1] × [0,2]` under gravity `0.98`. Case 1 has density ratio 10 and
viscosity ratio 10; case 2 has density ratio 1000 and viscosity ratio 100.
By default the side walls are free-slip and the top and bottom walls are
no-slip, as in the benchmark. Pass `FreeSlip()` or `NoSlip()` as `boundary`
to use the same condition on all four walls. The interface band is refined
to `max_level`, `dx = 1/(8 * 2^max_level)`; the rest of the domain stays at
least at `far_field_level`, clamped to `0:max_level`. Pass
`start_studio=studio` after loading TwoPhaseAMRStudio to follow the run live.
`recorder_options` overrides the recording schedule; other keywords are
passed to [`FlowSimulation`](@ref).
"""
function run_example(;
        case = 1,
        max_level = 4,
        far_field_level = max_level - 2,
        final_time = 3.0,
        velocity_model = OneVelocityModel(),
        boundary = VelocityBoundaryConditions(;
            left = FreeSlip(), right = FreeSlip(), bottom = NoSlip(), top = NoSlip(),
        ),
        start_studio = nothing,
        recorder_options = (;),
        kwargs...,
    )
    (; density, dynamic_viscosity, surface_tension) = hysing_parameters(case)
    radius = 0.25
    centre = (0.5, 0.5)
    # Liquid where the level set is negative: outside the bubble.
    levelset = (x, y) -> radius - hypot(x - centre[1], y - centre[2])
    far_field = clamp(far_field_level, 0, max_level)

    mesh = BlockTreeMesh(
        base_mesh = (8, 16), block_size = (8, 8),
        origin = (0.0, 0.0), extent = (1.0, 2.0),
        periodic = (false, false), max_level = max_level,
        refine = (key, origin, extent) -> key.level < far_field,
        adaptation = MeshAdaptationOptions(
            update_interval = 5, coarsen_delay = 2, minimum_level = far_field,
        ),
    )
    interface = InterfaceField(mesh, levelset)
    refinement = InterfaceRefinementIndicator(interface; level = max_level)
    adaptation = AMRWorkspace(mesh, interface)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(interface, levelset)
    end

    physics = TwoPhase(
        interface; density, dynamic_viscosity, surface_tension,
        gravity = (0.0, -0.98),
    )
    flow_options = (;
        velocity_model,
        boundary,
        diagnostics = (
            PhaseVolume(),
            Diagnostic(
                :circularity, bubble_circularity;
                label = "Circularity", dimension = PhysicalDimension(0, 0, 0),
            ),
            Diagnostic(
                :centroid, state -> bubble_centroid(state.interface);
                label = "Bubble centroid height",
                dimension = PhysicalDimension(0, 1, 0),
            ),
            Diagnostic(
                :rise_velocity, bubble_rise_velocity;
                label = "Bubble rise velocity",
                dimension = PhysicalDimension(0, 1, -1),
            ),
        ),
        adaptivity = Adaptivity(criteria = (refinement,)),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
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
    case = "--case=2" in ARGS ? 2 : 1
    if "--studio" in ARGS
        using TwoPhaseAMRStudio
        run = run_example(; case, start_studio = studio)
    else
        run = run_example(; case)
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
