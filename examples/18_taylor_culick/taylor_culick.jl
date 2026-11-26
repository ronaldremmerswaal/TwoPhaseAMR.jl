using TwoPhaseAMR

"""Largest `|y|` reached by the PLIC interface: half the thickest liquid."""
function liquid_half_thickness(interface)
    segments = plic_segments(interface)
    isempty(segments) && return 0.0
    return maximum(
        segment -> max(abs(segment[1][2]), abs(segment[2][2])), segments,
    )
end

"""
Liquid share `Σ ρ_l α_f g(u_f, axis) V_f` of a staggered integral, with the
mass-compatible liquid fraction `α_f` and dual volume `V_f` of each face.
"""
function liquid_integral(g, state, liquid_density)
    # The mixture density of the densities (2, 1) is 1 + α_f.
    face_density = mixture_density(state.interface, (2.0, 1.0))
    velocity = state.velocity isa TwoPhaseVelocity ? state.velocity[1] :
        state.velocity
    mesh = state.solver.mesh
    total = 0.0
    for (i, rho) in pairs(face_density)
        fraction = clamp(rho - 1, 0, 1)
        iszero(fraction) && continue
        total += liquid_density * fraction * mesh.face_dual_volumes[i] *
            g(velocity.interior[i], mesh.faces[i].axis)
    end
    return total
end

"""
    run_example(; ohnesorge=0.1, density_ratio=1000, gas_ohnesorge=2e-3,
                sheet_length=24, max_level=3, final_time=12.0,
                velocity_model=TwoVelocityModel(...), start_studio=nothing,
                recorder_options=(;), kwargs...)

Retract a planar liquid sheet of thickness `e = 1` from its free end. Lengths
are in units of `e`, densities of the liquid density and speeds of the
Taylor--Culick speed `V = sqrt(2σ/(ρ_l e)) = 1`, so `σ = 1/2`. The liquid
viscosity follows from the Ohnesorge number `Oh = μ_l/sqrt(ρ_l σ e)`, the gas
viscosity from `gas_ohnesorge` in the same units. The sheet lies on
`|y| < 1/2` from the free-slip wall `x = 0`, a symmetry plane, to a rounded
end at `x = sheet_length`, in the closed box `[0, L] × [-8.2, 7.8]` with `L`
the multiple of 8 at least 8 past the end; level `max_level` resolves the
thickness with `2^max_level` cells. Pass `start_studio=studio` after loading
TwoPhaseAMRStudio to follow the run live. `recorder_options` overrides the
recording schedule; other keywords are passed to [`FlowSimulation`](@ref).
"""
function run_example(;
        ohnesorge = 0.1,
        density_ratio = 1.0e3,
        gas_ohnesorge = 2.0e-3,
        sheet_length = 24,
        max_level = 3,
        final_time = 12.0,
        velocity_model = TwoVelocityModel(
            pressure_projection = CMDGFMProjection(),
        ),
        start_studio = nothing,
        recorder_options = (;),
        kwargs...,
    )
    thickness = 1.0
    density = (1.0, inv(density_ratio))
    # Taylor--Culick speed sqrt(2σ/(ρ_l e)) = 1.
    surface_tension = density[1] * thickness / 2
    viscosity_scale = sqrt(density[1] * surface_tension * thickness)
    dynamic_viscosity = (ohnesorge, gas_ohnesorge) .* viscosity_scale
    end_centre = sheet_length - thickness / 2
    levelset = (x, y) -> x <= end_centre ? abs(y) - thickness / 2 :
        hypot(x - end_centre, y) - thickness / 2

    # Unit base cells in 8 × 8 blocks, at least 8 units of gas past the end.
    # The box is shifted by 0.2 so that no sheet surface lies on a cell face
    # of any level: there both cells would be pure, without the PLIC segment
    # and curvature that surface tension and the rendering use.
    domain_length = 8 * Int(cld(sheet_length + 8, 8))
    mesh = BlockTreeMesh(
        base_mesh = (domain_length, 16), block_size = (8, 8),
        origin = (0.0, -8.2), extent = (float(domain_length), 16.0),
        periodic = (false, false), max_level = max_level,
        adaptation = MeshAdaptationOptions(
            update_interval = 5, coarsen_delay = 2,
        ),
    )
    # On the base level one cell holds the whole quarter arc of the rounded
    # end, a height graph with an infinite slope in either direction; two
    # forced subdivisions separate its horizontal and vertical tangents.
    initial = (; lipschitz = 1, min_depth = 2)
    interface = InterfaceField(mesh, levelset; initial...)
    refinement = InterfaceRefinementIndicator(interface; level = max_level)
    adaptation = AMRWorkspace(mesh, interface)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(interface, levelset; initial...)
    end

    physics = TwoPhase(
        interface; density, dynamic_viscosity, surface_tension,
    )
    flow_options = (;
        velocity_model, boundary = FreeSlip(),
        diagnostics = (
            Diagnostic(
                :tip_position,
                state -> interface_height(
                    state.interface; direction = :x, position = 0.0,
                );
                label = "Tip position", dimension = PhysicalDimension(0, 1, 0),
            ),
            Diagnostic(
                :edge_thickness,
                state -> 2liquid_half_thickness(state.interface);
                label = "Largest liquid thickness",
                dimension = PhysicalDimension(0, 1, 0),
            ),
            Diagnostic(
                :liquid_momentum,
                state -> liquid_integral(
                    (u, axis) -> axis == 1 ? u : 0.0, state, density[1],
                );
                label = "Liquid x-momentum",
                dimension = PhysicalDimension(1, 0, -1),
            ),
            Diagnostic(
                :liquid_kinetic_energy,
                state -> liquid_integral((u, axis) -> u^2 / 2, state, density[1]);
                label = "Liquid kinetic energy",
                dimension = PhysicalDimension(1, 1, -2),
            ),
            KineticEnergy(), SurfaceEnergy(), LinearMomentum(), PhaseVolume(),
        ),
        adaptivity = Adaptivity(
            criteria = max_level > 0 ? (refinement,) : (),
        ),
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
