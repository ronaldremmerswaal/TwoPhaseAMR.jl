using TwoPhaseAMR

"""
    coalescence_initial_condition(; radius=1, bridge=0.15radius)

Two circles of radius `radius` that touch at the origin, centred at
`(±radius, 0)`, joined by a liquid bridge of half-width `bridge`. The bridge
ends in two fillets: circles of radius
`fillet = bridge^2 / (2(radius - bridge))` centred on the symmetry line at
`(0, ±(bridge + fillet))`, tangent to both drops, so the outline is smooth.
Returns a [`PolygonIndicator`](@ref) of the outline (512 chords per drop arc,
64 per fillet; fractions follow by exact clipping) and the fillet radius.
"""
function coalescence_initial_condition(; radius = 1.0, bridge = 0.15radius)
    0 < bridge < radius ||
        throw(ArgumentError("bridge must lie between zero and the radius"))
    fillet = bridge^2 / (2(radius - bridge))
    centre = bridge + fillet
    # The fillets touch the drops at (±reach, ±height), seen at the angle
    # `drop` from the right drop's centre and `corner` from the top fillet's.
    reach = fillet * radius / (radius + fillet)
    height = centre * radius / (radius + fillet)
    drop = atan(height, reach - radius)
    corner = atan(height - centre, reach)
    arc(x, y, r, from, to, n) =
        [(x + r * cos(a), y + r * sin(a)) for a in range(from, to; length = n + 1)]
    # Counter-clockwise: right drop, top fillet (right to left), left drop,
    # bottom fillet; consecutive arcs share their end points.
    right = arc(radius, 0.0, radius, -drop, drop, 512)
    top = arc(0.0, centre, fillet, corner, -pi - corner, 64)
    left = arc(-radius, 0.0, radius, pi - drop, pi + drop, 512)
    bottom = [(x, -y) for (x, y) in reverse(top)]
    outline = [right[1:(end - 1)]; top[1:(end - 1)]; left[1:(end - 1)]; bottom[1:(end - 1)]]
    return (; polygon = PolygonIndicator(outline), fillet)
end

# Liquid length of the cell side `ξ = side/2` (local coordinates in
# [-1/2, 1/2]²) on the liquid side `nx ξ + ny η <= d` of a PLIC line.
function _liquid_side_length(nx, ny, d, side)
    offset = d - nx * side / 2
    iszero(ny) && return offset >= 0 ? 1.0 : 0.0
    crossing = offset / ny
    return ny > 0 ? clamp(crossing + 0.5, 0.0, 1.0) : clamp(0.5 - crossing, 0.0, 1.0)
end

"""
    bridge_half_width(interface)

Half the liquid length on the symmetry line `x = 0`: while the drops merge,
the half-width `r_b` of the bridge between them. Every cell with a side on the
line contributes the part of that side that its PLIC line leaves in the
liquid; the cells on both sides of the line are averaged.
"""
function bridge_half_width(interface)
    reconstruct!(interface)
    mesh = interface.mesh
    nx, ny = mesh.block_size
    wet_length = 0.0
    for (b, block) in pairs(mesh.blocks)
        hx, hy = block.spacing
        # Column i has its left side at x = 0 for i = i0 + 1 and its right
        # side there for i = i0.
        position = -block.origin[1] / hx
        i0 = round(Int, position)
        abs(position - i0) < 1.0e-6 && 0 <= i0 <= nx || continue
        fraction = interior(interface.conditional_fraction, b)
        normal_x = interior(interface.normal_x, b)
        normal_y = interior(interface.normal_y, b)
        intercept = interior(interface.intercept, b)
        for (i, side) in ((i0 + 1, -1), (i0, 1))
            1 <= i <= nx || continue
            for j in 1:ny
                c = fraction[i, j]
                wet = c <= 0 ? 0.0 : c >= 1 ? 1.0 :
                    _liquid_side_length(
                        normal_x[i, j], normal_y[i, j], intercept[i, j], side,
                    )
                wet_length += wet * hy
            end
        end
    end
    # Both columns count the full line once each.
    return wet_length / 4
end

"""
    deformation(interface)

Mode-2 deformation `(I_xx - I_yy)/(I_xx + I_yy)` of the liquid from its second
moments about the origin, the common centroid of the symmetric drops: `2/3`
for two touching circles and zero for one circle; `2ε` for a small shape
perturbation `r = R(1 + ε cos 2θ)`.
"""
function deformation(interface)
    mesh = interface.mesh
    nx, ny = mesh.block_size
    ixx = iyy = 0.0
    for (b, block) in pairs(mesh.blocks)
        fraction = interior(interface.fraction, b)
        hx, hy = block.spacing
        for j in 1:ny, i in 1:nx
            x, y = cellcenter(mesh, b, i, j)
            mass = fraction[i, j] * block.volume
            ixx += mass * (x^2 + hx^2 / 12)
            iyy += mass * (y^2 + hy^2 / 12)
        end
    end
    return (ixx - iyy) / (ixx + iyy)
end

"""
    mode2_period(; radius, density, surface_tension)

Linear period `2π/ω₂` of the planar (cylindrical) mode-2 shape oscillation,
`ω_n² = (n³ - n)σ / ((ρ_l + ρ_g) R³)` (Rayleigh 1879), for a circle of radius
`radius`.
"""
function mode2_period(; radius, density, surface_tension)
    n = 2
    return 2pi * sqrt(sum(density) * radius^3 / ((n^3 - n) * surface_tension))
end

"""
    run_example(; density_ratio=1000, ohnesorge=0, bridge=0.15, band_level=3,
                max_level=5, neck_time=0.3, final_time=nothing,
                start_studio=nothing, recorder_options=(;), kwargs...)

Run the coalescence of two equal planar drops of radius one in a light gas
without gravity (`ρ_l = σ = R = 1`, so time is in units of the capillary
time `sqrt(ρ_l R³/σ)`). The drops start at rest, joined by a bridge of
half-width `bridge` with tangent fillets (`coalescence_initial_condition`).
Both phases are inviscid by
default; `ohnesorge > 0` gives the liquid the viscosity
`ohnesorge * sqrt(ρ_l σ R)` and the gas `0.0175` times that. The interface
band is refined to `band_level`; until `neck_time`, curvature refinement with
eight cells per radius of curvature refines the fillets and the retracting
gap ends up to `max_level`. The default end time is two linear mode-2 periods
of the merged drop. Pass `start_studio=studio` after loading
TwoPhaseAMRStudio to follow the run live. `recorder_options` overrides the
recording schedule; other keywords are passed to [`FlowSimulation`](@ref),
including `velocity_model=TwoVelocityModel()`.
"""
function run_example(;
        density_ratio = 1.0e3,
        ohnesorge = 0.0,
        bridge = 0.15,
        band_level = 3,
        max_level = 5,
        neck_time = 0.3,
        final_time = nothing,
        start_studio = nothing,
        recorder_options = (;),
        kwargs...,
    )
    0 <= band_level <= max_level ||
        throw(ArgumentError("band_level must lie between 0 and max_level"))
    radius = 1.0
    surface_tension = 1.0
    density = (1.0, inv(density_ratio))
    liquid_viscosity = ohnesorge * sqrt(density[1] * surface_tension * radius)
    initial = coalescence_initial_condition(; radius, bridge)

    mesh = BlockTreeMesh(
        base_mesh = (32, 32), block_size = (8, 8),
        origin = (-4.0, -4.0), extent = (8.0, 8.0),
        periodic = (false, false), max_level = max_level,
        adaptation = MeshAdaptationOptions(update_interval = 5, coarsen_delay = 2),
    )
    interface = InterfaceField(mesh, initial.polygon)
    refinement = (
        InterfaceRefinementIndicator(interface; level = band_level),
        CurvatureRefinementIndicator(interface; cells_per_radius = 8, t_end = neck_time),
    )
    adaptation = AMRWorkspace(mesh, interface)
    for pass in 1:(max_level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed ||
            break
        initialize!(interface, initial.polygon)
    end

    physics = TwoPhase(
        interface; density, surface_tension,
        dynamic_viscosity = (liquid_viscosity, 0.0175liquid_viscosity),
    )
    length_dimension = PhysicalDimension(0, 1, 0)
    flow_options = (;
        boundary = FreeSlip(),
        diagnostics = (
            Diagnostic(
                :bridge_half_width, state -> bridge_half_width(state.interface);
                label = "Bridge half-width", dimension = length_dimension,
            ),
            Diagnostic(
                :deformation, state -> deformation(state.interface);
                label = "Deformation", dimension = PhysicalDimension(0, 0, 0),
            ),
            KineticEnergy(), SurfaceEnergy(), PhaseVolume(),
        ),
        adaptivity = Adaptivity(criteria = refinement),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    merged_radius = sqrt(integrate(interface) / pi)
    period = mode2_period(; radius = merged_radius, density, surface_tension)
    target_time = isnothing(final_time) ? 2period : final_time
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
        isnothing(driver) ? simulate!(flow, target_time; observer = recorder) :
            simulate!(driver, target_time)
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
