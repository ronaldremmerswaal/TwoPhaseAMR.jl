using TwoPhaseAMR, LinearAlgebra, Random

"""
    rayleigh_taylor_initial_condition(; case=:multimode, modes=nothing,
                                      amplitude=nothing, seed=2026)

Interface of the heavy-over-light layer on the unit-width periodic box. The
heavy fluid lies above `y = η(x)`,
`η(x) = Σ_m a cos(2π m x + φ_m)`, with integer wavenumbers `modes` and the
root-mean-square amplitude `amplitude` (each mode has `a = amplitude √(2/N)`).
`case = :single_mode` defaults to the mode `m = 1` with `φ = 0`, so the bubble
axis is `x = 0` and the spike axis `x = 1/2`; `case = :multimode` defaults to
`m = 8, …, 16` with phases drawn from `MersenneTwister(seed)`.
"""
function rayleigh_taylor_initial_condition(;
        case = :multimode, modes = nothing, amplitude = nothing, seed = 2026,
    )
    case in (:single_mode, :multimode) ||
        throw(ArgumentError("case must be :single_mode or :multimode"))
    single = case === :single_mode
    selected = collect(something(modes, single ? (1:1) : (8:16)))
    rms = something(amplitude, single ? 0.002 / sqrt(2) : 0.001)
    phases = single ? zeros(length(selected)) :
        2pi .* rand(MersenneTwister(seed), length(selected))
    each = rms * sqrt(2 / length(selected))
    elevation(x) = sum(
        each * cos(2pi * m * x + φ) for (m, φ) in zip(selected, phases)
    )
    # Bottom and top walls: by t = 6.5 a single-mode bubble has risen about
    # 0.8 wavelengths and its faster spike about 1.6; the multimode zone stays
    # within ±0.7.
    domain = single ? (-3.0, 2.0) : (-1.0, 1.0)
    return (;
        modes = selected, phases, amplitude = each, elevation,
        levelset = (x, y) -> elevation(x) - y,
        lipschitz = hypot(1, 2pi * each * sum(selected)), domain,
    )
end

"""
    linear_growth_rate(wavenumber; density, dynamic_viscosity, gravity,
                       surface_tension=0)

Growth rate `n` of the Rayleigh--Taylor normal mode `exp(ikx + nt)` of two
viscous, semi-infinite fluids: `density = (ρ_upper, ρ_lower)` and
`dynamic_viscosity = (μ_upper, μ_lower)`, gravity `gravity` pointing down and
surface tension `surface_tension`. The root is that of the 4×4 determinant of
the interface conditions (continuous velocity and tangential stress, normal
stress balanced by gravity and capillarity) for the perturbation
`w = A e^{∓ky} + B e^{∓qy}`, `q² = k² + n/ν`, which reduces to Chandrasekhar's
characteristic equation for equal kinematic viscosities. Without viscosity
it returns the inviscid `√(Agk − σk³/(ρ_upper + ρ_lower))`; zero if stable.
"""
function linear_growth_rate(
        k; density, dynamic_viscosity, gravity, surface_tension = 0.0,
    )
    upper, lower = density
    μ_upper, μ_lower = dynamic_viscosity
    restoring = (lower - upper) * gravity + surface_tension * k^2
    inviscid_squared = -k * restoring / (upper + lower)
    inviscid_squared > 0 || return 0.0
    inviscid = sqrt(inviscid_squared)
    iszero(μ_upper) && iszero(μ_lower) && return inviscid
    function determinant(n)
        q_lower = sqrt(k^2 + n * lower / μ_lower)
        q_upper = sqrt(k^2 + n * upper / μ_upper)
        return det(
            [
                1 1 -1 -1
                k q_lower k q_upper
                2k^2 * μ_lower (q_lower^2 + k^2) * μ_lower -2k^2 * μ_upper -(q_upper^2 + k^2) * μ_upper
                (restoring / n + lower * n / k + 2μ_lower * k) (restoring / n + 2μ_lower * q_lower) (upper * n / k + 2μ_upper * k) 2μ_upper * q_upper
            ]
        )
    end
    # Viscosity only lowers the rate: scan down from the inviscid value to
    # the first sign change, then bisect.
    previous_rate = inviscid
    previous_value = determinant(previous_rate)
    for rate in range(inviscid, 1.0e-6 * inviscid; length = 4001)[2:end]
        value = determinant(rate)
        if sign(value) != sign(previous_value)
            low, high = rate, previous_rate
            low_value = value
            for _ in 1:100
                middle = (low + high) / 2
                middle_value = determinant(middle)
                if sign(middle_value) == sign(low_value)
                    low, low_value = middle, middle_value
                else
                    high = middle
                end
            end
            return (low + high) / 2
        end
        previous_rate, previous_value = rate, value
    end
    return 0.0
end

"""
    bubble_velocity(; atwood, gravity, wavelength)

Asymptotic two-dimensional bubble velocity of Goncharov's potential-flow
model, `√(2A/(1+A) · g/(3k))` with `k = 2π/λ`, i.e. the Froude number
`U/√(Agλ/(1+A)) = 1/√(3π)`.
"""
bubble_velocity(; atwood, gravity, wavelength) =
    sqrt(2atwood / (1 + atwood) * gravity * wavelength / (6pi))

"""
    axis_tip(interface, x, phase; threshold=0.01)

Tip of a single-mode bubble (`phase = :gas`, the highest light fluid) or
spike (`phase = :liquid`, the lowest heavy fluid) on the vertical cell column
at `x`. A cell holding more than `threshold` of the phase contributes the
height of its interface, taking the light fluid below the heavy one; scanning
the whole column keeps the tip when fragments of the other fluid cross the
axis behind it.
"""
function axis_tip(interface, x, phase; threshold = 0.01)
    phase in (:gas, :liquid) || throw(ArgumentError("phase must be :gas or :liquid"))
    mesh = interface.mesh
    tip = phase === :gas ? -Inf : Inf
    for (b, block) in pairs(mesh.blocks)
        left = block.origin[1]
        left <= x < left + block.extent[1] || continue
        dx, dy = block.spacing
        i = clamp(floor(Int, (x - left) / dx) + 1, 1, mesh.block_size[1])
        fractions = interior(interface, b)
        for j in axes(fractions, 2)
            c = fractions[i, j]
            height = block.origin[2] + (j - c) * dy
            if phase === :gas && c < 1 - threshold
                tip = max(tip, height)
            elseif phase === :liquid && c > threshold
                tip = min(tip, height)
            end
        end
    end
    return tip
end

"""
    mixing_heights(interface; threshold=0.01)

Bubble and spike penetration `(h_b, h_s)` of the multimode mixing zone about
`y = 0`: the highest height whose horizontally averaged heavy (liquid) volume
fraction is below `1 - threshold`, and minus the lowest height whose average
exceeds `threshold`. Rows are averaged on the finest cell spacing; a coarser
cell contributes to every fine row it covers.
"""
function mixing_heights(interface; threshold = 0.01)
    mesh = interface.mesh
    bottom = mesh.origin[2]
    row = minimum(block -> block.spacing[2], mesh.blocks)
    profile = zeros(round(Int, mesh.extent[2] / row))
    for (b, block) in pairs(mesh.blocks)
        fractions = interior(interface, b)
        dx, dy = block.spacing
        covered = round(Int, dy / row)
        for j in axes(fractions, 2)
            first_row = round(Int, (block.origin[2] + (j - 1) * dy - bottom) / row)
            weight = sum(view(fractions, :, j)) * dx / mesh.extent[1]
            for r in (first_row + 1):(first_row + covered)
                profile[r] += weight
            end
        end
    end
    height(r) = bottom + (r - 1 / 2) * row
    # Interpolate the crossing between the first row past the threshold and
    # its neighbour on the pure side.
    top = findlast(<(1 - threshold), profile)
    bubble = if isnothing(top) || top == length(profile)
        height(something(top, 0))
    else
        a, b = profile[top], profile[top + 1]
        height(top) + row * ((1 - threshold) - a) / (b - a)
    end
    low = findfirst(>(threshold), profile)
    spike = if isnothing(low) || low == 1
        height(something(low, 1))
    else
        a, b = profile[low - 1], profile[low]
        height(low - 1) + row * (threshold - a) / (b - a)
    end
    return (bubble, -spike)
end

"""
    origin_heights(interface, dye)

Mean initial height of the heavy fluid that now lies below `y = 0` and mean
initial depth of the light fluid above it, from the dye that carries each
parcel's initial height. Both start near zero and grow as the mixing zone
draws fluid from further inside the pure layers.
"""
function origin_heights(interface, dye)
    mesh = interface.mesh
    heavy = heavy_label = light = light_label = 0.0
    for (b, block) in pairs(mesh.blocks)
        fractions = interior(interface, b)
        liquid, gas = interior(dye.liquid, b), interior(dye.gas, b)
        dx, dy = block.spacing
        for j in axes(fractions, 2), i in axes(fractions, 1)
            c = fractions[i, j] * dx * dy
            if block.origin[2] + (j - 1 / 2) * dy < 0
                heavy += c
                heavy_label += c * liquid[i, j]
            else
                light += dx * dy - c
                light_label += (dx * dy - c) * gas[i, j]
            end
        end
    end
    return (heavy_label / heavy, -light_label / light)
end

"""
    rayleigh_taylor_fields(flow, dye; colorrange=(-0.5, 0.5))

Visualization fields of a run: the flow's defaults and the `:dye`, drawn
with the light fluid's initial heights in blue and the heavy fluid's in red.
"""
function rayleigh_taylor_fields(flow, dye; colorrange = (-0.5, 0.5))
    fields = VisualizationFieldRegistry()
    register_default_fields!(fields, flow)
    register_field!(
        fields, :dye, dye; label = "Dye (initial height)", colormap = :balance,
        colorrange,
    )
    return fields
end

"""
    run_example(; case=:multimode, atwood=0.5, reynolds=nothing, bond=nothing,
                max_level=nothing, final_time=nothing, modes=nothing,
                amplitude=nothing, seed=2026, velocity_model=OneVelocityModel(),
                vorticity_tolerance=0.02, start_studio=nothing,
                recorder_options=(;), kwargs...)

Run a heavy-over-light Rayleigh--Taylor layer in the unit-width box,
periodic in `x` with free-slip walls, with gravity `g = 1` downward and equal
kinematic viscosities. `case = :single_mode` follows one wavelength on
`[0,1] × [-3,2]` (defaults: level 4, `reynolds = 1000`, no surface tension,
`T = 6.5`); `case = :multimode` follows modes 8–16 with seeded random phases
on `[0,1] × [-1,1]` (defaults: level 5, `reynolds = 3000`, `bond = 4e4`,
`T = 4`). `reynolds` is `L√(AgL/(1+A))/ν` for the box width `L = 1`, and
`bond = (ρ_h - ρ_l) g L²/σ` (`Inf` for no surface tension). A passive phase
scalar `:dye` carries each parcel's initial height. AMR keeps the interface
band and the vortical regions (`h|ω| > vorticity_tolerance`) at `max_level`.
Pass `start_studio=studio` after loading TwoPhaseAMRStudio to follow the run
live. `recorder_options` overrides the recording schedule; other keywords are
passed to [`FlowSimulation`](@ref).
"""
function run_example(;
        case = :multimode,
        atwood = 0.5,
        reynolds = nothing,
        bond = nothing,
        max_level = nothing,
        final_time = nothing,
        modes = nothing,
        amplitude = nothing,
        seed = 2026,
        velocity_model = OneVelocityModel(),
        vorticity_tolerance = 0.02,
        start_studio = nothing,
        recorder_options = (;),
        kwargs...,
    )
    initial = rayleigh_taylor_initial_condition(; case, modes, amplitude, seed)
    single = case === :single_mode
    level = something(max_level, single ? 4 : 5)
    gravity = 1.0
    density = ((1 + atwood) / (1 - atwood), 1.0)
    reynolds_number = something(reynolds, single ? 1000.0 : 3000.0)
    kinematic_viscosity = sqrt(atwood * gravity / (1 + atwood)) / reynolds_number
    dynamic_viscosity = density .* kinematic_viscosity
    bond_number = something(bond, single ? Inf : 4.0e4)
    surface_tension = (density[1] - density[2]) * gravity / bond_number

    bottom, top = initial.domain
    mesh = BlockTreeMesh(
        base_mesh = (8, round(Int, 8 * (top - bottom))), block_size = (8, 8),
        origin = (0.0, bottom), extent = (1.0, top - bottom),
        periodic = (true, false), max_level = level,
        adaptation = MeshAdaptationOptions(update_interval = 5, coarsen_delay = 2),
    )
    interface = InterfaceField(mesh, initial.levelset; lipschitz = initial.lipschitz)
    band = InterfaceRefinementIndicator(interface; level)
    adaptation = AMRWorkspace(mesh, interface)
    for pass in 1:(level + 1)
        adapt!(adaptation, band; step = pass, force = true).changed || break
        initialize!(interface, initial.levelset; lipschitz = initial.lipschitz)
    end

    two_velocities = velocity_model isa TwoVelocityModel
    velocity = two_velocities ? (FaceVelocity(mesh), FaceVelocity(mesh)) :
        FaceVelocity(mesh)
    vortical = Tuple(
        VorticityRefinementIndicator(
            field; refine_tolerance = vorticity_tolerance,
            maximum_level = level,
        ) for field in (two_velocities ? velocity : (velocity,))
    )
    dye = PhaseScalar(interface; liquid = (x, y) -> y, gas = (x, y) -> y)
    physics = TwoPhase(
        interface; density, dynamic_viscosity, surface_tension,
        gravity = (0.0, -gravity), scalars = (dye,),
    )
    height = PhysicalDimension(0, 1, 0)
    penetration = if single
        (
            Diagnostic(
                :bubble_height, state -> axis_tip(state.interface, 0.0, :gas);
                label = "Bubble height", dimension = height,
            ),
            Diagnostic(
                :spike_height, state -> -axis_tip(state.interface, 0.5, :liquid);
                label = "Spike depth", dimension = height,
            ),
        )
    else
        (
            Diagnostic(
                :bubble_height, state -> mixing_heights(state.interface)[1];
                label = "Bubble height", dimension = height,
            ),
            Diagnostic(
                :spike_height, state -> mixing_heights(state.interface)[2];
                label = "Spike depth", dimension = height,
            ),
            Diagnostic(
                :spike_origin, state -> origin_heights(state.interface, dye)[1];
                label = "Origin of heavy fluid below y = 0", dimension = height,
            ),
            Diagnostic(
                :bubble_origin, state -> origin_heights(state.interface, dye)[2];
                label = "Origin of light fluid above y = 0", dimension = height,
            ),
        )
    end
    flow_options = (;
        velocity_model, velocity, boundary = FreeSlip(),
        diagnostics = (
            penetration..., PhaseVolume(), KineticEnergy(), PotentialEnergy(),
            CellCount(),
        ),
        adaptivity = Adaptivity(criteria = level > 0 ? (band, vortical...) : ()),
        kwargs...,
    )
    flow = FlowSimulation(mesh, physics; flow_options...)
    target_time = something(final_time, single ? 6.5 : 4.0)
    fields = rayleigh_taylor_fields(
        flow, dye; colorrange = single ? (-1.0, 1.0) : (-0.5, 0.5),
    )
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
    parameters = (;
        atwood, gravity, density, dynamic_viscosity, kinematic_viscosity,
        surface_tension, reynolds = reynolds_number, bond = bond_number, initial,
    )
    return (; flow, result, recorder, app, dye, parameters)
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
