# Chapter 13 (stability and accuracy): time-step bounds of representative
# flows, the step sequence of simulate!, and an empirical stability margin.
# Included by generate.jl; `generate.jl stability` regenerates only these
# numbers (see the end of generate.jl).
#
# The flow bound eq:stab:flow-bound is recombined here from the code's
# advective bounds (evaluated independently in Chapters 5 and 6) and an
# independent evaluation of the capillary and gravity bounds with the smallest
# leaf cell width; the selection rule eq:stab:selection is replayed on the
# step sequences and bounds that simulate! records. The assertions fail if the
# code no longer matches the statements of Chapter 13.

# --- Fixtures ------------------------------------------------------------------

# Every flow records, after each step, the bound of the next step. The
# records start after the boundary values are imposed: the WY bound rejects a
# velocity with permeable walls, and initialisers need not satisfy them.
const STAB_CFL = 0.45

function stab_track!(flow)
    AMR._enforce_current_velocity_boundary!(flow.velocity, flow.solver, flow.time)
    add_diagnostic!(flow, MaximumVelocity())
    add_diagnostic!(flow, CFLTimeStep(cfl = STAB_CFL))
    return flow
end

# Standing capillary wave of examples/06_capillary_wave (density ratio 1000,
# Laplace number 3000, one wavelength, free-slip walls) on 32 x 96 level-0
# cells whose interface band is refined once (smallest width 1/64).
const STAB_WAVE = (; amplitude = 0.01, laplace = 3000.0, ratio = 1.0e3, level = 1)

function stab_capillary_wave(; velocity_model = OneVelocityModel())
    k = 2pi
    density = (1.0, inv(STAB_WAVE.ratio))
    nu = sqrt(1.0 / STAB_WAVE.laplace)
    levelset = (x, y) -> y - 0.25 - STAB_WAVE.amplitude * cos(k * x)
    mesh = BlockTreeMesh(;
        base_mesh = (32, 96), block_size = (8, 8), origin = (0.0, -1.5),
        extent = (1.0, 3.0), periodic = (true, false), max_level = STAB_WAVE.level,
    )
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    refinement = InterfaceRefinementIndicator(interface; level = STAB_WAVE.level)
    adaptation = AMRWorkspace(mesh, interface)
    for pass in 1:(STAB_WAVE.level + 1)
        adapt!(adaptation, refinement; step = pass, force = true).changed || break
        initialize!(interface, levelset; lipschitz = 1)
    end
    physics = TwoPhase(
        interface; density, dynamic_viscosity = density .* nu, surface_tension = 1.0,
    )
    flow = stab_track!(FlowSimulation(mesh, physics; velocity_model, boundary = FreeSlip()))
    # Linear-theory angular frequency of the inviscid wave: the scale of its
    # velocity amplitude a*omega.
    omega = sqrt(1.0 * k^3 / sum(density))
    return (; name = :wave, flow, mesh, density, sigma = 1.0, gravity = 0.0, omega)
end

# Flow past a cylinder of examples/04_cylinder_flow at Reynolds number 100:
# unit inflow, 56 x 32 level-0 cells of width 1/4, the cut cells of the
# cylinder refined once (width 1/8).
function stab_cylinder(; solid = true)
    mesh = BlockTreeMesh(;
        base_mesh = (56, 32), block_size = (4, 4), origin = (-4.0, -4.0),
        extent = (14.0, 8.0), periodic = (false, false), max_level = 1,
    )
    cylinder = (x, y) -> hypot(x, y) - 0.5
    geometry = GeometryField(mesh, cylinder; lipschitz = 1, atol = 1.0e-7, maxdepth = 24)
    inflow = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0
    velocity = FaceVelocity(mesh, inflow)
    refinement = GeometryRefinementIndicator(geometry; level = 1)
    adaptation = AMRWorkspace(mesh, geometry, velocity)
    for pass in 1:2
        adapt!(adaptation, refinement; step = pass, force = true).changed || break
        initialize!(geometry.solid, cylinder; lipschitz = 1)
        refresh_geometry!(geometry)
        set_velocities!(velocity, inflow)
    end
    boundary = VelocityBoundaryConditions(
        left = Inflow(inflow), right = Outflow(),
        bottom = ZeroGradientVelocity(), top = ZeroGradientVelocity(),
    )
    # Without the solid: the same inflow on the same refined mesh, the bound
    # against which the cut-cell reduction is measured.
    flow = stab_track!(
        FlowSimulation(
            mesh, SinglePhase(density = 1.0, dynamic_viscosity = 0.01);
            velocity, geometry = solid ? geometry : nothing, boundary,
        ),
    )
    return (; name = :cylinder, flow, mesh, geometry)
end

# Kelvin--Helmholtz wave of the two-velocity test suite (equal densities,
# Weber number 5, no gravity; test/integration/interfacial_instability_problem.jl)
# on its coarsest mesh of 8 cells per wavelength, with frozen interface fluxes
# and the CMDGFM projection. With `ratio = 3, weber = 0.5, bond = 0.75,
# liquid_above = true` it is the combined Kelvin--Helmholtz/Rayleigh--Taylor
# wave of examples/07_interfacial_instability. Standing wave (zero mean
# interface velocity) with the linear-theory initial velocities.
const STAB_KH = (; amplitude = 0.01, weber = 5.0, cells = 8)

function stab_phase_velocity(base, perturbation, above, k)
    above && return (x, y, time, axis) -> axis == 1 ?
        base - k * perturbation * cos(k * x) * exp(-k * max(y, 0)) :
        k * perturbation * sin(k * x) * exp(-k * max(y, 0))
    return (x, y, time, axis) -> axis == 1 ?
        base + k * perturbation * cos(k * x) * exp(k * min(y, 0)) :
        k * perturbation * sin(k * x) * exp(k * min(y, 0))
end

function stab_kelvin_helmholtz(;
        cells = STAB_KH.cells, ratio = 1.0, weber = STAB_KH.weber, bond = 0.0,
        liquid_above = false,
    )
    k = 2pi
    a = STAB_KH.amplitude
    density = (1.0, inv(ratio))
    total = sum(density)
    sigma = total / k^3
    jump = total * sqrt(weber / prod(density)) / k
    speeds = (-density[2] * jump / total, density[1] * jump / total)
    gravity = bond == 0 ? 0.0 : bond * total / (k * (density[1] - density[2]))
    mesh = BlockTreeMesh(;
        base_mesh = (cells, 3cells), block_size = (8, 8), origin = (0.0, -1.5),
        extent = (1.0, 3.0), periodic = (true, false),
    )
    levelset = liquid_above ? ((x, y) -> a * cos(k * x) - y) :
        ((x, y) -> y - a * cos(k * x))
    interface = InterfaceField(mesh, levelset; lipschitz = hypot(1, a * k))
    liquid = stab_phase_velocity(speeds[1], -a * speeds[1], liquid_above, k)
    gas = stab_phase_velocity(speeds[2], -a * speeds[2], !liquid_above, k)
    physics = TwoPhase(
        interface; density, dynamic_viscosity = (0.0, 0.0), surface_tension = sigma,
        gravity = (0.0, -gravity),
    )
    flow = stab_track!(
        FlowSimulation(
            mesh, physics;
            velocity_model = TwoVelocityModel(
                interface_transport = FrozenInterfaceFluxes(),
                pressure_projection = CMDGFMProjection(),
            ),
            velocity = (liquid, gas), boundary = FreeSlip(),
        ),
    )
    radicand = 1 - weber - (liquid_above ? bond : -bond)
    return (;
        name = bond == 0 ? :kh : :khrt, flow, mesh, density, sigma, gravity,
        speeds, radicand,
    )
end

# Decaying Taylor--Green vortex on 8 x 8 periodic cells (single phase), or the
# fluid at rest, for the selection rule.
function stab_taylor_green(; rest = false, viscosity = 0.05)
    mesh = BlockTreeMesh(; block_size = (8, 8), periodic = (true, true))
    velocity = rest ? nothing : (
            (x, y, time, axis) -> axis == 1 ? sinpi(2x) * cospi(2y) :
            -cospi(2x) * sinpi(2y)
        )
    return stab_track!(
        FlowSimulation(
            mesh, SinglePhase(density = 1.0, dynamic_viscosity = viscosity); velocity,
        ),
    )
end

# --- Bounds ----------------------------------------------------------------------

# Smallest leaf cell width, the h of the capillary and gravity bounds.
stab_min_width(mesh) = minimum(minimum(block.spacing) for block in mesh.blocks)

# Capillary and gravity bounds evaluated from their formulas
# (eq:vof:dt-forces), independently of the code.
function stab_force_bounds(case)
    h = stab_min_width(case.mesh)
    hasproperty(case, :density) || return (; h, capillary = Inf, gravity = Inf)
    rho = case.density
    capillary = case.sigma > 0 ? sqrt(sum(rho) * h^3 / (4pi * case.sigma)) : Inf
    gravity = case.gravity > 0 && rho[1] != rho[2] ?
        sqrt(sum(rho) * h / (2abs(rho[1] - rho[2]) * case.gravity)) : Inf
    return (; h, capillary, gravity)
end

# The advective bound of eq:stab:flow-bound from the code's per-transport
# bounds, after imposing the boundary values as simulate! does.
function stab_advective_bound(flow, cfl)
    AMR._enforce_current_velocity_boundary!(flow.velocity, flow.solver, flow.time)
    workspace = flow.workspace
    isnothing(flow.solver.two_phase) &&
        return stable_timestep(flow.velocity, workspace.advection; cfl)
    return stable_timestep(flow.velocity, workspace.two_phase.advection; cfl)
end

# All bounds of a flow at its current state, the code's flow bound and the
# name of the active one.
function stab_bounds(case; cfl = STAB_CFL)
    flow = case.flow
    advective = stab_advective_bound(flow, cfl)
    forces = stab_force_bounds(case)
    combined = min(advective, forces.capillary, forces.gravity)
    code = stable_timestep(flow; cfl)
    @assert code == combined || isapprox(code, combined; rtol = 1.0e-14)
    active = combined == advective ? :advective :
        combined == forces.capillary ? :capillary : :gravity
    return (; advective, forces.capillary, forces.gravity, forces.h, code, active)
end

# --- Step sequences --------------------------------------------------------------

# Steps and next-step bounds recorded by simulate! (one sample per physical
# step; a completion sample at the same step is dropped).
function stab_history(flow)
    history = diagnostic_history(flow.diagnostics, :cfl_timestep)
    keep = [i == 1 || history.steps[i] != history.steps[i - 1] for i in eachindex(history.steps)]
    times = history.times[keep]
    return (; times, steps = diff(times), bounds = history.values[keep])
end

# The steps are recovered from the recorded times, so a step equals its
# proposal up to the rounding of the times.
stab_same(step, value, time) = abs(step - value) <= 4eps(max(abs(time), 1.0))

# Replays eq:stab:selection on a recorded automatic run that started without a
# retained proposal: proposal_n = min(proposal_(n-1), bound_n); every step but
# the last equals its proposal, the last one ends at the target.
function stab_replay(record, target)
    proposal = Inf
    proposals = Float64[]
    for n in eachindex(record.steps)
        proposal = min(proposal, record.bounds[n])
        push!(proposals, proposal)
    end
    last = length(record.steps)
    regular = all(
        stab_same(record.steps[n], proposals[n], record.times[n + 1]) for n in 1:(last - 1)
    )
    remaining = target - record.times[last]
    final = remaining - proposals[last] < 1.0e-6 * proposals[last] ||
        stab_same(record.steps[last], proposals[last], target)
    return (; proposals, ok = regular && final && record.times[end] == target)
end

# Bounds at the start and at the end of a run of simulate! over `steps` times
# the first step, with the replay of its steps.
function stab_case(case; steps = STAB_CASE_STEPS)
    start = stab_bounds(case)
    flow = case.flow
    target = flow.time + steps * start.code
    simulate!(flow, target)
    record = stab_history(flow)
    replay = stab_replay(record, target)
    @assert replay.ok
    @assert issorted(replay.proposals; rev = true)
    finish = stab_bounds(case)
    return (; start, finish, record, replay, steps = length(record.steps))
end

# Bounds of the four flows of tab:stab:cases; the first three then run over
# STAB_CASE_STEPS times their first step.
const STAB_CASE_STEPS = 12

function case_bounds_check()
    wave = stab_case(stab_capillary_wave())
    cylinder_case = stab_cylinder()
    cylinder = stab_case(cylinder_case)
    plain = stab_bounds(stab_cylinder(solid = false))
    mesh = cylinder_case.mesh
    (nx, ny), halo = mesh.block_size, mesh.halo
    smallest = minimum(
        value for block in cylinder_case.geometry.open_fraction.blocks
            for value in view(block, (halo + 1):(halo + nx), (halo + 1):(halo + ny))
            if 0 < value < 1;
        init = 1.0,
    )
    kh = stab_case(stab_kelvin_helmholtz())
    khrt = stab_bounds(
        stab_kelvin_helmholtz(;
            cells = 16, ratio = 3.0, weber = 0.5, bond = 0.75, liquid_above = true,
        ),
    )
    @assert wave.start.active == :capillary && isinf(wave.start.advective)
    @assert cylinder.start.active == :advective && kh.start.active == :advective
    @assert khrt.active == :capillary && isfinite(khrt.gravity)
    # The wave and the cylinder keep their first proposal; the bound of the
    # growing Kelvin--Helmholtz wave falls and shortens its steps.
    @assert wave.steps == cylinder.steps == STAB_CASE_STEPS < kh.steps
    @assert all(==(cylinder.replay.proposals[1]), cylinder.replay.proposals)
    @assert kh.replay.proposals[end] < kh.replay.proposals[1]
    return (; wave, cylinder, plain, smallest, kh, khrt)
end

# --- Selection rule on a small periodic flow -------------------------------------

function stab_accepts(flow, dt)
    try
        step!(flow.velocity, flow.pressure, flow.solver, dt, flow.workspace; time = flow.time)
        return true
    catch error
        error isa ArgumentError || rethrow()
        return false
    end
end

function step_selection_check()
    # Infinite bound: the first step is the interval / 128.
    rest = stab_taylor_green(; rest = true)
    capped = simulate!(rest, 1.0)
    rest_record = stab_history(rest)
    @assert capped.steps == 128 &&
        all(n -> stab_same(rest_record.steps[n], 1 / 128, rest_record.times[n + 1]), 1:128)

    # The proposal never grows while the decaying vortex raises its bound.
    flow = stab_taylor_green()
    proposal = stable_timestep(flow; cfl = STAB_CFL)
    run = simulate!(flow, 10proposal)
    record = stab_history(flow)
    @assert all(
        n -> stab_same(record.steps[n], proposal, record.times[n + 1]),
        1:(length(record.steps) - 1),
    )
    growth = record.bounds[end] / proposal

    # A remainder below 1e-6 of the proposal stretches the last step; a
    # larger one is a step of its own.
    stretched = stab_taylor_green()
    simulate!(stretched, 3proposal + 0.5e-6 * proposal)
    stretched_record = stab_history(stretched)
    short = stab_taylor_green()
    simulate!(short, 3proposal + 2.0e-6 * proposal)
    short_record = stab_history(short)
    @assert length(stretched_record.steps) == 3 && length(short_record.steps) == 4
    @assert stretched_record.steps[end] > proposal && stretched_record.times[end] > 3proposal

    # A second call resumes with the retained proposal, not with the
    # shortened last step of the first; chunk ends on the step grid reproduce
    # the uninterrupted run.
    chunked = stab_taylor_green()
    simulate!(chunked, 2.5proposal)
    simulate!(chunked, 5proposal)
    chunked_record = stab_history(chunked)
    @assert length(chunked_record.steps) == 6
    @assert stab_same(chunked_record.steps[3], 0.5proposal, 2.5proposal) &&
        stab_same(chunked_record.steps[4], proposal, 3.5proposal)
    whole = stab_taylor_green()
    simulate!(whole, 4proposal)
    aligned = stab_taylor_green()
    simulate!(aligned, 2proposal)
    simulate!(aligned, 4proposal)
    @assert length(stab_history(aligned).steps) == 4
    aligned_difference = maximum(abs, whole.velocity.interior .- aligned.velocity.interior)
    @assert aligned_difference == 0 && whole.pressure.blocks == aligned.pressure.blocks

    # The check of step!: (1 + 1e-6 + 32 eps) times the bound at cfl = 1/2.
    hard = stable_timestep(stab_taylor_green(); cfl = 0.5)
    accepted = stab_accepts(stab_taylor_green(), (1 + 0.9e-6) * hard)
    rejected = !stab_accepts(stab_taylor_green(), (1 + 2.0e-6) * hard)
    @assert accepted && rejected
    return (;
        capped_steps = capped.steps, proposal, steps = run.steps, growth,
        stretched = stretched_record.steps[end] / proposal,
        short = short_record.steps[end] / proposal, aligned_difference,
        accepted, rejected,
    )
end

# --- Empirical stability margin --------------------------------------------------

# `steps` steps of fixed size m * (selected step at the start), each through
# the prepared two-velocity path of simulate! (the frozen apertures of the
# time-step start are set by the bound evaluation), so that step! does not
# refuse a step above the bound. Records the largest velocity after each step
# and the first error.
function stab_margin_run(build, m; steps)
    case = build()
    flow = case.flow
    AMR._enforce_current_velocity_boundary!(flow.velocity, flow.solver, flow.time)
    selected = stable_timestep(flow; cfl = STAB_CFL)
    dt = m * selected
    speeds = Float64[]
    failure = ""
    for _ in 1:steps
        try
            AMR._enforce_current_velocity_boundary!(flow.velocity, flow.solver, flow.time)
            stable_timestep(flow; cfl = STAB_CFL)
            stats = step!(
                flow.velocity, flow.pressure, flow.solver, dt, flow.workspace;
                time = flow.time, _transport_geometry_prepared = true,
            )
            stats.converged || (failure = "projection did not converge"; break)
        catch error
            failure = first(split(sprint(showerror, error), '\n'))
            break
        end
        flow.time += dt
        state = SimulationState(
            flow.velocity, flow.pressure, flow.solver, flow.time, 0, flow.workspace,
        )
        speed = diagnostic_value(MaximumVelocity(), state)
        push!(speeds, speed)
        isfinite(speed) || (failure = "non-finite velocity"; break)
    end
    return (; m, selected, dt, speeds, failure, case)
end

# Runs at the given multiples; a run is stable if it completes its steps
# without an error and its largest velocity stays at most `limit`. The margin
# is the largest multiple below the first unstable one.
function stab_margin(build, multiples; steps, limit)
    runs = [stab_margin_run(build, m; steps) for m in multiples]
    stable = [
        isempty(r.failure) && length(r.speeds) == steps && maximum(r.speeds) <= limit
            for r in runs
    ]
    unstable = findfirst(!, stable)
    @assert !isnothing(unstable) && unstable > 1 && all(stable[1:(unstable - 1)])
    return (; runs, stable, margin = multiples[unstable - 1], unstable = runs[unstable])
end

function stability_margin_check()
    # Capillary wave with two velocities (capillary bound active): stable if
    # the velocity stays below the linear-theory amplitude a*omega.
    wave_build = () -> stab_capillary_wave(velocity_model = TwoVelocityModel())
    scale = STAB_WAVE.amplitude * wave_build().omega
    wave = stab_margin(
        wave_build, (1.0, 2.0, 2.25, 2.5, 2.75, 3.0); steps = 40, limit = scale,
    )
    blowup = findfirst(>(scale), wave.unstable.speeds)
    @assert wave.margin >= 2 && !isnothing(blowup) &&
        maximum(wave.unstable.speeds) > 10scale
    # Kelvin--Helmholtz wave (advective bound active): the transport refuses
    # the step once a fraction leaves [0, 1].
    kh = stab_margin(stab_kelvin_helmholtz, (1.0, 1.2, 1.35, 1.5, 1.75); steps = 6, limit = Inf)
    @assert kh.margin >= 1.2 &&
        startswith(kh.unstable.failure, "WY conditional fraction out of bounds")
    return (;
        wave, scale, blowup, wave_steps = 40,
        wave_stable_speed = maximum(wave.runs[findfirst(==(wave.margin), getproperty.(wave.runs, :m))].speeds) / scale,
        wave_unstable_speed = maximum(wave.unstable.speeds) / scale,
        kh, kh_steps = 6, kh_completed = length(kh.unstable.speeds),
    )
end

# --- Outputs ---------------------------------------------------------------------

stab_value(x) = isinf(x) ? "\\infty" : x >= 0.01 ? @sprintf("%.3f", x) : sci(x)

function stability_outputs(write_output)
    cases = case_bounds_check()
    selection = step_selection_check()
    margin = stability_margin_check()
    println("stability case bounds:       ", map(c -> hasproperty(c, :start) ? c.start : c, cases))
    println("stability selection check:   ", selection)
    println(
        "stability margin:            ", (;
            wave = (margin.wave.margin, margin.wave.unstable.m, margin.wave_stable_speed, margin.wave_unstable_speed, margin.blowup),
            kh = (margin.kh.margin, margin.kh.unstable.m, margin.kh_completed, margin.kh.unstable.failure[1:min(end, 60)]),
        )
    )
    w, c, k, r = cases.wave, cases.cylinder, cases.kh, cases.khrt
    cylinder_bounds = c.record.bounds
    return Any[
        ("chkStabCaseSteps", w.steps),
        ("chkStabWaveH", frac(w.start.h)), ("chkStabWaveCap", stab_value(w.start.capillary)),
        ("chkStabWaveAdvEnd", stab_value(w.finish.advective)),
        ("chkStabWaveAdvRatio", @sprintf("%.0f", w.finish.advective / w.start.capillary)),
        ("chkStabCylH", frac(c.start.h)), ("chkStabCylAdv", stab_value(c.start.advective)),
        ("chkStabCylPlain", stab_value(cases.plain.advective)),
        ("chkStabCylReduction", @sprintf("%.1f", cases.plain.advective / c.start.advective)),
        ("chkStabCylOpen", fix3(cases.smallest)),
        ("chkStabCylBoundOne", stab_value(cylinder_bounds[2])),
        ("chkStabCylBoundEnd", stab_value(c.finish.advective)),
        ("chkStabCylGap", @sprintf("%.1f", c.finish.advective / c.start.advective)),
        ("chkStabKhH", frac(k.start.h)), ("chkStabKhAdv", stab_value(k.start.advective)),
        ("chkStabKhCap", stab_value(k.start.capillary)),
        ("chkStabKhAdvEnd", stab_value(k.finish.advective)), ("chkStabKhSteps", k.steps),
        ("chkStabKhrtH", frac(r.h)), ("chkStabKhrtAdv", stab_value(r.advective)),
        ("chkStabKhrtCap", stab_value(r.capillary)), ("chkStabKhrtGrav", stab_value(r.gravity)),
        ("chkStabSelSteps", selection.steps),
        ("chkStabSelGrowth", @sprintf("%.2f", selection.growth)),
        ("chkStabSelStretch", @sprintf("%.7f", selection.stretched)),
        ("chkStabSelShort", sci(selection.short)),
        ("chkStabSelCapped", selection.capped_steps),
        ("chkStabWaveMarginSteps", margin.wave_steps),
        ("chkStabWaveMargin", fmt(margin.wave.margin)),
        ("chkStabWaveUnstable", fmt(margin.wave.unstable.m)),
        ("chkStabWaveStableSpeed", @sprintf("%.2f", margin.wave_stable_speed)),
        ("chkStabWaveUnstableSpeed", @sprintf("%.0f", margin.wave_unstable_speed)),
        ("chkStabWaveBlowup", margin.blowup),
        ("chkStabKhMarginSteps", margin.kh_steps),
        ("chkStabKhMargin", fmt(margin.kh.margin)),
        ("chkStabKhMarginCfl", @sprintf("%.2f", STAB_CFL * margin.kh.margin)),
        ("chkStabKhUnstable", fmt(margin.kh.unstable.m)),
        ("chkStabKhUnstableCfl", @sprintf("%.2f", STAB_CFL * margin.kh.unstable.m)),
        ("chkStabKhFailStep", margin.kh_completed + 1),
    ]
end

# `generate.jl stability`: rewrite only the Chapter 13 macros of checks.tex,
# at the place where the complete run writes them (after the halo census),
# and keep every other line.
function stability_checks_only(write_output)
    macros = stability_outputs(write_output)
    names = Set(String(name) for (name, _) in macros)
    lines = filter(readlines(joinpath(OUTPUT, "checks.tex"))[2:end]) do line
        found = match(r"^\\newcommand\\(\w+)\{", line)
        isnothing(found) || !(found.captures[1] in names)
    end
    anchor = findfirst(line -> startswith(line, "\\newcommand\\chkHaloPhysical{"), lines)
    @assert !isnothing(anchor)
    own = [rstrip(macro_line(name, value)) for (name, value) in macros]
    lines = [lines[1:anchor]; own; lines[(anchor + 1):end]]
    return write_output("checks.tex", join(lines, "\n") * "\n")
end
