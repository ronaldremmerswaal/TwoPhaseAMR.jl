# Per-sub-step budget of the two-velocity capillary standing wave (#178).
#
# `budget_step!` is a copy of `step!(::TwoPhaseVelocity, ...)`
# (src/two_phase/two_velocity/stepping.jl:277-364) with snapshots after every
# sub-step and optional in-script hooks. src/ is not modified.
using LinearAlgebra, Printf, Test, TwoPhaseAMR
const TP = TwoPhaseAMR
include(joinpath(@__DIR__, "wave_case.jl"))

# ---------------------------------------------------------------- setup ----
"""Build the `wave_case` flow (test/integration/two_phase_waves.jl:86-154)."""
function wave_flow(n, density_ratio; velocity_model = :two_velocity)
    wavelength = 1.0
    wavenumber = 2pi
    liquid_density = 1.0
    gas_density = liquid_density / density_ratio
    surface_tension = 1.0
    kinematic_viscosity = sqrt(surface_tension * liquid_density * wavelength / 3000)
    viscosity = (liquid_density * kinematic_viscosity, gas_density * kinematic_viscosity)
    levelset = (x, y) -> y - 0.25 - wavelength / 100 * cospi(2x / wavelength)
    mesh = _two_phase_wave_mesh(:uniform, n)
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    initial_velocity = velocity_model === :one_velocity ? nothing :
        ((x, y, t, axis) -> 0.0, (x, y, t, axis) -> 0.0)
    flow = FlowSimulation(
        mesh, TwoPhase(
            interface; dynamic_viscosity = viscosity,
            density = (liquid_density, gas_density),
            surface_tension = SurfaceTension(surface_tension; method = CGFM()),
            gravity = (0.0, -0.0),
        );
        velocity = initial_velocity,
        velocity_model = velocity_model === :one_velocity ? OneVelocityModel() :
            TwoVelocityModel(; pressure_projection = CMDGFMProjection()),
        boundary = FreeSlip(),
    )
    omega0 = sqrt(surface_tension * wavenumber^3 / (liquid_density + gas_density))
    final_time = 0.25 * 2pi / omega0
    reference = t -> prosperetti_amplitude(
        t; liquid_density, gas_density, kinematic_viscosity,
        gravity = 0.0, surface_tension, wavenumber,
    )
    return (; flow, interface, final_time, reference)
end

# ---------------------------------------------------------- measurement ----
"""Staggered liquid fraction of every interior face (same as the projection)."""
face_fractions(interface) = [
    clamp(TP._staggered_liquid_fraction(interface, interface.geometry, f), 0.0, 1.0)
        for f in interface.mesh.faces
]

snapshot(v::TwoPhaseVelocity) = (copy(v[1].interior), copy(v[2].interior))

# :hook is the in-script variant applied after the projection (zero for the baseline).
const STAGES = (:transport, :diffusion, :forces, :old_gradient, :projection, :hook)

"""
Per-stage change on faces where both phases are finite in both snapshots:
`jx`, `jy` = max |Δ(u_g - u_l)| on x- / y-faces, `mix` = max |Δū| of the
mass-weighted mixture velocity ū = (m_l u_l + m_g u_g)/m (fractions `c`).
"""
function stage_change(before, after, c, rho, axis)
    jx = jy = mix = 0.0
    for i in eachindex(c)
        lb, gb = before[1][i], before[2][i]
        la, ga = after[1][i], after[2][i]
        ml, mg = rho[1] * c[i], rho[2] * (1 - c[i])
        mb = (ml * (isfinite(lb) ? lb : 0.0) + mg * (isfinite(gb) ? gb : 0.0)) / (ml + mg)
        ma = (ml * (isfinite(la) ? la : 0.0) + mg * (isfinite(ga) ? ga : 0.0)) / (ml + mg)
        mix = max(mix, abs(ma - mb))
        all(isfinite, (lb, gb, la, ga)) || continue
        d = abs((ga - la) - (gb - lb))
        axis[i] == 1 ? (jx = max(jx, d)) : (jy = max(jy, d))
    end
    return (; jx, jy, mix)
end

function jump_size(v, axis)
    jx = jy = 0.0
    for i in eachindex(v[1])
        l, g = v[1][i], v[2][i]
        (isfinite(l) && isfinite(g)) || continue
        axis[i] == 1 ? (jx = max(jx, abs(g - l))) : (jy = max(jy, abs(g - l)))
    end
    return (; jx, jy)
end

# ------------------------------------------------------ instrumented step ----
"""
Copy of the two-velocity `step!` (src/two_phase/two_velocity/stepping.jl:277-364)
with a snapshot after every sub-step. `hooks.projection(velocity, flow)` is an
in-script variant applied after the projection (recorded as stage `:hook`).
"""
function budget_step!(flow, dt, reference_dt, time; hooks = (;), record = nothing)
    velocity, pressure, solver, workspace =
        flow.velocity, flow.pressure, flow.system, flow.workspace
    T = Float64
    tolerance, max_cycles = 1.0e-9, 1000
    geometry_prepared = false
    workspace.transport_geometry_prepared = false
    model = solver.two_phase
    phase = workspace.two_phase
    time_step, reference_step, current_time, dimensionless_tolerance, cycle_limit =
        TP._prepare_step_controls!(
        velocity, solver, workspace, dt, reference_dt, time, tolerance, max_cycles,
    )
    state = TP._two_velocity_flow_workspace!(phase, model)
    TP._copy_velocity!(state.old_velocity, velocity)
    TP._enforce_phase_velocity_boundary!(
        state.old_velocity, solver, current_time, phase.inactive_failures,
    )
    snaps = Pair{Symbol, Any}[:start => snapshot(state.old_velocity)]
    inherited_flux = time_step == reference_step ? nothing :
        TP._capture_two_velocity_inherited_flux!(solver, workspace, state)
    TP._transport_substep!(
        state.old_velocity, solver, workspace, time_step, current_time, model;
        geometry_prepared,
    )
    midpoint = current_time + time_step / 2
    convected = state.convected
    TP._body_force_substep!(convected, time_step, midpoint, flow.force)
    TP._enforce_phase_velocity_boundary!(convected, solver, midpoint, phase.inactive_failures)
    push!(snaps, :transport => snapshot(convected))
    TP._diffusion_substep!(
        velocity, solver, workspace, time_step, current_time, model;
        tolerance = dimensionless_tolerance, max_cycles = cycle_limit,
    )
    TP._enforce_phase_velocity_boundary!(
        velocity, solver, current_time + time_step, phase.inactive_failures,
    )
    push!(snaps, :diffusion => snapshot(velocity))
    pressure_operator = TP._two_velocity_pressure_operator!(solver, workspace, state, model)
    TP._phase_force_substep!(velocity, workspace, time_step, model, pressure_operator)
    push!(snaps, :forces => snapshot(velocity))
    TP._projection_weighted_gradient!(
        phase.acceleration, pressure, pressure_operator,
        workspace.projection.elliptic, model.interface.geometry,
    )
    TP._add_common_phase_acceleration!(velocity, phase.acceleration, -time_step)
    push!(snaps, :old_gradient => snapshot(velocity))
    inherited_cell_flux = isnothing(inherited_flux) ? nothing :
        TP._copy_from_field!(
        workspace.projection.elliptic.input, inherited_flux,
        pressure_operator.references,
    )
    stats = TP._project_two_velocity!(
        velocity, pressure, solver, workspace, state, time_step;
        pressure_prepared = true, tolerance = dimensionless_tolerance,
        _inherited_cell_flux = inherited_cell_flux, _reference_dt = reference_step,
        max_iterations = cycle_limit,
    )
    push!(snaps, :projection => snapshot(velocity))
    if haskey(hooks, :projection)
        hooks.projection(velocity, flow)
        push!(snaps, :hook => snapshot(velocity))
    end
    stats.converged || error("projection did not converge: $stats")
    isnothing(record) || record(snaps, stats)
    return stats
end

"""Run one case to the quarter period; returns error and per-stage budget."""
function run_case(n, density_ratio, factor; hooks = (;), verbose = false)
    (; flow, interface, final_time, reference) = wave_flow(n, density_ratio)
    mesh = interface.mesh
    axis = [f.axis for f in mesh.faces]
    rho = flow.system.two_phase.density
    initial_moment = _cosine_wave_amplitude(interface)
    dt = factor * stable_timestep(flow)
    selected = dt
    times = [0.0]
    amplitudes = [1.0]
    sums = Dict(s => zeros(3) for s in STAGES)   # Σ_steps of (jx, jy, mix)
    final_jump = Ref((; jx = 0.0, jy = 0.0))
    nsteps = 0
    time = 0.0
    while final_time - time > 16eps() * final_time
        if factor == 1
            # simulate!(dt = nothing): keep the initial proposal while stable.
            selected = min(selected, stable_timestep(flow))
        end
        remaining = final_time - time
        step = remaining - selected < 1.0e-6 * selected ? remaining : selected
        record = (snaps, stats) -> begin
            c = face_fractions(interface)
            for k in 2:length(snaps)
                d = stage_change(snaps[k - 1].second, snaps[k].second, c, rho, axis)
                s = sums[snaps[k].first]
                s[1] += d.jx; s[2] += d.jy; s[3] += d.mix
            end
            final_jump[] = jump_size(snaps[end].second, axis)
        end
        budget_step!(flow, step, max(selected, step), time; hooks, record)
        time = step == remaining ? final_time : time + step
        nsteps += 1
        push!(times, time)
        push!(amplitudes, _cosine_wave_amplitude(interface) / initial_moment)
    end
    errors = amplitudes .- reference.(times)
    error = _trapezoidal_l2(times, errors)
    verbose && @printf("n=%d DR=%g factor=%g steps=%d error=%.6g\n", n, density_ratio, factor, nsteps, error)
    return (; error, nsteps, dt, sums, final_jump = final_jump[], times, amplitudes, errors, flow)
end
