# Check 01: the stage lists of one step -- sec:overview:step-one (one velocity, one and two phases) and sec:overview:step-two (two velocities).
#
# Label:  sec:overview:step-one, sec:overview:step-two, tab:overview:thesis (stage column)
# Source: docs/techref/chapters/01_overview.tex:322-476 (lists), 336-370 (table); src/single_phase/stepping.jl (step!), src/two_phase/stepping.jl,
#         src/two_phase/two_velocity/stepping.jl (step!), src/navier_stokes/stages.jl
# Method: INSTRUMENTED RUN. `_trace.jl` loads a copy of the package in which the stage functions append their name and selected arguments to a
#         log; the log of one `simulate!`/`step!` is compared with the stage list of the chapter, in order, with the recorded times of the
#         boundary stages (t^n, t^n + dt/2, t^{n+1}) and the time-step checks.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/ch-overview/scripts/step01_stage_order.jl

include(joinpath(@__DIR__, "_fixtures.jl"))

const KEEP = [
    :step!, :_transport_substep!, :_check_transport_timestep, :_body_force_substep!, :_enforce_predictor_velocity_boundary!,
    :_enforce_phase_velocity_boundary!, :_diffusion_substep!, :implicit_midpoint_diffusion!, :_refresh_pressure_substep!,
    :_one_velocity_pressure_solver!, :_phase_force_substep!, :surface_tension!, :gravity!, :_pressure_predictor!, :_project_prepared!,
    :_capture_two_velocity_inherited_flux!, :_two_velocity_pressure_operator!, :_phase_force_pressure_operator!,
    :_projection_weighted_gradient!, :_add_common_phase_acceleration!, :_project_two_velocity!, :reconstruct!,
]

# one step with a fixed dt, from t = 0.25 so that the recorded times are distinguishable from the step size
function one_step(flow; dt = 1.0e-3, time = 0.25)
    flow.time = time
    _, log = Tracer.record() do
        A.step!(flow.velocity, flow.pressure, flow.system, dt, flow.workspace; time)
    end
    return log
end

seq(log; drop = (:reconstruct!,)) = Symbol[e.name for e in stage_log(log, KEEP) if !(e.name in drop)]

dt = 1.0e-3; t0 = 0.25
approx(a, b) = isapprox(a, b; atol = 1.0e-14)

println("== one phase (sec:overview:step-one, items 1-3, 4, 6-8; items 5 'with one phase v3 = v2')")
flow = single_flow()
log = one_step(flow)
expected = [
    :step!, :_transport_substep!, :_check_transport_timestep, :_body_force_substep!, :_enforce_predictor_velocity_boundary!,
    :_diffusion_substep!, :implicit_midpoint_diffusion!, :_refresh_pressure_substep!, :_one_velocity_pressure_solver!,
    :_phase_force_substep!, :_enforce_predictor_velocity_boundary!, :_pressure_predictor!, :_projection_weighted_gradient!,
    :_project_prepared!,
]
check("one phase: stage order", seq(log) == expected, join(seq(log), " > "))
bc = [e.args for e in log if e.name === :_enforce_predictor_velocity_boundary!]
check(
    "one phase: boundary values at t+dt/2 (no geometry stage) and at t+dt after the forces (with covered-face zeroing)",
    length(bc) == 2 && approx(bc[1].time, t0 + dt / 2) && !bc[1].enforce_geometry && approx(bc[2].time, t0 + dt) && bc[2].enforce_geometry,
)
body = only(e for e in log if e.name === :_body_force_substep!)
check("one phase: body acceleration at the midpoint", approx(body.args.midpoint, t0 + dt / 2))
check(
    "one phase: the boundary values at t+dt/2 are imposed also without a body acceleration (force = nothing)",
    findfirst(e -> e.name === :_body_force_substep!, log) < findfirst(e -> e.name === :_enforce_predictor_velocity_boundary!, log),
)
# alternating sweep order: x,y then y,x (two advect! calls)
_, log2 = Tracer.record() do
    for k in 1:3
        A.step!(flow.velocity, flow.pressure, flow.system, dt, flow.workspace; time = t0 + k * dt)
    end
end
axes_seq = [e.args.sweep_axis for e in log2 if e.name === :_dispatch_convection!]
check("one phase: directional sweeps alternate (x,x,y,y | y,y,x,x | ...), one advect! with two component calls per sweep", axes_seq == [2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1] || axes_seq == [1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2], string(axes_seq))

println("\n== two phases, one velocity (same list plus item 5)")
flow = drop_flow()
log = one_step(flow)
expected = [
    :step!, :_transport_substep!, :_check_transport_timestep, :_body_force_substep!, :_enforce_predictor_velocity_boundary!,
    :_diffusion_substep!, :implicit_midpoint_diffusion!, :_refresh_pressure_substep!, :_one_velocity_pressure_solver!,
    :_phase_force_substep!, :surface_tension!, :_add_common_phase_acceleration!, :gravity!, :_add_common_phase_acceleration!,
    :_enforce_predictor_velocity_boundary!, :_pressure_predictor!, :_projection_weighted_gradient!, :_project_prepared!,
]
check("one velocity, sigma>0, g!=0, mu>0: stage order (surface tension, then gravity)", seq(log) == expected, join(seq(log), " > "))
check(
    "the pressure coefficient is refreshed (refresh = true) before the force stage",
    let i = findfirst(e -> e.name === :_one_velocity_pressure_solver!, stage_log(log, KEEP)),
            j = findfirst(e -> e.name === :surface_tension!, log)
        i !== nothing && j !== nothing && any(e -> e.name === :_refresh_pressure_substep!, log)
    end,
)
# the pressure coefficient of the force stage and the projection is built from the TRANSPORTED fraction C^{n+1}, not from C^n
let
    mesh = unit_mesh(periodic = (true, false))
    flow = drop_flow(mesh = mesh, boundary = A.FreeSlip(), velocity = (0.3, 0.0), viscosity = (0.0, 0.0))
    interface = flow.system.two_phase.interface
    old_fraction = [copy(b) for b in interface.fraction.blocks]
    before = copy(flow.system.pressure_solver.operator.face_coefficient)
    one_step(flow; dt = 1.0e-2)
    after = flow.system.pressure_solver.operator.face_coefficient
    rebuilt = inv.(A.mixture_density(interface, flow.system.two_phase.density))
    moved = any(old_fraction[i] != interface.fraction.blocks[i] for i in eachindex(old_fraction))
    check(
        "the face coefficient used by the force stage equals 1/rho of the TRANSPORTED fraction (the drop moved; the coefficient changed)",
        moved && maximum(abs.(after .- rebuilt)) <= 1.0e-14 * maximum(abs.(rebuilt)) && maximum(abs.(after .- before)) > 0,
        "max |after - rebuilt| = $(maximum(abs.(after .- rebuilt))), max |after - before| = $(maximum(abs.(after .- before)))",
    )
end
check(
    "unequal densities: the preconditioner of the pressure operator is refreshed (after the coefficient, before the forces); equal densities: not",
    let log_u = log, log_e = one_step(drop_flow(density = (1.0, 1.0)))
        iu = findfirst(e -> e.name === :_refresh_geometric_multigrid!, log_u)
        ir = findfirst(e -> e.name === :_refresh_pressure_substep!, log_u)
        ip = findfirst(e -> e.name === :_phase_force_substep!, log_u)
        iu !== nothing && ir < iu < ip && !any(e -> e.name === :_refresh_geometric_multigrid!, log_e)
    end,
)
log0 = one_step(drop_flow(viscosity = (0.0, 0.0), sigma = 0.0, gravity = (0.0, 0.0)))
check(
    "mu = 0 omits the viscous solve, sigma = 0 and g = 0 omit both forces",
    !(:implicit_midpoint_diffusion! in names_of(log0)) && !(:surface_tension! in names_of(log0)) && !(:gravity! in names_of(log0)) &&
        :_diffusion_substep! in names_of(log0),
)
log_s = one_step(drop_flow(gravity = (0.0, 0.0)))
log_g = one_step(drop_flow(sigma = 0.0))
check(
    "sigma>0 only: surface tension only; g!=0 only: gravity only",
    :surface_tension! in names_of(log_s) && !(:gravity! in names_of(log_s)) && :gravity! in names_of(log_g) && !(:surface_tension! in names_of(log_g)),
)
# the interface is advanced by the transport stage: WY sweeps in alternating order
_, logw = Tracer.record() do
    for k in 1:3
        A.step!(flow.velocity, flow.pressure, flow.system, dt, flow.workspace; time = t0 + k * dt)
    end
end
wy_axes = [e.args.axis for e in logw if e.name === :_compute_directional_wy_fluxes!]
check("two phases: WY sweeps alternate in order", wy_axes == [1, 2, 2, 1, 1, 2] || wy_axes == [2, 1, 1, 2, 2, 1], string(wy_axes))

println("\n== two velocities, CMDGFM + frozen interface fluxes, no solid (sec:overview:step-two)")
tv = A.TwoVelocityModel()
flow = drop_flow(model = tv)
log = one_step(flow)
expected = [
    :step!, :_enforce_phase_velocity_boundary!, :_transport_substep!, :_check_transport_timestep, :_body_force_substep!,
    :_enforce_phase_velocity_boundary!, :_diffusion_substep!, :implicit_midpoint_diffusion!, :_enforce_phase_velocity_boundary!,
    :_two_velocity_pressure_operator!, :_refresh_pressure_substep!, :_phase_force_substep!, :surface_tension!, :_add_common_phase_acceleration!,
    :gravity!, :_add_common_phase_acceleration!, :_projection_weighted_gradient!, :_add_common_phase_acceleration!, :_project_two_velocity!,
    :_projection_weighted_gradient!,   # inside the projection
]
check("two velocities: stage order", seq(log) == expected, join(seq(log), " > "))
bc = [e.args.time for e in log if e.name === :_enforce_phase_velocity_boundary!]
check(
    "two velocities: boundary values at t^n, t^n+dt/2 and t^{n+1} (the last one BEFORE the forces)",
    length(bc) == 3 && approx(bc[1], t0) && approx(bc[2], t0 + dt / 2) && approx(bc[3], t0 + dt),
)
acc = [e.args.dt for e in log if e.name === :_add_common_phase_acceleration!]
check("two velocities: forces add +dt F, the old-pressure predictor subtracts (dt = -dt)", length(acc) == 3 && approx(acc[1], dt) && approx(acc[2], dt) && approx(acc[3], -dt))
ip = findfirst(e -> e.name === :_two_velocity_pressure_operator!, log)
ir = findfirst(e -> e.name === :reconstruct!, log[ip:end])
check("two velocities, CMDGFM, no solid: the transported interface is reconstructed in the pressure-map stage", ir !== nothing && log[ip + ir - 1].name === :reconstruct! && ip + ir - 1 < findfirst(e -> e.name === :_refresh_pressure_substep!, log[ip:end]) + ip - 1)
pj = only(e for e in log if e.name === :_project_two_velocity!)
check("two velocities, CMDGFM: the projection is called with the pressure prepared (old-pressure predictor)", pj.args.pressure_prepared === true)
check("a direct step! checks the bounds in the transport stage (geometry not prepared)", only(e for e in log if e.name === :_transport_substep!).args.geometry_prepared === false)
check("no saved flux for dt = dtref", !(:_capture_two_velocity_inherited_flux! in names_of(log)))

println("\n== two velocities with a solid (CMDGFM uses its cached operator)")
flow = drop_flow(model = tv, solid = true)
log = one_step(flow)
nm = seq(log; drop = ())
ip = findfirst(==(:_two_velocity_pressure_operator!), nm)
check(
    "CMDGFM with a solid: the cached operator is used, the one-velocity coefficient is not refreshed",
    ip !== nothing && nm[ip + 1] === :_phase_force_pressure_operator! && !(:_refresh_pressure_substep! in nm),
    join(nm[ip:end], " > "),
)

println("\n== two velocities, shortened step (dt < dtref) saves the mixture flux (CMDGFM only)")
flow = drop_flow(model = tv)
_, log = Tracer.record() do
    A.simulate!(flow, 0.0025; dt = 1.0e-3)   # steps 1e-3, 1e-3, 0.5e-3 (dtref = 1e-3)
end
steps = [e for e in log if e.name === :step!]
caps = [i for (i, e) in pairs(log) if e.name === :_capture_two_velocity_inherited_flux!]
check("simulate! with a shortened last step: dt = (1e-3, 1e-3, 5e-4), dtref = 1e-3", length(steps) == 3 && approx(steps[3].args.dt, 5.0e-4) && all(s -> approx(s.args._reference_dt, 1.0e-3), steps), string([(s.args.dt, s.args._reference_dt) for s in steps]))
istep3 = findlast(e -> e.name === :step!, log)
check(
    "the flux is saved only in the shortened step, right after the start boundary values and before the transport",
    length(caps) == 1 && caps[1] > istep3 && log[caps[1] - 1].name in (:_enforce_phase_velocity_boundary!, :_prepare_step_controls!) ||
        (length(caps) == 1 && caps[1] > istep3 && findfirst(e -> e.name === :_transport_substep!, log[istep3:end]) + istep3 - 1 > caps[1]),
)

println("\n== two velocities, simulate! with automatic step: the bound evaluated by simulate! prepares the transport")
flow = drop_flow(model = tv)
_, log = Tracer.record() do
    A.simulate!(flow, 0.002)   # dt = nothing: automatic step from the bounds
end
ist = findfirst(e -> e.name === :step!, log)
check(
    "frozen carrier, dt = nothing: stable_timestep before step!, transport called with geometry_prepared = true, no second check",
    any(e -> e.name === :stable_timestep, log[1:ist]) &&
        only(e for e in log if e.name === :_transport_substep!).args.geometry_prepared === true &&
        !(:_check_transport_timestep in names_of(log)),
)
flow = drop_flow(model = tv)
_, log = Tracer.record() do
    A.simulate!(flow, 0.002; dt = 1.0e-3)
end
check(
    "frozen carrier, prescribed dt: the transport stage checks the bounds itself",
    all(e -> e.args.geometry_prepared === false, filter(e -> e.name === :_transport_substep!, log)) && :_check_transport_timestep in names_of(log),
)
flow = drop_flow(model = A.TwoVelocityModel(interface_transport = A.ProjectedLiquidVelocity()))
_, log = Tracer.record() do
    A.simulate!(flow, 0.002)
end
check(
    "projected-liquid carrier, dt = nothing: geometry_prepared = false (the step checks the bounds itself)",
    all(e -> e.args.geometry_prepared === false, filter(e -> e.name === :_transport_substep!, log)) && :_check_transport_timestep in names_of(log),
)
_, log = Tracer.record() do
    A.simulate!(flow, flow.time + 0.004; dt = 1.0e-3)
end
wy = [e.args.axis for e in log if e.name === :_compute_directional_wy_fluxes!]
check("projected-liquid carrier: alternating sweeps over 4 steps", wy == [1, 2, 2, 1, 1, 2, 2, 1] || wy == [2, 1, 1, 2, 2, 1, 1, 2], string(wy))

println("\n== two velocities, mimetic projection (legacy)")
mim = A.TwoVelocityModel(pressure_projection = A.MimeticProjection())
flow = drop_flow(model = mim)
log = one_step(flow)
nm = seq(log)
expected = [
    :step!, :_enforce_phase_velocity_boundary!, :_transport_substep!, :_check_transport_timestep, :_body_force_substep!,
    :_enforce_phase_velocity_boundary!, :_diffusion_substep!, :implicit_midpoint_diffusion!, :_enforce_phase_velocity_boundary!,
    :_refresh_pressure_substep!, :_phase_force_substep!, :surface_tension!, :_add_common_phase_acceleration!, :gravity!,
    :_add_common_phase_acceleration!, :_project_two_velocity!, :_projection_weighted_gradient!,
]
check("mimetic, no solid, phase forces: one-velocity operator refreshed, forces, projection WITHOUT old-pressure predictor", nm == expected, join(nm, " > "))
check("mimetic: the projection is called with pressure_prepared = phase_forces (true)", only(e for e in log if e.name === :_project_two_velocity!).args.pressure_prepared === true)

log = one_step(drop_flow(model = mim, sigma = 0.0, gravity = (0.0, 0.0)))
nm = seq(log)
check(
    "mimetic without phase forces: no pressure map is formed (no refresh, no force stage), pressure_prepared = false",
    !(:_refresh_pressure_substep! in nm) && !(:_phase_force_substep! in nm) && !(:_phase_force_pressure_operator! in nm) &&
        only(e for e in log if e.name === :_project_two_velocity!).args.pressure_prepared === false,
    join(nm, " > "),
)
log = one_step(drop_flow(model = mim, solid = true))
nm = seq(log; drop = ())
check(
    "mimetic with a solid and forces: its cached operator (not the one-velocity refresh) forms the map",
    :_phase_force_pressure_operator! in nm && !(:_refresh_pressure_substep! in nm),
    join(nm, " > "),
)
flow = drop_flow(model = mim)
_, log = Tracer.record() do
    A.simulate!(flow, 0.0025; dt = 1.0e-3)
end
check("mimetic, shortened step: no saved flux (absolute projection)", !(:_capture_two_velocity_inherited_flux! in names_of(log)))

println("\n== one velocity, shortened step: reference step passed to the projection")
flow = drop_flow()
_, log = Tracer.record() do
    A.simulate!(flow, 0.0025; dt = 1.0e-3)
end
pr = [(e.args.dt, e.args._reference_dt) for e in log if e.name === :_project_prepared!]
check("one velocity: projection receives (dt, dtref) = (1e-3,1e-3), (1e-3,1e-3), (5e-4,1e-3)", length(pr) == 3 && approx(pr[3][1], 5.0e-4) && all(p -> approx(p[2], 1.0e-3), pr), string(pr))

finish("step01_stage_order")
