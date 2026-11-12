# Projection-only experiment: which CMDGFM constraint row creates the tangential
# phase-velocity jump u_g − u_l in a continuous, resolved shear field?
#   julia --project=benchmark/validation/viscous_kh \
#         benchmark/validation/viscous_kh/cmdgfm_jump/projection_only.jl
# → cmdgfm_jump/results.txt (tables) and cmdgfm_jump/faces.txt (per-face log)
#
# Setup as jump_reproducer.jl (uniform n×n mesh on (0,1)×(−1/2,1/2), periodic in x,
# free-slip walls, equal densities/viscosities, σ = 0). Both phases carry the same
# u = 0.5 tanh(y/δ), v = 0, δ = 0.05, i.e. u_g − u_l = 0 on every face. ONLY the
# two-velocity CMDGFM projection is called (src/two_phase/two_velocity/flow_workspace.jl:97).
# Variants re-solve the assembled CMDGFM system directly (sparse LU) after an
# in-script change of the stencils / right-hand side; src/ is not modified.
using TwoPhaseAMR
using Printf, LinearAlgebra, SparseArrays
BLAS.set_num_threads(1)
const P = TwoPhaseAMR

const OUT = IOBuffer()
const FACES = IOBuffer()
report(args...) = (println(args...); println(OUT, args...))
facelog(args...) = println(FACES, args...)

const δ = 0.05
U(y) = 0.5 * tanh(y / δ)
dU(y) = 0.5 / δ * sech(y / δ)^2

# interface y = η(x; h); `window(x)` selects the x-faces that are measured
const CASES = [
    (name = "flat y=0", η = (x, h) -> 0.0, window = x -> true),
    (name = "flat y=0.37h", η = (x, h) -> 0.37h, window = x -> true),
    (name = "cos a=0.01", η = (x, h) -> 0.01 * cospi(2x), window = x -> true),
    # triangle wave (periodic, slope ±s); faces within 0.1 of the kinks are excluded
    (name = "tri s=0.06", η = (x, h) -> 0.06 * (0.25 - abs(x - 0.5)), window = x -> min(abs(x - 0.5), x, 1 - x) > 0.1),
    (name = "tri s=0.3", η = (x, h) -> 0.3 * (0.25 - abs(x - 0.5)), window = x -> min(abs(x - 0.5), x, 1 - x) > 0.1),
]

function build(n, η)
    h = 1 / n
    mesh = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (8, 8), origin = (0.0, -0.5), extent = (1.0, 1.0),
        periodic = (true, false),
    )
    interface = InterfaceField(mesh, (x, y) -> y - η(x, h); lipschitz = 1)
    physics = TwoPhase(interface; density = (1.0, 1.0), dynamic_viscosity = (1.0e-3, 1.0e-3))
    u0 = (x, y, t, axis) -> axis == 1 ? U(y) : 0.0
    flow = FlowSimulation(mesh, physics; velocity_model = TwoVelocityModel(), velocity = u0, boundary = FreeSlip())
    return mesh, flow
end

snapshot(v) = (copy(v[1].interior), copy(v[2].interior), copy(v[1].boundary), copy(v[2].boundary))
function restore!(v, s)
    copyto!(v[1].interior, s[1]); copyto!(v[2].interior, s[2])
    copyto!(v[1].boundary, s[3]); copyto!(v[2].boundary, s[4])
    return v
end

"max |u_g − u_l| over measured x-faces carrying both phases, and its face index"
function max_jump(ul, ug, mesh, window)
    best, where = 0.0, 0
    for (i, face) in pairs(mesh.faces)
        face.axis == 1 && window(face.center[1]) || continue
        (isfinite(ul[i]) && isfinite(ug[i])) || continue
        d = abs(ug[i] - ul[i])
        d > best && ((best, where) = (d, i))
    end
    return best, where
end

"Integrated outflux of Σ_phases a_phase u_phase (phases selected by `use`), boundary included."
function cell_flux(data, interface, vel, use)
    mesh = interface.mesh
    nx, ny = mesh.block_size
    result = zeros(ncells(mesh))
    la, ga = data.liquid_aperture, data.gas_aperture
    for (i, face) in pairs(mesh.faces)
        m = (use[1] ? P._phase_product(la[i], vel[1].interior[i]) : 0.0) +
            (use[2] ? P._phase_product(ga[i], vel[2].interior[i]) : 0.0)
        result[P._cell_index(face.minus, nx, ny)] += face.area * m
        result[P._cell_index(face.plus, nx, ny)] -= face.area * m
    end
    for (i, face) in pairs(mesh.boundary_faces)
        l, g = P._phase_boundary_face_apertures(interface, interface.geometry, i)
        m = (use[1] ? P._phase_product(l, vel[1].boundary[i]) : 0.0) +
            (use[2] ? P._phase_product(g, vel[2].boundary[i]) : 0.0)
        result[P._cell_index(face.cell, nx, ny)] += face.side * face.area * m
    end
    return result
end

"Row decomposition of `_cmdgfm_normal_velocity_row` (cmdgfm_projection.jl:721-739)."
function row_parts(ul, ug, s)
    face = ug[s.face] - ul[s.face]
    transverse = 0.0
    for (c, w) in zip(s.transverse_faces, s.transverse_weights)
        transverse += s.beta * s.ratio * w * (ug[c] - ul[c])
    end
    inference = s.gas_inference_face > 0 ?
        (s.beta - 1) * (ug[s.gas_inference_face] - ul[s.liquid_inference_face]) : 0.0
    return face, transverse, inference
end

copy_stencil(s; kw...) = P.CMDGFMJumpStencil(
    get(kw, :face, s.face), copy(s.transverse_faces), copy(s.transverse_weights),
    get(kw, :ratio, s.ratio), get(kw, :gas, s.gas_inference_face),
    get(kw, :liquid, s.liquid_inference_face), get(kw, :beta, s.beta),
    s.resolved, s.full_remove,
)

"""
Assemble the CMDGFM system exactly as `_assemble_cmdgfm_cached!` (cmdgfm_projection.jl:895-958),
with a hook on the resolved stencils and on the assembled blocks/rhs, solve it with
sparse LU and apply the phase corrections as `_project_cmdgfm_assembled!` (:1673-1692).
"""
function variant_project!(flow, dt; stencil_map = identity, consistent_inference = false, rhs_mod! = nothing)
    solver, workspace = flow.system, flow.workspace
    model = solver.two_phase
    state = P._two_velocity_flow_workspace!(workspace.two_phase, model)
    projection, pws = state.projection_solver, state.projection_workspace
    interface = projection.interface
    mesh = interface.mesh
    vel = flow.velocity
    P._refresh_pressure_substep!(solver, workspace, model)
    P._mark_inactive_phase_velocities!(vel, interface, Int[])
    data = P._projection_face_data(projection, pws; storage = pws.pressure_cache.face_data)
    pws.pressure_cache.face_data = data
    cache = P._cmdgfm_pressure_cache!(pws.pressure_cache, projection, data; shared_operator = solver.pressure_solver.operator)
    all = P._cmdgfm_jump_stencils(projection, data; incident_faces = cache.incidence, vertex_patches = cache.vertex_patches)
    stencils = [stencil_map(copy_stencil(s)) for s in all if s.resolved]
    mixed = [s.face for s in stencils]
    pressure_jump = P._cmdgfm_pressure_jump(cache, data, mixed, mesh)
    common_gradient = P._cmdgfm_gradient_rows(cache, interface.geometry)
    jp, jb, jrhs, _ = P._cmdgfm_jump_blocks(
        stencils, mixed, data, common_gradient, cache.pressure_indices, vel, dt, Float64;
        pressure_lookup = cache.pressure_lookup,
    )
    ul, ug = vel[1].interior, vel[2].interior
    if consistent_inference
        # inferred jump from same-face jumps: (u_g − ū)(gas face) − (u_l − ū)(liquid face)
        # = γ_lg [u](gas face) + γ_gl [u](liquid face); its ξ columns equal the existing
        # (β−1)γ entries and its common-pressure columns vanish.
        jp = spzeros(size(jp)...)
        for (r, s) in pairs(stencils)
            s.gas_inference_face > 0 || continue
            gf, lf = s.gas_inference_face, s.liquid_inference_face
            jrhs[r] -= (s.beta - 1) * (ug[gf] - ul[lf]) / dt
            γlg = data.liquid_mass[gf] / data.total_mass[gf]
            γgl = data.gas_mass[lf] / data.total_mass[lf]
            jrhs[r] += (s.beta - 1) * (γlg * (ug[gf] - ul[gf]) + γgl * (ug[lf] - ul[lf])) / dt
        end
    end
    matrix = P._cmdgfm_block_matrix!(cache, cache.reduced_pressure, pressure_jump, jp, jb)
    flux = P._cmdgfm_cell_flux!(cache.cell_flux, data, interface, vel)
    np = length(cache.pressure_indices)
    rhs = vcat([flux[c] / dt for c in cache.pressure_indices], jrhs)
    isnothing(rhs_mod!) || rhs_mod!(rhs, np, stencils, ul, ug, dt)
    solution = lu(SparseMatrixCSC(matrix)) \ rhs
    increment = pws.pressure_increment
    P._resize_interface_cache!(increment)
    P._copy_projection_cell_solution!(increment, zip(cache.pressure_indices, view(solution, 1:np)))
    fill!(pws.jump, 0.0)
    pws.jump[mixed] .= view(solution, (np + 1):length(solution))
    gradients = (pws.liquid_gradient, pws.gas_gradient)
    P.cmdgfm_gradient!(gradients, increment, projection, pws, cache.operator, data)
    P._projection_correct_phase_velocities!(vel, gradients, dt, Float64)
    return (; stencils, solution, np)
end

remove_inference_rhs!(rhs, np, stencils, ul, ug, dt) =
    for (r, s) in pairs(stencils)
        s.gas_inference_face > 0 &&
            (rhs[np + r] -= (s.beta - 1) * (ug[s.gas_inference_face] - ul[s.liquid_inference_face]) / dt)
    end
remove_transverse_rhs!(rhs, np, stencils, ul, ug, dt) =
    for (r, s) in pairs(stencils)
        rhs[np + r] -= row_parts(ul, ug, s)[2] / dt
    end
remove_divergence_rhs!(rhs, np, stencils, ul, ug, dt) = (rhs[1:np] .= 0.0)

const VARIANTS = [
    ("direct LU, unmodified", (;)),
    ("rhs: inference term removed", (; rhs_mod! = remove_inference_rhs!)),
    ("stencil: beta=1, no inference", (; stencil_map = s -> s.gas_inference_face > 0 ? copy_stencil(s; beta = 1.0, gas = 0, liquid = 0) : s)),
    ("consistent (same-face) inference", (; consistent_inference = true)),
    ("rhs: transverse term removed (H1)", (; rhs_mod! = remove_transverse_rhs!)),
    ("rhs: divergence rows zeroed (H2)", (; rhs_mod! = remove_divergence_rhs!)),
]

"""
Analyse one projection of the current flow.velocity (jump-free input). Returns a
NamedTuple of table entries; writes the per-face log.
"""
function analyse!(flow, mesh, dt, n, case; label = case.name, logfaces = true)
    h = 1 / n
    solver, workspace = flow.system, flow.workspace
    model = solver.two_phase
    state = P._two_velocity_flow_workspace!(workspace.two_phase, model)
    projection, pws = state.projection_solver, state.projection_workspace
    interface = projection.interface
    vel = flow.velocity
    input = snapshot(vel)
    ul0, ug0 = input[1], input[2]
    jump_in = max_jump(ul0, ug0, mesh, case.window)[1]
    # ---- library projection (unchanged code path) ----
    pressure = flow.pressure
    for b in pressure.blocks
        fill!(b, 0.0)
    end
    stats = P._project_two_velocity!(
        vel, pressure, solver, workspace, state, dt;
        pressure_prepared = false, tolerance = sqrt(eps()), max_iterations = 500,
    )
    ul1, ug1 = copy(vel[1].interior), copy(vel[2].interior)
    data = pws.pressure_cache.face_data
    stencils = [copy_stencil(s) for s in pws.stencils]
    mixed = copy(pws.mixed_faces)
    np = length(pws.pressure_indices)
    rhs = copy(pws.right_hand_side)
    ξ = copy(pws.jump)
    Gc = copy(pws.pressure_gradient.interior)
    Gl, Gg = copy(pws.liquid_gradient.interior), copy(pws.gas_gradient.interior)
    jump_out, iout = max_jump(ul1, ug1, mesh, case.window)
    # ---- divergence of the input (H2 / case d) ----
    restore!(vel, input)
    vol = [mesh.blocks[div(c - 1, prod(mesh.block_size)) + 1].volume for c in 1:ncells(mesh)]
    cut = falses(ncells(mesh))
    nx, ny = mesh.block_size
    for (i, face) in pairs(mesh.faces)
        if 0 < data.liquid_aperture[i] < 1 || 0 < data.liquid_fraction[i] < 1
            cut[P._cell_index(face.minus, nx, ny)] = cut[P._cell_index(face.plus, nx, ny)] = true
        end
    end
    fm = cell_flux(data, interface, vel, (true, true))
    fl = cell_flux(data, interface, vel, (true, false))
    divmix = maximum(abs.(fm) ./ vol)
    divliq = maximum(abs(fl[c]) / vol[c] for c in eachindex(fl) if cut[c]; init = 0.0)
    aperture_sum_dev = maximum(abs(data.liquid_aperture[i] + data.gas_aperture[i] - 1) for i in eachindex(mesh.faces))
    # ---- per-row decomposition (velocity units) ----
    rmax = fmax = tmax = imax = 0.0
    rowafter = 0.0
    for (r, s) in pairs(stencils)
        f, t, inf = row_parts(ul0, ug0, s)
        @assert abs((f + t + inf) - rhs[np + r] * dt) <= 1.0e-12 * (1 + abs(f) + abs(t) + abs(inf))
        rmax = max(rmax, abs(f + t + inf)); fmax = max(fmax, abs(f))
        tmax = max(tmax, abs(t)); imax = max(imax, abs(inf))
        rowafter = max(rowafter, abs(sum(row_parts(ul1, ug1, s))))
    end
    # ---- decomposition of Δ(u_g − u_l): pressure part vs ξ part ----
    dξmax = 0.0; ppart_max = 0.0; djmax = 0.0; dpmax = 0.0
    for i in eachindex(mesh.faces)
        (isfinite(ul1[i]) && isfinite(ug1[i])) || continue
        Δj = (ug1[i] - ul1[i]) - (ug0[i] - ul0[i])
        ξpart = -dt * (data.zeta_gas[i] - data.zeta_liquid[i]) / data.total_mass[i] * ξ[i]
        @assert abs((ug1[i] - ug0[i]) + dt * Gg[i]) <= 1.0e-12 && abs((ul1[i] - ul0[i]) + dt * Gl[i]) <= 1.0e-12
        ppart_max = max(ppart_max, abs(Δj - ξpart))  # = −dt(Gc − Gc): common pressure part
        dξmax = max(dξmax, abs(dt * ξ[i]))
        djmax = max(djmax, abs(Δj))
        dpmax = max(dpmax, abs(dt * Gc[i]))
    end
    # ---- per-face log: resolved x-face rows ----
    ninf = count(s -> s.gas_inference_face > 0, stencils)
    nres_x = count(s -> mesh.faces[s.face].axis == 1, stencils)
    if logfaces
        facelog(@sprintf("\n## %s, δ/h = %.1f (n = %d, dt = %.3e): %d resolved jump rows (%d x-faces), %d with inference", label, δ * n, n, dt, length(stencils), nres_x, ninf))
        facelog("#  face      x        (y−η)/h   c_face  ratio    beta    gas/liq inference Δy/h  inference term  row before  dt·ξ      Δ(u_g−u_l)  pred (1−β)ΔU  2U'h·sgn")
        for s in stencils
            face = mesh.faces[s.face]
            face.axis == 1 || continue
            x, y = face.center
            f, t, inf = row_parts(ul0, ug0, s)
            gy = s.gas_inference_face > 0 ? (mesh.faces[s.gas_inference_face].center[2] - y) / h : NaN
            ly = s.liquid_inference_face > 0 ? (mesh.faces[s.liquid_inference_face].center[2] - y) / h : NaN
            Δj = (ug1[s.face] - ul1[s.face]) - (ug0[s.face] - ul0[s.face])
            facelog(
                @sprintf(
                    "%6d  %.4f  %+.3f  %.3f  %+8.2f  %.4f  %+5.1f %+5.1f   %+.4e   %+.4e  %+.4e  %+.4e  %+.4e  %+.4e",
                    s.face, x, (y - case.η(x, h)) / h, data.liquid_fraction[s.face], s.ratio, s.beta,
                    gy, ly, inf, f + t + inf, -dt * ξ[s.face], Δj, -inf, sign(-inf) * 2dU(y) * h,
                )
            )
        end
    end
    # max-face information
    iout == 0 && (iout = first(i for (i, f) in pairs(mesh.faces) if f.axis == 1))  # no face with both phases
    face = mesh.faces[iout]
    s_at = findfirst(s -> s.face == iout, stencils)
    β_at = isnothing(s_at) ? NaN : stencils[s_at].beta
    r_at = isnothing(s_at) ? NaN : stencils[s_at].ratio
    pred_at = isnothing(s_at) ? NaN : -row_parts(ul0, ug0, stencils[s_at])[3]
    # ---- variants (direct LU on the assembled system) ----
    variants = Float64[]
    for (vname, kw) in VARIANTS
        restore!(vel, input)
        variant_project!(flow, dt; kw...)
        push!(variants, max_jump(vel[1].interior, vel[2].interior, mesh, case.window)[1])
    end
    restore!(vel, input)
    return (;
        label, n, jump_in, jump_out, rmax, fmax, tmax, imax, rowafter, dξmax, ppart_max, djmax, dpmax,
        divmix, divliq, aperture_sum_dev, ninf, nres = length(stencils), nres_x, variants,
        converged = stats.converged, x = face.center[1], yh = (face.center[2] - case.η(face.center[1], h)) / h,
        twoUh = 2dU(face.center[2]) * h, β_at, r_at, pred_at,
    )
end

"Input of the in-step projection: body of step! up to the old-pressure update (as jump_trace.jl)."
function instep_input!(flow, dt)
    velocity, pressure, solver, workspace = flow.velocity, flow.pressure, flow.system, flow.workspace
    geometry_prepared = workspace.transport_geometry_prepared
    workspace.transport_geometry_prepared = false
    model = solver.two_phase
    phase = workspace.two_phase
    time_step, reference_step, current_time, tol, cycles = P._prepare_step_controls!(
        velocity, solver, workspace, dt, dt, 0.0, sqrt(eps()), nothing,
    )
    state = P._two_velocity_flow_workspace!(phase, model)
    P._copy_velocity!(state.old_velocity, velocity)
    P._enforce_phase_velocity_boundary!(state.old_velocity, solver, current_time, phase.inactive_failures)
    P._transport_substep!(state.old_velocity, solver, workspace, time_step, current_time, model; geometry_prepared)
    P._enforce_phase_velocity_boundary!(state.convected, solver, current_time + time_step / 2, phase.inactive_failures)
    P._diffusion_substep!(velocity, solver, workspace, time_step, current_time, model; tolerance = tol, max_cycles = cycles)
    P._enforce_phase_velocity_boundary!(velocity, solver, current_time + time_step, phase.inactive_failures)
    operator = P._two_velocity_pressure_operator!(solver, workspace, state, model)
    P._projection_weighted_gradient!(phase.acceleration, pressure, operator, workspace.projection.elliptic, model.interface.geometry)
    P._add_common_phase_acceleration!(velocity, phase.acceleration, -time_step)
    return velocity
end

function main()
    rows = []
    for case in CASES, n in (32, 64, 128)
        mesh, flow = build(n, case.η)
        dt = stable_timestep(flow; cfl = 0.4)
        push!(rows, analyse!(flow, mesh, dt, n, case))
        println(stderr, "done ", case.name, " n=", n)
    end
    # the reproducer's actual step: projection input = state (f) of jump_trace.jl
    for n in (32, 64, 128)
        case = CASES[3]
        mesh, flow = build(n, case.η)
        dt = stable_timestep(flow; cfl = 0.4)
        instep_input!(flow, dt)
        push!(rows, analyse!(flow, mesh, dt, n, case; label = "cos, in-step input", logfaces = false))
        println(stderr, "done in-step n=", n)
    end
    report("# CMDGFM projection-only experiment (u_g = u_l = 0.5 tanh(y/δ), v = 0, δ = 0.05)\n")
    report("Table A: jump created by ONE library projection (max over measured x-faces) and the jump rows of the jump-free input.")
    report("  row parts in velocity units: face [u]_f, transverse βrΣw[v], inference (β−1)(u_g(gas face) − u_l(liquid face));")
    report("  Δ(u_g−u_l) = pressure part + ξ part, ξ part = −dt (ζ_g−ζ_l)/m ξ = −dt ξ (projection/common.jl:294-319).")
    report(@sprintf("%-20s %5s %9s %9s | %9s %9s %9s %9s | %9s %9s %9s | %5s %5s %5s | %s",
        "case", "δ/h", "jump in", "jump out", "row max", "face", "transv", "infer", "row after", "max dt|ξ|", "p-part", "rows", "x-rows", "infer", "conv"))
    for r in rows
        report(@sprintf("%-20s %5.1f %9.2e %9.3e | %9.3e %9.1e %9.1e %9.3e | %9.1e %9.3e %9.1e | %5d %5d %5d | %s",
            r.label, δ * r.n, r.jump_in, r.jump_out, r.rmax, r.fmax, r.tmax, r.imax, r.rowafter, r.dξmax, r.ppart_max, r.nres, r.nres_x, r.ninf, r.converged))
    end
    report("\nTable B: location of the max jump and the inference prediction there.")
    report(@sprintf("%-20s %5s %7s %8s %8s %7s %10s %10s %9s", "case", "δ/h", "x", "(y−η)/h", "ratio", "beta", "jump out", "−infer", "2|U'|h"))
    for r in rows
        report(@sprintf("%-20s %5.1f %7.4f %+8.3f %+8.2f %7.4f %10.3e %+10.3e %9.3e",
            r.label, δ * r.n, r.x, r.yh, r.r_at, r.β_at, r.jump_out, r.pred_at, r.twoUh))
    end
    report("\nTable C: input divergence (H2) and max dt|∇p_common| (common to both phases).")
    report(@sprintf("%-20s %5s %12s %12s %12s %12s", "case", "δ/h", "max|div mix|", "max|div liq| cut", "|a_l+a_g−1|", "max dt|Gp|"))
    for r in rows
        report(@sprintf("%-20s %5.1f %12.2e %16.3e %12.1e %12.3e", r.label, δ * r.n, r.divmix, r.divliq, r.aperture_sum_dev, r.dpmax))
    end
    report("\nTable D: jump created after an in-script modification (direct sparse-LU solve of the assembled system).")
    report(@sprintf("%-20s %5s ", "case", "δ/h") * join((@sprintf("%-12s", "V$k") for k in 0:(length(VARIANTS) - 1)), " "))
    for r in rows
        report(@sprintf("%-20s %5.1f ", r.label, δ * r.n) * join((@sprintf("%-12.3e", v) for v in r.variants), " "))
    end
    for (k, (vname, _)) in pairs(VARIANTS)
        report("  V$(k - 1): $vname")
    end
    write(joinpath(@__DIR__, "results.txt"), String(take!(OUT)))
    write(joinpath(@__DIR__, "faces.txt"), "# Per-face log (resolved x-face jump rows; library projection of the jump-free input)\n" * String(take!(FACES)))
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
