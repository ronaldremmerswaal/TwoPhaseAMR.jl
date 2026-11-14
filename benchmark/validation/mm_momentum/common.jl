# Shared helpers of the metric-consistent momentum transport study.
#   julia --project=/home/ronald/Documents/wt-mm-momentum <script>.jl
using LinearAlgebra, Printf, SparseArrays, Statistics
BLAS.set_num_threads(1)
using TwoPhaseAMR
const TP = TwoPhaseAMR
const WT = "/home/ronald/Documents/wt-mm-momentum"
@assert startswith(Base.active_project(), WT)
const OUT = "/home/ronald/Documents/mm-runs/mm_momentum"


# Activity rule of the phase values (env MM_ACTIVITY=volume|aperture), small-
# aperture ties (MM_TIES=1|0) and the merge factor κ (MM_KAPPA); output suffix.
const ACTIVITY = Symbol(get(ENV, "MM_ACTIVITY", "volume"))
const TIES = get(ENV, "MM_TIES", "1") == "1"
TP._PHASE_ACTIVITY[] = ACTIVITY
TP._MIMETIC_SMALL_TIES[] = TIES
TP._MIMETIC_MERGE_KAPPA[] = parse(Float64, get(ENV, "MM_KAPPA", "1.0"))
const PLAIN = get(ENV, "MM_APERTURE", "current") == "plain"
TP._APERTURE_PLAIN[] = PLAIN
const SUFFIX = (ACTIVITY === :aperture ? "_ap" * (TIES ? "" : "nt") : "") * (PLAIN ? "_pl" : "") * get(ENV, "MM_SUFFIX", "")
"Designs: label => (mode, theta)."
const DESIGNS = (
    D0 = (:none, 1.0),     # current: W transport only
    DI = (:theta, 1.0),    # increment form 𝓜⁺Δu = W⁺Δu_W
    DS = (:theta, 0.5),    # skew/energy form
    DP = (:increment_conservative, 1.0),  # DI + uniform momentum restoration
    DF = (:theta, 0.0),    # flux form (momentum exact)
    DG = (:redistribute, 0.5),  # DS + skew redistribution Y + remainder
    DGP = (:redistribute, 0.5), # DG + uniform momentum restoration
)

function set_design!(name)
    mode, theta = DESIGNS[name]
    TP._MIMETIC_TRANSPORT[] = mode
    TP._MIMETIC_TRANSPORT_THETA[] = theta
    TP._MIMETIC_TRANSPORT_FIX[] = name === :DGP
    return nothing
end

"Periodic (or walled) uniform mesh with an interface given by a level set."
function make_flow(levelset; n, density, sigma = 0.0, periodic = (true, true),
        velocity = nothing, viscosity = (0.0, 0.0), fraction_floor = sqrt(eps()),
        ctu_threshold = nothing, scheme = LaxWendroff(), boundary = FreeSlip(), gravity = (0.0, 0.0))
    mesh = BlockTreeMesh(block_size = (n, n), periodic = periodic)
    interface = InterfaceField(mesh, levelset; lipschitz = 1, fraction_floor)
    physics = TwoPhase(interface; density, dynamic_viscosity = viscosity,
        surface_tension = SurfaceTension(sigma; method = CGFM()), gravity)
    numerics = NavierStokesOptions(advection = AdvectionOptions(; scheme))
    flow = FlowSimulation(mesh, physics; velocity_model = TwoVelocityModel(pressure_projection = MimeticProjection()),
        boundary, velocity, numerics)
    if !isnothing(ctu_threshold)
        flow.workspace.two_phase.advection = WYAdvectionWorkspace(interface; ctu_threshold)
    end
    return flow, interface
end

state_of(flow) = TP._two_velocity_flow_workspace!(flow.workspace.two_phase, flow.system.two_phase)

"𝓜-momentum per column (liquid x, liquid y, gas x, gas y, mixture x, mixture y) and 𝓜-energy of `velocity` with the current assembly."
function metric_invariants(flow; velocity = flow.velocity)
    state = state_of(flow)
    solver, work = state.projection_solver, state.projection_workspace
    assembly = TP._mimetic_assemble!(solver, work)
    values = Float64[]
    TP._mimetic_gather!(values, assembly, velocity)
    M = TP._mimetic_metric_matrix(work)
    mu = M * values
    Z = TP._mimetic_constant_fields(assembly, solver.interface.mesh)
    return TP._mimetic_columns(Z, mu, assembly, solver.interface.mesh), dot(values, mu) / 2,
        TP._mimetic_w_momentum(assembly, solver.interface.mesh, values)
end

"Max |u - U·n| over active phase values of `velocity` (both phases), relative to |U|."
function constant_deviation(velocity, mesh, U)
    worst = 0.0
    for phase in 1:2, (index, face) in pairs(mesh.faces)
        u = velocity[phase].interior[index]
        isfinite(u) || continue
        worst = max(worst, abs(u - U[face.axis]))
    end
    return worst / hypot(U...)
end

"Smallest positive min(C, 1 - C) over the cells."
function min_fraction(interface)
    best = 1.0
    for ref in TP._cell_references(interface.mesh)
        c = TP._conditional_fraction(interface, ref)
        m = min(c, 1 - c)
        m > 0 && (best = min(best, m))
    end
    return best
end

do_step!(flow, dt) = (s = step!(flow.velocity, flow.pressure, flow.system, dt, flow.workspace; time = flow.time, tolerance = 1.0e-10, max_cycles = 3000); flow.time += dt; flow.steps += 1; s)
