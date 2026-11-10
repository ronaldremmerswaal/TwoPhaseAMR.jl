# Chapter 11 (operators and linear solvers): iteration counts and convergence
# factors of representative linear systems with and without their default
# preconditioner, the spectrum and symmetry of the projected V-cycle, the
# null spaces of the cell operator, hierarchy reuse and refresh, and the
# allocations of repeated (warm) solves. Included by generate.jl after
# projection.jl; `generate.jl solvers` regenerates only these numbers (see the
# end of generate.jl). The assertions fail if the code no longer matches the
# statements of Chapter 11.

using LinearAlgebra: I, Symmetric, cholesky, diag, eigvals, ldiv!, nullspace

# --- Fixtures --------------------------------------------------------------------

const SOLV_RTOL = 1.0e-10
const SOLV_MAX_CYCLES = 20_000
const SOLV_DENSITY = (1000.0, 1.0)
const SOLV_VISCOSITY = (0.1, 0.002)
# A drop on the unit square (walls); the refinement band follows its circle.
const SOLV_DROP = (; centre = (0.513, 0.468), radius = 0.2371, band = 0.02)
const SOLV_DT = 0.01

solv_drop(x, y) = hypot(x - SOLV_DROP.centre[1], y - SOLV_DROP.centre[2]) - SOLV_DROP.radius
# Solid band |y - 0.46| < 0.17: two fluid components, cut and covered cells.
solv_band_solid(x, y) = abs(y - 0.46) - 0.17

# Deterministic noise in [-1/2, 1/2): a right-hand side with all frequencies.
solv_noise(index) = (value = sin(12.9898 * index) * 43758.5453; value - floor(value) - 0.5)

solv_uniform(n) = BlockTreeMesh(;
    base_mesh = (n, n), block_size = (8, 8), periodic = (false, false),
)

# Refine every block whose box comes within `band` of the drop circle.
function solv_near_circle(origin, extent; radius = SOLV_DROP.radius)
    xs = (origin[1], origin[1] + extent[1])
    ys = (origin[2], origin[2] + extent[2])
    cx, cy = SOLV_DROP.centre
    nearest = hypot(clamp(cx, xs...) - cx, clamp(cy, ys...) - cy)
    farthest = maximum(hypot(x - cx, y - cy) for x in xs, y in ys)
    return nearest - SOLV_DROP.band <= radius <= farthest + SOLV_DROP.band
end

solv_amr(n; levels = 2, radius = SOLV_DROP.radius) = BlockTreeMesh(;
    base_mesh = (n, n), block_size = (8, 8), periodic = (false, false),
    max_level = levels,
    refine = (key, origin, extent) -> solv_near_circle(origin, extent; radius),
)

solv_drop_field(mesh) = InterfaceField(mesh, solv_drop; lipschitz = 1)

function solv_noise_cells(operator)
    field = CellField(operator.mesh)
    for (index, ref) in pairs(operator.references)
        AMR._set!(field, solv_noise(index), ref)
    end
    return field
end

function solv_noise_faces(mesh)
    field = FaceVelocity(mesh)
    for index in eachindex(field.interior)
        field.interior[index] = solv_noise(index)
    end
    return field
end

solv_factor(stats, cycles) = (stats.final_l2 / stats.initial_l2)^(1 / max(cycles, 1))

# --- Cell elliptic systems (Sections "systems", "multigrid", "measured") ---------

# One solve of L p = b for a noise right-hand side to rtol = 1e-10 of the
# initial strong residual, then the same solve repeated from zero (warm).
function solv_poisson(operator; preconditioner = :tree_geometric, options = (;))
    solver = EllipticSolver(operator; preconditioner, preconditioner_options = options)
    rhs = solv_noise_cells(operator)
    solution = CellField(operator.mesh)
    stats = solve!(solution, solver, rhs; rtol = SOLV_RTOL, max_cycles = SOLV_MAX_CYCLES)
    @assert stats.converged
    result = (;
        unknowns = length(operator.volumes), cycles = stats.cycles,
        factor = solv_factor(stats, stats.cycles), method = stats.method,
        levels = isnothing(preconditioner) ? 0 : length(solver.preconditioner.levels),
        sizes = isnothing(preconditioner) ? Int[] :
            [length(level.volumes) for level in solver.preconditioner.levels],
        components = length(stats.removed_rhs_component_means),
        weights = isnothing(preconditioner) ? Float64[] :
            [level.relaxation_weight for level in solver.preconditioner.levels],
    )
    for block in eachindex(operator.mesh.blocks)
        fill!(interior(solution, block), 0.0)
    end
    bytes = @allocated solve!(solution, solver, rhs; rtol = SOLV_RTOL, max_cycles = SOLV_MAX_CYCLES)
    return (; result..., bytes, repeat_cycles = solver.stats.cycles)
end

function solv_pressure_cases()
    drop_operator(mesh) = PoissonOperator(solv_drop_field(mesh); density = SOLV_DENSITY)
    split_geometry(n) = GeometryField(solv_uniform(n), solv_band_solid; lipschitz = 1)
    cases = Any[]
    for n in (64, 128, 256)
        push!(cases, (; name = "uniform$n", operator = PoissonOperator(solv_uniform(n)), options = (;)))
    end
    push!(cases, (; name = "amr32", operator = PoissonOperator(solv_amr(32)), options = (;)))
    push!(cases, (; name = "amr64", operator = PoissonOperator(solv_amr(64)), options = (;)))
    push!(cases, (; name = "drop128", operator = drop_operator(solv_uniform(128)), options = (;)))
    push!(cases, (; name = "dropamr", operator = drop_operator(solv_amr(64)), options = (;)))
    geometry = split_geometry(128)
    push!(cases, (; name = "split128", operator = PoissonOperator(geometry), options = (; geometry)))
    return map(cases) do case
        with = solv_poisson(case.operator; options = case.options)
        without = solv_poisson(case.operator; preconditioner = nothing)
        (; case.name, with, without)
    end
end

# Projected V-cycle as a matrix B (columns B e_j): symmetry, range in the
# compatible subspace, and the spectrum of B A there (Z' A Z = L L').
function solv_vcycle_check(operator; options = (;))
    preconditioner = TreeGeometricMultigridPreconditioner(operator; options...)
    n = length(operator.volumes)
    B = zeros(n, n)
    unit = zeros(n)
    column = zeros(n)
    for j in 1:n
        unit[j] = 1.0
        ldiv!(column, preconditioner, unit)
        B[:, j] .= column
        unit[j] = 0.0
    end
    A = Matrix(operator.matrix)
    active = findall(>(0), operator.volumes)
    labels = AMR._elliptic_component_labels(operator)
    anchors = operator.components.anchors
    C = zeros(length(anchors), length(active))
    for (k, root) in pairs(anchors), (i, cell) in pairs(active)
        labels[cell] == root && (C[k, i] = operator.volumes[cell])
    end
    Z = isempty(anchors) ? Matrix(1.0I, length(active), length(active)) : nullspace(C)
    Baa = B[active, active]
    L = cholesky(Symmetric(Z' * A[active, active] * Z)).L
    M = L' * (Z' * Baa * Z) * L
    lambda = eigvals(Symmetric((M + M') / 2))
    inactive = setdiff(1:n, active)
    kappa = maximum(lambda) / minimum(lambda)
    cg = solv_poisson(operator; options)
    return (;
        n, active = length(active), components = length(anchors),
        levels = length(preconditioner.levels),
        asymmetry = maximum(abs, B - B') / maximum(abs, B),
        range = isempty(anchors) ? 0.0 : maximum(abs, C * Baa) / maximum(abs, Baa),
        inactive = isempty(inactive) ? 0.0 :
            maximum(abs, B[inactive, active]) + maximum(abs, B[active, inactive]),
        lambda_min = minimum(lambda), lambda_max = maximum(lambda), kappa,
        bound = (sqrt(kappa) - 1) / (sqrt(kappa) + 1),
        cycles = cg.cycles, factor = cg.factor,
    )
end

function solv_vcycle_cases()
    geometry = GeometryField(solv_uniform(32), solv_band_solid; lipschitz = 1)
    return (;
        uniform16 = solv_vcycle_check(PoissonOperator(solv_uniform(16))),
        uniform32 = solv_vcycle_check(PoissonOperator(solv_uniform(32))),
        amr = solv_vcycle_check(PoissonOperator(solv_amr(16))),
        drop = solv_vcycle_check(PoissonOperator(solv_drop_field(solv_uniform(32)); density = SOLV_DENSITY)),
        split = solv_vcycle_check(PoissonOperator(geometry); options = (; geometry)),
    )
end

# Null spaces: the indicator of every unanchored component is a null vector;
# open (Outflow) faces anchor exactly the components they touch.
function solv_component_check()
    mesh = solv_uniform(32)
    results = map((nothing, (false, false, false, true), (false, true, false, false))) do sides
        geometry = GeometryField(mesh, solv_band_solid; lipschitz = 1)
        boundary = isnothing(sides) ? nothing :
            VelocityBoundaryConditions(
                left = NoSlip(), right = sides[2] ? Outflow() : NoSlip(),
                bottom = NoSlip(), top = sides[4] ? Outflow() : NoSlip(),
            )
        operator = PoissonOperator(geometry; velocity_boundary = boundary)
        labels = AMR._elliptic_component_labels(operator)
        roots = sort!(unique(filter(!iszero, labels)))
        null = maximum(
            maximum(abs, operator.matrix * Float64.(labels .== root); init = 0.0)
                for root in operator.components.anchors;
            init = 0.0,
        )
        (;
            components = length(roots), anchored = length(roots) - length(operator.components.anchors),
            singular = operator.singular, null,
        )
    end
    return (; closed = results[1], top = results[2], right = results[3])
end

# Hierarchy reuse after a change of the refined band, and the in-place
# refresh after a change of the two-phase coefficient on a fixed mesh.
function solv_reuse_check()
    old = TreeGeometricMultigridPreconditioner(PoissonOperator(solv_amr(32)))
    operator = PoissonOperator(solv_amr(32; radius = SOLV_DROP.radius + 0.06))
    reused = TreeGeometricMultigridPreconditioner(operator; reuse = old)
    fresh = TreeGeometricMultigridPreconditioner(operator)
    rhs = [solv_noise(index) for index in eachindex(operator.volumes)]
    a, b = similar(rhs), similar(rhs)
    ldiv!(a, reused, rhs)
    ldiv!(b, fresh, rhs)
    reuse = (;
        levels = length(reused.levels), reused = reused.reused_levels,
        difference = maximum(abs, a - b) / maximum(abs, b),
    )

    mesh = solv_uniform(64)
    before = PoissonOperator(solv_drop_field(mesh); density = SOLV_DENSITY)
    preconditioner = TreeGeometricMultigridPreconditioner(before)
    factorization = preconditioner.coarse_factorization
    moved = InterfaceField(mesh, (x, y) -> solv_drop(x - 0.05, y + 0.03); lipschitz = 1)
    after = PoissonOperator(moved; density = SOLV_DENSITY, template = before)
    AMR._refresh_geometric_multigrid!(preconditioner, after)
    rebuilt = TreeGeometricMultigridPreconditioner(after)
    rhs = [solv_noise(index) for index in eachindex(after.volumes)]
    a, b = similar(rhs), similar(rhs)
    ldiv!(a, preconditioner, rhs)
    ldiv!(b, rebuilt, rhs)
    refresh = (;
        same_factorization = preconditioner.coarse_factorization === factorization,
        difference = maximum(abs, a - b) / maximum(abs, b),
    )
    return (; reuse, refresh)
end

# --- Shifted (viscous) systems ------------------------------------------------------

# Single-phase shifted system V (I - s L) u = V b with s = sigma h^2 (h the
# smallest cell width), noise right-hand side, both components.
function solv_staggered(mesh, sigma; preconditioner = :tree_geometric)
    h = minimum(minimum(block.spacing) for block in mesh.blocks)
    diffusion = StaggeredDiffusionOperator(mesh; viscosity = 1.0, boundary = FreeSlip())
    operator = StaggeredEllipticOperator(diffusion; diffusion = sigma * h^2)
    solver = StaggeredEllipticSolver(operator; preconditioner)
    rhs = solv_noise_faces(mesh)
    solution = FaceVelocity(mesh)
    stats = solve!(solution, solver, rhs; tolerance = SOLV_RTOL, max_cycles = SOLV_MAX_CYCLES)
    @assert stats.converged
    cycles = max(stats.x.cycles, stats.y.cycles)
    result = (;
        unknowns = length(mesh.faces), cycles, method = stats.x.method,
        factor = max(solv_factor(stats.x, stats.x.cycles), solv_factor(stats.y, stats.y.cycles)),
        levels = isnothing(preconditioner) ? 0 : length(solver.x.preconditioner.levels),
    )
    fill!(solution.interior, 0.0)
    bytes = @allocated solve!(solution, solver, rhs; tolerance = SOLV_RTOL, max_cycles = SOLV_MAX_CYCLES)
    return (; result..., bytes)
end

# Two phases, one velocity: (M + dt/2 E'WE) u = M b on the drop fixture.
function solv_variable(mesh, dt; preconditioner = :tree_geometric)
    interface = solv_drop_field(mesh)
    operator = StaggeredDiffusionOperator(interface; viscosity = SOLV_VISCOSITY, boundary = FreeSlip())
    density = mixture_density(interface, SOLV_DENSITY)
    solver = AMR.VariableStaggeredDiffusionSolver(operator, density, dt; preconditioner)
    source = solv_noise_faces(mesh)
    zero_old = FaceVelocity(mesh)
    destination = FaceVelocity(mesh)
    implicit_midpoint_diffusion!(
        destination, source, operator, dt, solver;
        old_velocity = zero_old, density, tolerance = SOLV_RTOL, max_cycles = SOLV_MAX_CYCLES,
    )
    stats = solver.stats
    @assert stats.converged
    inertia = density .* operator.volumes
    stiffness = diag(operator.stiffness)
    ratio = maximum(dt / 2 * stiffness[i] / inertia[i] for i in eachindex(inertia) if inertia[i] > 0)
    result = (;
        unknowns = length(mesh.faces), cycles = stats.cycles, method = stats.method,
        factor = solv_factor(stats, stats.cycles), ratio,
        levels = isnothing(preconditioner) ? 0 : length(solver.preconditioner.levels),
    )
    fill!(destination.interior, 0.0)
    bytes = @allocated implicit_midpoint_diffusion!(
        destination, source, operator, dt, solver;
        old_velocity = zero_old, density, tolerance = SOLV_RTOL, max_cycles = SOLV_MAX_CYCLES,
    )
    return (; result..., bytes)
end

function solv_viscous_cases()
    uniform = solv_uniform(128)
    amr = solv_amr(64)
    cases = (
        (; name = "shift1", run = p -> solv_staggered(uniform, 1.0; preconditioner = p)),
        (; name = "shift100", run = p -> solv_staggered(uniform, 100.0; preconditioner = p)),
        (; name = "shiftamr", run = p -> solv_staggered(amr, 100.0; preconditioner = p)),
        (; name = "variable", run = p -> solv_variable(solv_uniform(64), 1.0; preconditioner = p)),
        (; name = "variableamr", run = p -> solv_variable(amr, 1.0; preconditioner = p)),
    )
    return map(case -> (; case.name, with = case.run(:tree_geometric), without = case.run(nothing)), cases)
end

# --- CMDGFM projection at density ratio 1000 -----------------------------------------

solv_common(x, y, axis) = axis == 1 ?
    0.4 + sinpi(2x) * cospi(y) : -0.2 + cospi(x) * sinpi(2y)
solv_jump(x, y, axis) = axis == 1 ? 0.37 + 0.1sinpi(2y) : -0.21 + 0.1cospi(2x)

function solv_two_predictor(field)
    mesh = field.mesh
    liquid = FaceVelocity(mesh, (x, y, t, axis) -> solv_common(x, y, axis))
    gas = FaceVelocity(mesh, (x, y, t, axis) -> solv_common(x, y, axis) + solv_jump(x, y, axis))
    velocity = (liquid, gas)
    enforce_no_permeation!(velocity)
    AMR._mark_inactive_phase_velocities!(velocity, field)
    return velocity
end

# One projection of the predictor at presstol = 1e-10 from a zero guess, then
# the same projection repeated (allocations of a repeated call). Without a
# preconditioner BiCGSTAB needs about 10^4 iterations: a larger budget. The
# reduction factor uses the code's dimensionless residual defect
# e(r) = dt^2 max(max_c |r_c|/|c|, max_f |r_f|/h_f) of the right-hand side
# (the residual of the zero guess) and of the final true residual.
const SOLV_PLAIN_BICGSTAB = 100_000

function solv_cmdgfm(mesh; preconditioner = :block)
    field = solv_drop_field(mesh)
    solver = CMDGFMProjectionSolver(field; density = SOLV_DENSITY, preconditioner)
    work = CMDGFMProjectionWorkspace(solver)
    budget = isnothing(preconditioner) ? SOLV_PLAIN_BICGSTAB : SOLV_MAX_CYCLES
    stats = project!(
        solv_two_predictor(field), CellField(mesh), solver, SOLV_DT, work;
        tolerance = SOLV_RTOL, max_iterations = budget, warm_start = false,
    )
    # Copies of the right-hand side and of the final true residual; the
    # cell volumes and face widths of the defect come from a re-assembly of
    # the same predictor after the measured projection.
    rhs, residual = copy(work.right_hand_side), copy(work.correction_rhs)
    assembled = AMR._assemble_cmdgfm_system!(solver, work, SOLV_DT, solv_two_predictor(field))
    @assert assembled.right_hand_side == rhs
    initial = maximum(AMR._cmdgfm_residual_defects(rhs, assembled, SOLV_DT))
    final = maximum(AMR._cmdgfm_residual_defects(residual, assembled, SOLV_DT))
    factor = (final / initial)^(1 / max(stats.iterations, 1))
    velocity = solv_two_predictor(field)
    pressure = CellField(mesh)
    bytes = @allocated project!(
        velocity, pressure, solver, SOLV_DT, work;
        tolerance = SOLV_RTOL, max_iterations = budget, warm_start = false,
    )
    return (;
        unknowns = stats.pressure_unknowns + stats.jump_unknowns,
        jumps = stats.jump_unknowns, cycles = stats.iterations, stats.converged,
        initial, final, factor,
        defect = max(stats.mixture_volume_defect, stats.normal_velocity_defect),
        bytes, preconditioner = stats.preconditioner,
    )
end

function solv_cmdgfm_cases()
    return map(((:uniform, solv_uniform(64)), (:amr, solv_amr(32)))) do (name, mesh)
        (;
            name, block = solv_cmdgfm(mesh), ilu = solv_cmdgfm(mesh; preconditioner = :ilu),
            none = solv_cmdgfm(mesh; preconditioner = nothing),
        )
    end
end

# --- Outputs -------------------------------------------------------------------------

solv_rho(x) = x < 0.95 ? @sprintf("%.2f", x) : x < 0.995 ? @sprintf("%.3f", x) : @sprintf("%.4f", x)

function solvers_measurements()
    return (;
        pressure = solv_pressure_cases(), vcycle = solv_vcycle_cases(),
        components = solv_component_check(), reuse = solv_reuse_check(),
        viscous = solv_viscous_cases(), cmdgfm = solv_cmdgfm_cases(),
    )
end

solvers_outputs(write_output) = solvers_report(solvers_measurements())

function solvers_report(measured)
    (; pressure, vcycle, components, reuse, viscous, cmdgfm) = measured
    for case in pressure
        println("solvers pressure $(case.name): ", case.with, " | none ", case.without)
    end
    println("solvers vcycle:              ", vcycle)
    println("solvers components:          ", components)
    println("solvers reuse:               ", reuse)
    for case in viscous
        println("solvers viscous $(case.name): ", case.with, " | none ", case.without)
    end
    for case in cmdgfm
        println("solvers cmdgfm $(case.name): ", case)
    end

    # Mesh-independent multigrid counts, growing unpreconditioned counts, and
    # allocation-free repeated solves (Sections "multigrid" and "measured").
    uniform = [case for case in pressure if startswith(case.name, "uniform")]
    @assert maximum(case.with.cycles for case in uniform) <= minimum(case.with.cycles for case in uniform) + 3
    @assert issorted([case.without.cycles for case in uniform])
    @assert all(case -> case.with.bytes == 0 && case.without.bytes == 0, pressure)
    @assert all(case -> case.with.repeat_cycles == case.with.cycles, pressure)
    @assert only(case for case in pressure if case.name == "split128").with.components == 2
    for result in values(vcycle)
        @assert result.asymmetry <= 1.0e-12 && result.range <= 1.0e-12 && result.inactive == 0
        @assert result.lambda_min > 0
    end
    @assert components.closed.components == 2 && components.closed.anchored == 0 && components.closed.singular
    @assert components.top.anchored == 1 && components.top.singular
    @assert components.right.anchored == 2 && !components.right.singular
    @assert max(components.closed.null, components.top.null) <= 1.0e-12
    @assert reuse.reuse.reused > 0 && reuse.reuse.difference <= 1.0e-12
    @assert reuse.refresh.same_factorization && reuse.refresh.difference <= 1.0e-10
    # The repeated staggered solve allocates a small amount independent of the
    # mesh (iteration over the two-component tuple in solve!); the coupled
    # two-phase solve refreshes its hierarchy on every call.
    staggered_bytes = unique(case.with.bytes for case in viscous[1:3])
    @assert length(staggered_bytes) == 1 && 0 < only(staggered_bytes) <= 1024
    @assert all(case -> case.without.bytes == 0 && case.with.bytes > 0, viscous[4:5])
    @assert all(case -> case.block.converged, cmdgfm)
    # The converged runs meet the residual defects, so the independently
    # evaluated final defect is at most the tolerance.
    @assert all(case -> case.block.final <= SOLV_RTOL, cmdgfm)
    @assert all(case -> !case.none.converged || case.none.cycles >= 100case.block.cycles, cmdgfm)

    byname(cases, name) = only(case for case in cases if case.name == name)
    rows = Any[]
    for (key, case) in (
            ("UniformA", "uniform64"), ("UniformB", "uniform128"), ("UniformC", "uniform256"),
            ("AmrA", "amr32"), ("AmrB", "amr64"), ("Drop", "drop128"), ("DropAmr", "dropamr"),
            ("Split", "split128"),
        )
        result = byname(pressure, case)
        append!(
            rows, [
                ("chkSolv$(key)Unknowns", result.with.unknowns),
                ("chkSolv$(key)Levels", result.with.levels),
                ("chkSolv$(key)Cycles", result.with.cycles),
                ("chkSolv$(key)Factor", solv_rho(result.with.factor)),
                ("chkSolv$(key)Plain", result.without.cycles),
                ("chkSolv$(key)PlainFactor", solv_rho(result.without.factor)),
            ]
        )
    end
    for (key, case) in (
            ("ShiftA", "shift1"), ("ShiftB", "shift100"), ("ShiftAmr", "shiftamr"),
            ("Variable", "variable"), ("VariableAmr", "variableamr"),
        )
        result = byname(viscous, case)
        append!(
            rows, [
                ("chkSolv$(key)Unknowns", result.with.unknowns),
                ("chkSolv$(key)Levels", result.with.levels),
                ("chkSolv$(key)Cycles", result.with.cycles),
                ("chkSolv$(key)Factor", solv_rho(result.with.factor)),
                ("chkSolv$(key)Plain", result.without.cycles),
                ("chkSolv$(key)PlainFactor", solv_rho(result.without.factor)),
                ("chkSolv$(key)Method", String(result.with.method)),
            ]
        )
    end
    variable = byname(viscous, "variable")
    # Iterations of a projection that missed its tolerance carry a dagger.
    counted(r) = r.converged ? string(r.cycles) : "$(r.cycles)\\textsuperscript{\\dag}"
    for case in cmdgfm
        key = case.name == :uniform ? "CmdUniform" : "CmdAmr"
        append!(
            rows, [
                ("chkSolv$(key)Unknowns", case.block.unknowns),
                ("chkSolv$(key)Jumps", case.block.jumps),
                ("chkSolv$(key)Cycles", counted(case.block)),
                ("chkSolv$(key)Ilu", counted(case.ilu)),
                ("chkSolv$(key)IluDefect", sci(case.ilu.defect)),
                ("chkSolv$(key)Plain", counted(case.none)),
                ("chkSolv$(key)Factor", solv_rho(case.block.factor)),
                ("chkSolv$(key)IluFactor", solv_rho(case.ilu.factor)),
                ("chkSolv$(key)PlainFactor", solv_rho(case.none.factor)),
                ("chkSolv$(key)Initial", sci(case.block.initial)),
                ("chkSolv$(key)Bytes", @sprintf("%.0f", case.block.bytes / 1024)),
            ]
        )
    end
    weights(name) = byname(pressure, name).with.weights
    append!(
        rows, [
            ("chkSolvOmegaUniform", @sprintf("%.3f", minimum(weights("uniform128")))),
            ("chkSolvOmegaUniformMax", @sprintf("%.3f", maximum(weights("uniform128")))),
            ("chkSolvOmegaAmr", @sprintf("%.3f", minimum(weights("amr64")))),
            ("chkSolvOmegaDrop", @sprintf("%.3f", minimum(weights("drop128")))),
        ]
    )
    v = vcycle
    append!(
        rows, [
            ("chkSolvVariableRatio", @sprintf("%.0f", variable.with.ratio)),
            ("chkSolvVariableAmrRatio", @sprintf("%.0f", byname(viscous, "variableamr").with.ratio)),
            ("chkSolvKappaUniformA", @sprintf("%.2f", v.uniform16.kappa)),
            ("chkSolvKappaUniformB", @sprintf("%.2f", v.uniform32.kappa)),
            ("chkSolvKappaAmr", @sprintf("%.2f", v.amr.kappa)),
            ("chkSolvKappaDrop", @sprintf("%.2f", v.drop.kappa)),
            ("chkSolvKappaSplit", @sprintf("%.2f", v.split.kappa)),
            ("chkSolvBoundUniform", solv_rho(v.uniform32.bound)),
            ("chkSolvBoundDrop", solv_rho(v.drop.bound)),
            ("chkSolvVcycleCells", v.amr.n),
            ("chkSolvVcycleAsym", sci(maximum(r.asymmetry for r in values(v)))),
            ("chkSolvVcycleRange", sci(maximum(r.range for r in values(v)))),
            ("chkSolvLambdaMin", @sprintf("%.3f", minimum(r.lambda_min for r in values(v)))),
            ("chkSolvLambdaMax", @sprintf("%.3f", maximum(r.lambda_max for r in values(v)))),
            ("chkSolvNull", sci(max(components.closed.null, components.top.null))),
            ("chkSolvReuseLevels", reuse.reuse.levels),
            ("chkSolvReused", reuse.reuse.reused),
            ("chkSolvReuseDiff", sci(reuse.reuse.difference)),
            ("chkSolvRefreshDiff", sci(reuse.refresh.difference)),
            ("chkSolvCgUniform", v.uniform32.cycles),
            ("chkSolvCgUniformFactor", solv_rho(v.uniform32.factor)),
            ("chkSolvCgDrop", v.drop.cycles),
            ("chkSolvCgDropFactor", solv_rho(v.drop.factor)),
            ("chkSolvStaggeredBytes", only(staggered_bytes)),
            ("chkSolvVariableBytes", @sprintf("%.0f", maximum(case.with.bytes for case in viscous[4:5]) / 1024)),
            ("chkSolvCmdIluMiB", @sprintf("%.0f", maximum(case.ilu.bytes for case in cmdgfm) / 2^20)),
        ]
    )
    return rows
end

# `generate.jl solvers`: rewrite only the Chapter 11 macros of checks.tex, at
# the place where the complete run writes them (after the Chapter 10 macros),
# and keep every other line.
function solvers_checks_only(write_output)
    macros = solvers_outputs(write_output)
    names = Set(String(name) for (name, _) in macros)
    lines = filter(readlines(joinpath(OUTPUT, "checks.tex"))[2:end]) do line
        found = match(r"^\\newcommand\\(\w+)\{", line)
        isnothing(found) || !(found.captures[1] in names)
    end
    anchor = findfirst(line -> startswith(line, "\\newcommand\\chkTabMimIterations{"), lines)
    @assert !isnothing(anchor)
    own = [rstrip(macro_line(name, value)) for (name, value) in macros]
    lines = [lines[1:anchor]; own; lines[(anchor + 1):end]]
    return write_output("checks.tex", join(lines, "\n") * "\n")
end
