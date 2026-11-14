using Test, LinearAlgebra, SparseArrays, TwoPhaseAMR
include("../helpers.jl")
include("mimetic_lab_helpers.jl")

function _mimetic_case(
        n, levelset; density = (1.0, 1.0), periodic = (true, false), lipschitz = 1.5,
    )
    mesh = BlockTreeMesh(; block_size = (n, n), periodic)
    interface = InterfaceField(mesh, levelset; lipschitz)
    solver = MimeticProjectionSolver(interface; density)
    workspace = MimeticProjectionWorkspace(solver)
    return (; mesh, interface, solver, workspace)
end

function _mimetic_velocity(case, value)
    velocity = (FaceVelocity(case.mesh), FaceVelocity(case.mesh))
    for phase in 1:2, (index, face) in pairs(case.mesh.faces)
        velocity[phase].interior[index] = value(phase, face)
    end
    TwoPhaseAMR._mark_inactive_phase_velocities!(velocity, case.interface)
    return velocity
end

_mimetic_active(velocity) = [filter(isfinite, phase.interior) for phase in velocity]
_mimetic_maximum(velocity) = maximum(maximum(abs, values; init = 0.0) for values in _mimetic_active(velocity))

function _mimetic_jump(velocity, case)
    jump = 0.0
    for index in eachindex(case.mesh.faces)
        liquid, gas = velocity[1].interior[index], velocity[2].interior[index]
        isfinite(liquid) && isfinite(gas) && (jump = max(jump, abs(gas - liquid)))
    end
    return jump
end

_mimetic_circle(n; x = 0.5, y = 0.5, radius = 0.25) =
    (xx, yy) -> hypot(xx - x - 0.21 / n, yy - y - 0.37 / n) - radius

@testset "Mimetic projection option and validation" begin
    projection = MimeticProjection()
    @test projection isa AbstractTwoVelocityProjection
    # The projection implies its metric and force maps: it takes no options.
    @test_throws MethodError MimeticProjection(metric = :mfd)
    @test_throws MethodError MimeticProjectionSolver(
        InterfaceField(BlockTreeMesh(block_size = (8, 8)), (x, y) -> y - 0.53; lipschitz = 1);
        metric = :mfd,
    )
    @test TwoVelocityModel(pressure_projection = projection).pressure_projection === projection
    @test TwoVelocityModel().pressure_projection isa CMDGFMProjection

    circle(x, y) = hypot(x - 0.5, y - 0.5) - 0.2
    physics(interface) = TwoPhase(interface; density = (2.0, 1.0), dynamic_viscosity = (0.0, 0.0))
    model = TwoVelocityModel(pressure_projection = projection)
    # Adaptive meshes with hanging faces are rejected at construction.
    graded = BlockTreeMesh(
        base_mesh = (16, 16), block_size = (4, 4), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i >= 2,
    )
    @test length(unique(block.key.level for block in graded.blocks)) == 2
    graded_interface = InterfaceField(graded, circle; lipschitz = 1)
    @test_throws ArgumentError MimeticProjectionSolver(graded_interface)
    @test_throws ArgumentError FlowSimulation(
        graded, physics(graded_interface); velocity_model = model, boundary = FreeSlip(),
    )
    # Embedded solids are rejected, through the interface or the system.
    mesh = BlockTreeMesh(block_size = (16, 16), periodic = (false, false))
    geometry = GeometryField(mesh, (x, y) -> y - 0.1; lipschitz = 1)
    wet = InterfaceField(mesh, circle; geometry, lipschitz = 1)
    @test_throws ArgumentError MimeticProjectionSolver(wet)
    @test_throws ArgumentError FlowSimulation(
        mesh, physics(wet); geometry, velocity_model = model, boundary = FreeSlip(),
    )
    # Square cells and Float64 only.
    stretched = BlockTreeMesh(block_size = (16, 16), extent = (2.0, 1.0))
    @test_throws ArgumentError MimeticProjectionSolver(
        InterfaceField(stretched, circle; lipschitz = 1),
    )
    single = BlockTreeMesh(block_size = (16, 16), origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0))
    @test_throws ArgumentError MimeticProjectionSolver(
        InterfaceField(single, (x, y) -> hypot(x - 0.5f0, y - 0.5f0) - 0.2f0; lipschitz = 1),
    )
    uniform = BlockTreeMesh(block_size = (16, 16), periodic = (false, false))
    flow = FlowSimulation(
        uniform, physics(InterfaceField(uniform, circle; lipschitz = 1));
        velocity_model = model, boundary = FreeSlip(),
    )
    @test flow.system.two_phase.pressure_projection === projection
    # `simulate!` accounts the mimetic solves and reports them in the summary.
    dt = 1.0e-3  # the flow starts at rest without surface tension: no CFL limit
    result = simulate!(flow, 2dt; dt)
    @test result.projection.solves == result.steps == 2
    @test result.projection.converged
    report = IOBuffer()
    @test isnothing(print_simulation_summary(report, flow, result))
    summary = String(take!(report))
    @test occursin("iterations this run: $(result.projection.iterations)", summary)
    @test occursin("energy identity defect", summary)
    # The run accumulates the defective cells and lumped values of every solve
    # (total and largest per solve) and the summary prints both.
    @test (result.projection.defective_cells, result.projection.max_defective_cells) == (0, 0)
    @test (result.projection.lumped_values, result.projection.max_lumped_values) == (0, 0)
    @test occursin("defective cells this run (total, max per solve): (0, 0)", summary)
    @test occursin("lumped values this run (total, max per solve): (0, 0)", summary)
    case = _mimetic_case(16, _mimetic_circle(16))
    stats = project!(
        _mimetic_velocity(case, (phase, face) -> 0.0), CellField(case.mesh),
        case.solver, 1.0, case.workspace,
    )
    counted(defective, lumped) = MimeticProjectionStats(
        (
            name === :defective_cells ? defective : name === :lumped_values ? lumped :
                getfield(stats, name) for name in fieldnames(MimeticProjectionStats)
        )...,
    )
    run = ProjectionRunStats(Float64)
    for (defective, lumped) in ((3, 0), (1, 2), (0, 1))
        TwoPhaseAMR._accumulate_projection!(run, counted(defective, lumped))
    end
    @test (run.defective_cells, run.max_defective_cells) == (4, 3)
    @test (run.lumped_values, run.max_lumped_values) == (3, 2)
end

@testset "Mimetic metric: symmetry, definiteness, diagonal masses and consistency" begin
    for density in ((1.0, 1.0), (1000.0, 1.0))
        case = _mimetic_case(32, _mimetic_circle(32); density)
        assembly = TwoPhaseAMR._mimetic_assemble!(case.solver, case.workspace)
        band = assembly.band_matrix
        @test maximum(abs, band - band') <= 1.0e-14 * maximum(abs, band)
        @test isposdef(Symmetric(Matrix(band)))
        @test assembly.cut_cells > 0 && assembly.defective_cells == 0
        # Away from the band the metric is the diagonal phase mass of the code.
        off = findall(iszero, assembly.band_position)
        @test !isempty(off)
        @test assembly.diagonal[off] == assembly.dof_mass[off]
        # Consistency (note on kinetic metrics, prop. assembly): for equal
        # densities the uniform acceleration w = e_y (∇p = rho e_y) satisfies
        # 𝓜 w = -Dᵀ p with p = rho y at the pressure points of the cut cells (the
        # chord midpoints of these simple cells), the centres elsewhere.
        density[1] == density[2] || continue
        work = case.workspace
        topology = work.topology
        w = [case.mesh.faces[assembly.dof_face[r]].axis == 2 ? 1.0 : 0.0 for r in eachindex(assembly.dof_face)]
        mw = TwoPhaseAMR._mimetic_metric_product!(zeros(0), work, w)
        p = density[1] .* assembly.pressure_y
        residual = map(eachindex(w)) do r
            face = assembly.dof_face[r]
            # -(Dᵀ p)_r = |f| a (p_plus - p_minus)
            gradient = case.mesh.faces[face].area * assembly.dof_aperture[r] *
                (p[topology.face_cells[2, face]] - p[topology.face_cells[1, face]])
            mw[r] - gradient
        end
        @test maximum(abs, residual) <= 1.0e-12 * maximum(abs, mw)
    end
end

@testset "Mimetic projection: idempotence, energy identity and momentum" begin
    case = _mimetic_case(32, _mimetic_circle(32); density = (10.0, 1.0))
    predictor(phase, face) = (phase == 1 ? 1.0 : -0.7) * sin(2pi * face.center[1] + 1.3face.axis) +
        0.4cos(3pi * face.center[2] + phase)
    velocity = _mimetic_velocity(case, predictor)
    pressure = CellField(case.mesh)
    stats = project!(velocity, pressure, case.solver, 1.0e-2, case.workspace; tolerance = 1.0e-13)
    @test stats.converged
    @test stats.mixture_volume_defect <= 1.0e-13
    @test abs(stats.energy_identity_defect) <= 1.0e-12
    scale = maximum(abs, case.workspace.metric_predictor)
    # The first-moment momentum e_iᵀ𝓜u changes by the wall force only: not at
    # all along the periodic x axis (audit prop. metric-momentum, thm. mm(e)).
    @test abs(stats.momentum_change[1]) <= 1.0e-12 * scale * length(case.mesh.faces)
    @test abs(stats.momentum_change[2] - stats.momentum_wall_force[2]) <=
        1.0e-12 * scale * length(case.mesh.faces)
    # The transported momentum is not conserved by this metric; report it.
    @test all(isfinite, stats.transported_momentum_change)
    println(
        "mimetic projection transported momentum change (circle, rho=(10,1), n=32): ",
        stats.transported_momentum_change, " of ", stats.transported_momentum,
    )
    projected = deepcopy(velocity)
    again = project!(velocity, pressure, case.solver, 1.0e-2, case.workspace; tolerance = 1.0e-13)
    @test again.converged
    @test maximum(
        maximum(abs, filter(isfinite, velocity[phase].interior .- projected[phase].interior); init = 0.0)
            for phase in 1:2
    ) <= 1.0e-10 * _mimetic_maximum(projected)
    @test _phase_storage_contract(velocity, case.interface)
end

@testset "Mimetic projection keeps a smooth shear flow (T1)" begin
    shear(y) = 0.5tanh((y - 0.5) / 0.05)
    for n in (32, 64), levelset in (
                (x, y) -> y - 0.5 - 0.37 / n,
                (x, y) -> y - 0.5 - 0.01cos(2pi * x),
                (x, y) -> y - 0.5 - 0.06 * (0.25 - abs(mod(x, 1.0) - 0.5)),
            )
        # Every metric keeps a solenoidal single-valued field (audit prop.
        # metric-free), also in the triangle's kink cells.
        case = _mimetic_case(n, levelset; density = (3.0, 1.0))
        velocity = _mimetic_velocity(case, (phase, face) -> face.axis == 1 ? shear(face.center[2]) : 0.0)
        initial = deepcopy(velocity)
        stats = project!(velocity, CellField(case.mesh), case.solver, 1.0, case.workspace; tolerance = 1.0e-13)
        @test stats.converged
        @test _mimetic_jump(velocity, case) <= 1.0e-10
        @test maximum(
            maximum(abs, filter(isfinite, velocity[phase].interior .- initial[phase].interior); init = 0.0)
                for phase in 1:2
        ) <= 1.0e-10
    end
end

@testset "Mimetic projection under a uniform acceleration (T3)" begin
    # Laboratory geometry (exact apertures and staggered fractions): equal
    # densities are at rest to round-off, contrasts converge at first order
    # with the audit's numbers (tab. mm-tests).
    laboratory = Dict(
        (10.0, 32) => 0.0702, (10.0, 64) => 0.0385,
        (1000.0, 32) => 0.0789, (1000.0, 64) => 0.0435,
    )
    @test first(MimeticLab.t3_error(32, (1.0, 1.0))) <= 1.0e-9
    for ratio in (10.0, 1000.0)
        errors = map((32, 64)) do n
            # Round-off bounds the attainable divergence defect at 1000:1.
            error, out = MimeticLab.t3_error(n, (ratio, 1.0); tolerance = 1.0e-9)
            @test out.result.target_defect <= 1.0e-9
            @test error ≈ laboratory[(ratio, n)] rtol = 5.0e-3
            error
        end
        @test errors[2] < errors[1]
    end
    # Package geometry (PLIC apertures): rest for equal densities.
    for n in (32, 64)
        case = _mimetic_case(n, (x, y) -> y - 0.5 - 0.37 / n - 0.005cos(2pi * x))
        velocity = _mimetic_velocity(case, (phase, face) -> face.axis == 2 ? -1.0 : 0.0)
        stats = project!(velocity, CellField(case.mesh), case.solver, 1.0, case.workspace; tolerance = 1.0e-11)
        @test stats.converged
        @test _mimetic_maximum(velocity) <= 1.0e-9
    end
end

@testset "Mimetic forces balance constant curvature and gravity" begin
    for formulation in (CSF(), CGFM()), density in ((1.0, 1.0), (1000.0, 1.0))
        mesh = BlockTreeMesh(block_size = (32, 32), periodic = (false, false))
        interface = InterfaceField(mesh, _mimetic_circle(32); lipschitz = 1)
        sigma, curvature, dt = 0.7, 4.0, 1.0e-2
        flow = FlowSimulation(
            mesh, TwoPhase(
                interface; density, dynamic_viscosity = (0.0, 0.0),
                surface_tension = SurfaceTension(sigma; method = formulation),
            );
            velocity_model = TwoVelocityModel(pressure_projection = MimeticProjection()),
            boundary = FreeSlip(),
        )
        model = flow.system.two_phase
        state = TwoPhaseAMR._two_velocity_flow_workspace!(flow.workspace.two_phase, model)
        reconstruct!(interface)
        for block in eachindex(mesh.blocks)
            fill!(interior(interface.curvature, block), curvature)
        end
        TwoPhaseAMR._two_velocity_forces_and_pressure!(
            flow.velocity, flow.pressure, flow.system, flow.workspace, state, model, dt,
        )
        force = _mimetic_maximum(flow.velocity) / dt
        @test force > 1.0
        # The force is -𝓜⁻¹Dᵀφ with the cell potential φ = σκH of the jumps.
        work = state.projection_workspace
        assembly = work.assembly
        potential = map(TwoPhaseAMR._cell_references(mesh)) do reference
            c = TwoPhaseAMR._conditional_fraction(interface, reference)
            sigma * curvature * (formulation isa CSF ? c : Float64(c > 0.5))
        end
        fill!(resize!(work.predictor, length(assembly.dof_face)), 0.0)
        resize!(work.velocity, length(assembly.dof_face))
        TwoPhaseAMR._mimetic_recover!(work, -dt .* potential)
        gathered = zeros(0)
        TwoPhaseAMR._mimetic_gather!(gathered, assembly, flow.velocity)
        @test maximum(abs, gathered .- work.velocity) <= 1.0e-12 * dt * force
        stats = TwoPhaseAMR._project_two_velocity!(
            flow.velocity, flow.pressure, flow.system, flow.workspace, state, dt;
            tolerance = 1.0e-12, max_iterations = 2000,
        )
        @test stats.converged
        @test _mimetic_maximum(flow.velocity) <= 1.0e-8
    end
    # Gravity: rest for equal densities on a drop, and for a flat interface at
    # density ratio 1000; the hydrostatic pressure is recovered in both.
    for (levelset, density) in (
            (_mimetic_circle(32), (1.0, 1.0)),
            ((x, y) -> y - 0.5 - 0.37 / 32, (1000.0, 1.0)),
        )
        mesh = BlockTreeMesh(block_size = (32, 32), periodic = (true, false))
        interface = InterfaceField(mesh, levelset; lipschitz = 1)
        flow = FlowSimulation(
            mesh, TwoPhase(
                interface; density, dynamic_viscosity = (0.0, 0.0), gravity = (0.0, -9.81),
            );
            velocity_model = TwoVelocityModel(pressure_projection = MimeticProjection()),
            boundary = FreeSlip(),
        )
        for _ in 1:3
            stats = step!(
                flow.velocity, flow.pressure, flow.system, 1.0e-3, flow.workspace;
                tolerance = 1.0e-12, max_cycles = 2000,
            )
            @test stats.converged
        end
        @test _mimetic_maximum(flow.velocity) <= 1.0e-8
    end
    # Gravity on a heavy drop at density ratio 1000 is the uniform acceleration
    # of both phases: after one step every phase velocity is O(g dt) (the drop
    # starts to fall). A density-weighted jump force gave the gas value of a
    # mixed face the weight of the mixture, about 500 g dt.
    for density in ((1000.0, 1.0), (1.0, 1000.0))
        mesh = BlockTreeMesh(block_size = (32, 32), periodic = (true, false))
        interface = InterfaceField(mesh, _mimetic_circle(32); lipschitz = 1)
        flow = FlowSimulation(
            mesh, TwoPhase(
                interface; density, dynamic_viscosity = (0.0, 0.0), gravity = (0.0, -1.0),
            );
            velocity_model = TwoVelocityModel(pressure_projection = MimeticProjection()),
            boundary = FreeSlip(),
        )
        dt = 1.0e-3
        stats = step!(
            flow.velocity, flow.pressure, flow.system, dt, flow.workspace;
            tolerance = 1.0e-12, max_cycles = 2000,
        )
        @test stats.converged
        @test 0.5 <= _mimetic_maximum(flow.velocity) / dt <= 2.0
    end
end

# The phase polygon of the former construction (candidate MM) from the liquid
# apertures of the four sides of a unit cell (centre at the origin, sides
# bottom, right, top, left) and the liquid wet ends (`starts[k]`: at the
# counter-clockwise start of side `k`). Consecutive phase parts whose ends
# differ are joined by a chord. Returns the edges `(x1, y1, x2, y2, side)` (side
# 0: chord) and the number of chords.
const _MM_CORNERS = ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))
function _mm_polygon(apertures, starts, phase)
    segments = NTuple{5, Any}[]
    for k in 1:4
        b = phase == 1 ? apertures[k] : 1 - apertures[k]
        b == 0 && continue
        start = phase == 1 ? starts[k] : !starts[k]
        (ax, ay), (bx, by) = _MM_CORNERS[k], _MM_CORNERS[mod1(k + 1, 4)]
        push!(
            segments, b == 1 ? (ax, ay, bx, by, k) :
                start ? (ax, ay, ax + b * (bx - ax), ay + b * (by - ay), k) :
                (bx - b * (bx - ax), by - b * (by - ay), bx, by, k),
        )
    end
    edges = NTuple{5, Any}[]
    chords = 0
    for q in eachindex(segments)
        current, next = segments[q], segments[mod1(q + 1, length(segments))]
        push!(edges, current)
        if hypot(current[3] - next[1], current[4] - next[2]) > 1.0e-12
            push!(edges, (current[3], current[4], next[1], next[2], 0))
            chords += 1
        end
    end
    return edges, chords
end

_mm_area(edges) = sum(e -> (e[1] * e[4] - e[3] * e[2]) / 2, edges)

# The MM block of a simple polygon with its chord flux slaved, on the wetted
# sides in increasing side order, and the chord midpoint (former package code,
# techref eqs. mimetic-local and mimetic-slave in their polygon form).
function _mm_block(edges)
    area = _mm_area(edges)
    cx = sum(e -> (e[1] + e[3]) * (e[1] * e[4] - e[3] * e[2]) / 6, edges) / area
    cy = sum(e -> (e[2] + e[4]) * (e[1] * e[4] - e[3] * e[2]) / 6, edges) / area
    lengths = [hypot(e[3] - e[1], e[4] - e[2]) for e in edges]
    flux = hcat([e[4] - e[2] for e in edges], [e[1] - e[3] for e in edges])
    moments = hcat(
        lengths .* [(e[1] + e[3]) / 2 - cx for e in edges],
        lengths .* [(e[2] + e[4]) / 2 - cy for e in edges],
    )
    projector = I - flux * inv(flux' * flux) * flux'
    metric = moments * moments' / area + Diagonal(lengths) * projector * Diagonal(lengths) / 2
    chord = findfirst(e -> e[5] == 0, edges)
    sides = [q for q in eachindex(edges) if edges[q][5] > 0]
    slave = zeros(length(edges), length(sides))
    for (i, q) in pairs(sides)
        slave[q, i] = 1.0
        slave[chord, i] = -lengths[q] / lengths[chord]
    end
    midpoint = ((edges[chord][1] + edges[chord][3]) / 2, (edges[chord][2] + edges[chord][4]) / 2)
    return slave' * metric * slave, area, midpoint
end

# MMd blocks of one unit cell from the package: pressure point and, per phase,
# the block, the Gram tensor G = Σ n_k R_kᵀ and the consistency flag. `point`
# replaces the package's pressure point.
function _mmd_cell(apertures, starts; point = nothing)
    TP = TwoPhaseAMR
    xs, ys, ex, ey = TP._mimetic_pressure_point(apertures, starts)
    isnothing(point) || ((xs, ys) = point)
    phases = map(1:2) do phase
        block, sides, lengths, moments = zeros(4, 4), zeros(Int, 4), zeros(4), zeros(4, 2)
        m, consistent = TP._mimetic_gram_block!(
            block, sides, lengths, moments, apertures, starts, phase, xs, ys, ex, ey,
        )
        normals = [TP._mimetic_normal(Float64, sides[q])[i] for q in 1:m, i in 1:2]
        (; block = block[1:m, 1:m], gram = normals' * moments[1:m, :], normals, moments = moments[1:m, :], consistent)
    end
    return (xs, ys), phases
end

# The note's data point: the point of the symmetry line η × x* = τ closest to
# the mean of the phase boundaries on the partially wetted sides.
function _mmd_data_point(apertures, starts)
    TP = TwoPhaseAMR
    parts = [TP._mimetic_side_parts(Float64, k, apertures[k], starts[k]) for k in 1:4]
    crossed = [first(parts[k]) for k in 1:4 if 0 < apertures[k] < 1]
    xr, yr = sum(first, crossed) / length(crossed), sum(last, crossed) / length(crossed)
    normals = [TP._mimetic_normal(Float64, k) for k in 1:4]
    ex = sum(apertures[k] * normals[k][1] for k in 1:4)
    ey = sum(apertures[k] * normals[k][2] for k in 1:4)
    tau = sum(apertures[k] * (normals[k][1] * parts[k][2][2] - normals[k][2] * parts[k][2][1]) for k in 1:4)
    defect = (tau - (ex * yr - ey * xr)) / (ex^2 + ey^2)
    return xr - defect * ey, yr + defect * ex
end

@testset "Mimetic Gram blocks: the polygon block where the polygon exists, degenerate data" begin
    # Simple cut cells (note on kinetic metrics, prop. gram(d)): the pressure
    # point is the chord midpoint, G = |P| I and the block is that of MM.
    for (apertures, starts) in (
            ((0.3, 0.4, 0.0, 0.0), (false, true, false, false)),  # corner clip
            ((1.0, 0.6, 0.0, 0.3), (false, true, false, false)),  # trapezoid
            ((1.0, 1.0, 0.5, 0.0), (false, false, true, false)),  # chord to a corner
            ((0.02, 1.0, 1.0, 0.97), (false, false, false, true)),  # gas sliver
        )
        point, phases = _mmd_cell(apertures, starts)
        for phase in 1:2
            edges, chords = _mm_polygon(apertures, starts, phase)
            @test chords == 1
            block, area, midpoint = _mm_block(edges)
            local result = phases[phase]
            @test result.consistent
            @test all(isapprox.(point, midpoint; atol = 1.0e-14))
            @test result.gram ≈ area * I atol = 1.0e-14
            @test result.block ≈ block atol = 1.0e-13
            @test result.block * result.normals ≈ result.moments atol = 1.0e-14
        end
        # Prop. gram(b): the Gram tensors of the two phases add up to the cell.
        @test phases[1].gram + phases[2].gram ≈ I atol = 1.0e-14
    end
    # Zero-area phase: the liquid wets one per cent of the bottom only. At the
    # data point (the wetted midpoint) G^l = 0 lies at the end of the
    # admissible interval, where the block would vanish (block SPD needs
    # G ≻ 0: ker m = N ker G). The pressure point keeps the margin 0.01h from
    # that end: G^l = b·0.01 e_y e_yᵀ and the liquid block is the wetted length
    # times the margin, consistent. The gas block is consistent too.
    point, phases = _mmd_cell((0.01, 0.0, 0.0, 0.0), (true, false, false, false))
    @test all(isapprox.(point, (-0.495, -0.49); atol = 1.0e-15))
    @test phases[1].consistent && phases[2].consistent
    @test phases[1].gram ≈ [0.0 0.0; 0.0 1.0e-4] atol = 1.0e-15
    @test phases[1].block ≈ [1.0e-4;;] atol = 1.0e-15
    @test phases[2].gram ≈ I - phases[1].gram atol = 1.0e-15
    # The metric is continuous where the data point crosses an end of the
    # admissible interval (review data: without the margin at both sides of
    # the end the largest block entry jumped from 2.3 to 3.0e3).
    path = map(range(0.374, 0.3752; length = 241)) do a2
        point, phases = _mmd_cell((0.0406, a2, 0.2513, 0.3723), (false, true, false, true))
        @test all(result -> result.consistent, phases)
        maximum(abs, phases[1].block)
    end
    @test maximum(path) < 10
    @test maximum(abs, diff(path)) < 1.0e-3
    # A liquid sliver near a grid line (a Rayleigh–Taylor state at h = λ/16):
    # at the data point G^l has the eigenvalue -3e-5 (the point lies 1e-4 h
    # beyond the top face). Along the symmetry line an admissible interval
    # exists; the pressure point is clamped into it and both blocks are
    # consistent and positive definite.
    sliver = ((0.0, 9.0e-5, 0.39584, 0.00049), (false, false, true, true))
    _, unclamped = _mmd_cell(sliver...; point = _mmd_data_point(sliver...))
    @test !unclamped[1].consistent && minimum(eigvals(Symmetric(unclamped[1].gram))) < -1.0e-5
    point, phases = _mmd_cell(sliver...)
    @test point[2] < 0.5
    for result in phases
        @test result.consistent
        @test result.gram ≈ result.gram' atol = 1.0e-15
        @test result.block * result.normals ≈ result.moments atol = 1.0e-12
        @test isposdef(Symmetric(result.block))
    end
    # A film along a grid line, η = 0 (liquid at the bottom of the left and
    # right sides): G^l = diag(t, 0) for every point, and the kernel condition
    # fixes the height of the pressure point to the film's mid-height (the
    # data point, the film's top, is inconsistent).
    film = ((0.0, 0.2, 0.0, 0.2), (false, true, false, false))
    _, unclamped = _mmd_cell(film...; point = (0.0, -0.3))
    @test !unclamped[1].consistent
    point, phases = _mmd_cell(film...)
    @test all(isapprox.(point, (0.0, -0.4); atol = 1.0e-15))
    @test phases[1].gram ≈ [0.2 0.0; 0.0 0.0] atol = 1.0e-15
    for result in phases
        @test result.consistent
        @test result.block * result.normals ≈ result.moments atol = 1.0e-14
        @test isposdef(Symmetric(result.block))
    end
    # No admissible pressure point on the line (counterexample of the
    # derivation check, two corner pieces): g22(1 - g22) < g12². The cell is
    # defective; its blocks are the least-squares fallback, SPD.
    corners = ((0.5, 0.0, 0.3, 0.1), (true, false, true, false))
    _, phases = _mmd_cell(corners...)
    @test !all(result -> result.consistent, phases)
    for result in phases
        @test result.block ≈ result.block' atol = 1.0e-15
        @test isposdef(Symmetric(result.block))
    end
    # η = 0 and τ ≠ 0: the liquid wets the bottom of the right side and the top
    # of the left side with equal apertures (two chords). No point makes the
    # Gram tensors symmetric; the cell is defective with SPD fallback blocks.
    point, phases = _mmd_cell((0.0, 0.3, 0.0, 0.3), (false, true, false, true))
    @test _mm_polygon((0.0, 0.3, 0.0, 0.3), (false, true, false, true), 1)[2] == 2
    for result in phases
        @test !result.consistent
        @test abs(result.gram[1, 2] - result.gram[2, 1]) > 0.1
        @test result.block ≈ result.block' atol = 1.0e-15
        @test isposdef(Symmetric(result.block))
    end
    # Assembled on synthetic face data of a gas-filled mesh. (i) Only the left
    # and right faces of one cell are wetted: that cell is defective and its
    # two neighbours hold zero-area liquid. (ii) One horizontal face is wetted
    # over 99 % or 1 %: both of its cells give the liquid value a block of the
    # wetted length times the margin of the pressure point. (iii) A wetted
    # length of 1e-12 h: those blocks are below 1e-12 ρh², the value gets its
    # lumped mass instead, counted in `lumped_values`. The assembled metric is
    # positive definite in all of them.
    n = 8
    case = _mimetic_case(n, (x, y) -> 1.0)
    mesh = case.mesh
    TP = TwoPhaseAMR
    topology = TP._mimetic_topology(mesh)
    cell = TP._cell_index(CellRef(1, 4, 4), n, n)
    faces = length(mesh.faces)
    boundaries = length(mesh.boundary_faces)
    for (wetted, expected) in (
            (((topology.sides[2, cell], true, 0.3), (topology.sides[4, cell], false, 0.3)), (3, 1, 0)),
            (((topology.sides[3, cell], true, 0.99),), (2, 0, 0)),
            (((topology.sides[1, cell], false, 0.99),), (2, 0, 0)),
            (((topology.sides[3, cell], true, 0.01),), (2, 0, 0)),
            (((topology.sides[3, cell], true, 1.0e-12),), (2, 0, 1)),
        )
        fraction = zeros(faces)
        aperture = zeros(faces)
        flags = zeros(Int8, faces)
        for (face, low, a) in wetted
            fraction[face] = a / 3
            aperture[face] = a
            flags[face] = low ? TP._MIMETIC_LIQUID_LOW : TP._MIMETIC_LIQUID_HIGH
        end
        data = (;
            liquid_fraction = fraction, liquid_aperture = aperture, gas_aperture = 1 .- aperture,
            liquid_mass = fraction, gas_mass = 1 .- fraction,
        )
        assembly = TP._mimetic_assemble_from_data!(
            case.workspace, mesh, data, flags, zeros(boundaries),
            falses(boundaries), (1.0, 1.0),
        )
        @test (assembly.cut_cells, assembly.defective_cells, assembly.lumped_values) == expected
        @test isposdef(Symmetric(Matrix(assembly.band_matrix)))
    end
end

@testset "Mimetic projection rejects cut cells at walls" begin
    # A drop on a wall has cut cells at the domain boundary.
    wall = _mimetic_case(32, _mimetic_circle(32; y = 0.1))
    velocity = _mimetic_velocity(wall, (phase, face) -> 0.0)
    @test_throws ArgumentError project!(velocity, CellField(wall.mesh), wall.solver, 1.0, wall.workspace)
end

# Liquid apertures and wet ends of the four sides of `cell` in the assembly.
function _mimetic_cell_data(case, cell)
    TP = TwoPhaseAMR
    work = case.workspace
    topology = work.topology
    apertures = ntuple(4) do k
        face = topology.sides[k, cell]
        face > 0 ? work.face_data.liquid_aperture[face] : work.boundary_aperture[-face]
    end
    starts = ntuple(4) do k
        face = topology.sides[k, cell]
        0 < apertures[k] < 1 && face > 0 || return false
        low = work.assembly.flags[face] == TP._MIMETIC_LIQUID_LOW
        return TP._MIMETIC_STARTS_LOW[k] ? low : !low
    end
    return apertures, starts
end

@testset "Mimetic metric on averaged apertures that close no region" begin
    # The face apertures are the mean of the two adjacent PLIC traces. Near a
    # grid line (θ h above it) and at corner clips next to a pure cell they
    # give cut cells whose phase parts would need two chords or bound no area:
    # no polygon exists, the Gram tensor does.
    cases = (
        (32, (x, y) -> y - 0.5 - 0.01 / 32 - 0.005cos(2pi * x)),
        (32, (x, y) -> y - 0.5 - 0.03 / 32 - 0.005cos(2pi * x)),
        (32, (x, y) -> y - 0.5 - 0.97 / 32 - 0.005cos(2pi * x)),
        (64, _mimetic_circle(64)),
        (128, (x, y) -> y - 0.5 - 0.37 / 128 - 0.05cos(2pi * x)),
        (32, (x, y) -> y - 0.5 - 0.01cos(2pi * x)),
    )
    nonsimple = 0
    for (n, levelset) in cases, density in ((1.0, 1.0), (1000.0, 1.0))
        case = _mimetic_case(n, levelset; density)
        velocity = _mimetic_velocity(case, (phase, face) -> face.axis == 2 ? -1.0 : 0.0)
        # Round-off bounds the attainable divergence defect at 1000:1
        # (2.8e-9 on the n = 128 wave).
        stats = project!(
            velocity, CellField(case.mesh), case.solver, 1.0, case.workspace;
            tolerance = density[1] == density[2] ? 1.0e-11 : 1.0e-8,
        )
        @test stats.converged
        @test stats.defective_cells == 0 && stats.lumped_values == 0
        @test stats.gridline_faces == 0
        work = case.workspace
        assembly = work.assembly
        band = assembly.band_matrix
        @test maximum(abs, band - band') <= 1.0e-14 * maximum(abs, band)
        @test isposdef(Symmetric(Matrix(band)))
        # Rest under a uniform force with equal densities (prop. meaning(a)).
        density[1] == density[2] && @test _mimetic_maximum(velocity) <= 1.0e-11
        # Consistency per phase and axis (prop. assembly): 𝓜 χ^π_k = R^π_k for
        # the field χ^π_k that is one on the values of phase π on the faces of
        # axis k and zero elsewhere, R^π_k = ρ^π a |f| (x*₊ - x*₋)_k on every
        # value of phase π (periodic differences).
        topology = work.topology
        point = (assembly.pressure_x, assembly.pressure_y)
        dofs = eachindex(assembly.dof_face)
        axis(r) = case.mesh.faces[assembly.dof_face[r]].axis
        for phase in 1:2, k in 1:2
            field = [assembly.dof_phase[r] == phase && axis(r) == k ? 1.0 : 0.0 for r in dofs]
            image = TwoPhaseAMR._mimetic_metric_product!(zeros(0), work, field)
            target = map(dofs) do r
                assembly.dof_phase[r] == phase || return 0.0
                face = assembly.dof_face[r]
                difference = point[k][topology.face_cells[2, face]] - point[k][topology.face_cells[1, face]]
                case.mesh.periodic[k] && (difference -= round(difference))
                density[phase] * case.mesh.faces[face].area * assembly.dof_aperture[r] * difference
            end
            @test maximum(abs, image .- target) <= 1.0e-12 * maximum(abs, image)
        end
        # Count the cut cells whose apertures form no simple polygon.
        density[1] == density[2] || continue
        for cell in axes(topology.sides, 2)
            apertures, starts = _mimetic_cell_data(case, cell)
            all(==(apertures[1]), apertures) && apertures[1] in (0.0, 1.0) && continue
            nonsimple += any(1:2) do phase
                edges, chords = _mm_polygon(apertures, starts, phase)
                !isempty(edges) && (chords != 1 || abs(_mm_area(edges)) < 1.0e-12)
            end
        end
    end
    println("mimetic metric: cut cells without a simple aperture polygon in the averaged-aperture cases: ", nonsimple)
    @test nonsimple > 0
end

@testset "Mimetic projection with an interface on a grid line" begin
    # The interface lies on the faces y = 1/2: no cell is cut. The two phase
    # values of those faces are identified (u^l = u^g) and both cells are uncut.
    n = 32
    flat(x, y) = y - 0.5
    predictor(phase, face) = face.axis == 1 ? (phase == 1 ? -0.5 : 0.5) + 0.1sin(2pi * face.center[2]) :
        (phase == 1 ? 0.3 : -0.2) * cos(2pi * face.center[1])
    case = _mimetic_case(n, flat; density = (10.0, 1.0))
    velocity = _mimetic_velocity(case, predictor)
    pressure = CellField(case.mesh)
    before = deepcopy(velocity)
    stats = project!(velocity, pressure, case.solver, 1.0e-2, case.workspace; tolerance = 1.0e-12)
    @test stats.converged
    @test stats.gridline_faces == n
    @test stats.cut_cells == 0 && stats.defective_cells == 0
    @test abs(stats.energy_identity_defect) <= 1.0e-12
    @test abs(stats.momentum_change[1]) <= 1.0e-12 * length(case.mesh.faces) *
        maximum(abs, case.workspace.metric_predictor)
    # Without cut cells 𝓜 is the diagonal phase mass W; identifying the phase
    # values is part of the orthogonal projection, so Pythagoras holds in the
    # full two-phase norm, including the jump energy removed on the interface.
    function phase_energy(fields)
        data = case.workspace.face_data
        return sum(eachindex(case.mesh.faces)) do index
            volume = case.mesh.face_dual_volumes[index]
            sum(1:2) do phase
                mass = (phase == 1 ? data.liquid_mass : data.gas_mass)[index]
                value = fields[phase][index]
                mass > 0 && isfinite(value) ? volume * mass * value^2 : 0.0
            end
        end
    end
    initial = (before[1].interior, before[2].interior)
    final = (velocity[1].interior, velocity[2].interior)
    initial_energy = phase_energy(initial)
    @test abs(
        initial_energy - phase_energy(final) -
            phase_energy((initial[1] .- final[1], initial[2] .- final[2])),
    ) <= 1.0e-12 * initial_energy
    # Normal continuity on the interface faces, idempotence.
    for (index, face) in pairs(case.mesh.faces)
        face.axis == 2 && abs(face.center[2] - 0.5) < 1.0e-12 || continue
        @test velocity[1].interior[index] == velocity[2].interior[index]
    end
    projected = deepcopy(velocity)
    project!(velocity, pressure, case.solver, 1.0e-2, case.workspace; tolerance = 1.0e-12)
    @test maximum(
        maximum(abs, filter(isfinite, velocity[phase].interior .- projected[phase].interior); init = 0.0)
            for phase in 1:2
    ) <= 1.0e-10 * _mimetic_maximum(projected)

    # Limit of the cut-cell construction: a chord approaching the face.
    flux(case, velocity) = map(eachindex(case.mesh.faces)) do index
        data = case.workspace.face_data
        TwoPhaseAMR._projection_phase_sum(
            data.liquid_aperture[index], velocity[1].interior[index],
            data.gas_aperture[index], velocity[2].interior[index],
        )
    end
    reference = flux(case, projected)
    distances = map((1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4)) do theta
        cut = _mimetic_case(n, (x, y) -> y - 0.5 - theta / n; density = (10.0, 1.0))
        u = _mimetic_velocity(cut, predictor)
        project!(u, CellField(cut.mesh), cut.solver, 1.0e-2, cut.workspace; tolerance = 1.0e-12)
        maximum(abs, flux(cut, u) .- reference)
    end
    println("mimetic grid-line limit: max mixture-flux distance for theta = 1e-1..1e-4: ", distances)
    @test all(diff(collect(distances)) .< 0)
    @test last(distances) <= 1.0e-2 * first(distances)

    # Rest states with the grid-aligned interface: gravity at density ratio
    # 1000 and a constant-curvature pressure jump through full time steps.
    mesh = BlockTreeMesh(block_size = (n, n), periodic = (true, false))
    interface = InterfaceField(mesh, flat; lipschitz = 1)
    flow = FlowSimulation(
        mesh, TwoPhase(
            interface; density = (1000.0, 1.0), dynamic_viscosity = (0.0, 0.0),
            gravity = (0.0, -9.81), surface_tension = SurfaceTension(0.3),
        );
        velocity_model = TwoVelocityModel(pressure_projection = MimeticProjection()),
        boundary = FreeSlip(),
    )
    for _ in 1:3
        stats = step!(
            flow.velocity, flow.pressure, flow.system, 1.0e-3, flow.workspace;
            tolerance = 1.0e-11, max_cycles = 2000,
        )
        @test stats.converged
    end
    @test _mimetic_maximum(flow.velocity) <= 1.0e-8
end
