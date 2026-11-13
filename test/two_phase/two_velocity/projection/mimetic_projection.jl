using Test, LinearAlgebra, SparseArrays, TwoPhaseAMR
include("../helpers.jl")
include("mimetic_lab_helpers.jl")

function _mimetic_case(
        n, levelset; density = (1.0, 1.0), metric = MFDCutCellMetric(),
        periodic = (true, false), lipschitz = 1.5,
    )
    mesh = BlockTreeMesh(; block_size = (n, n), periodic)
    interface = InterfaceField(mesh, levelset; lipschitz)
    solver = MimeticProjectionSolver(interface; density, metric)
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
    @test projection.metric isa MFDCutCellMetric
    @test projection.metric.small_cell_stabilization
    @test projection.metric.nonsimple_cells === :error
    @test MFDCutCellMetric(nonsimple_cells = :lumped).nonsimple_cells === :lumped
    @test !MFDCutCellMetric(small_cell_stabilization = false).small_cell_stabilization
    @test_throws ArgumentError MFDCutCellMetric(nonsimple_cells = :ignore)
    @test_throws ArgumentError MimeticProjection(metric = :mfd)
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
end

@testset "Mimetic metric: symmetry, definiteness, diagonal masses and consistency" begin
    for density in ((1.0, 1.0), (1000.0, 1.0))
        case = _mimetic_case(32, _mimetic_circle(32); density)
        assembly = TwoPhaseAMR._mimetic_assemble!(case.solver, case.workspace)
        band = assembly.band_matrix
        @test maximum(abs, band - band') <= 1.0e-14 * maximum(abs, band)
        @test isposdef(Symmetric(Matrix(band)))
        @test assembly.cut_cells > 0 && assembly.lumped_polygons == 0
        # Away from the band the metric is the diagonal phase mass of the code.
        off = findall(iszero, assembly.band_position)
        @test !isempty(off)
        @test assembly.diagonal[off] == assembly.dof_mass[off]
        # Consistency (audit thm. mm(b)): for equal densities the uniform
        # acceleration w = e_y (∇p = rho e_y) satisfies 𝓜 w = -Dᵀ p with
        # p = rho y at the chord midpoints of the cut cells, the centres elsewhere.
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
        # metric-free); the lumped fallback covers the triangle's kink cells.
        case = _mimetic_case(
            n, levelset; density = (3.0, 1.0),
            metric = MFDCutCellMetric(nonsimple_cells = :lumped),
        )
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

@testset "Mimetic projection rejects unsupported cut cells" begin
    # A film thinner than a cell has two chords per cut cell.
    film(x, y) = abs(y - 0.5 - 0.37 / 32) - 0.3 / 64
    case = _mimetic_case(32, film)
    velocity = _mimetic_velocity(case, (phase, face) -> face.axis == 2 ? -1.0 : 0.0)
    @test_throws ArgumentError project!(velocity, CellField(case.mesh), case.solver, 1.0, case.workspace)
    lumped = _mimetic_case(32, film; metric = MFDCutCellMetric(nonsimple_cells = :lumped))
    velocity = _mimetic_velocity(lumped, (phase, face) -> face.axis == 2 ? -1.0 : 0.0)
    stats = project!(velocity, CellField(lumped.mesh), lumped.solver, 1.0, lumped.workspace; tolerance = 1.0e-12)
    @test stats.converged
    @test stats.lumped_polygons > 0
    # A drop on a wall has cut cells at the domain boundary.
    for metric in (MFDCutCellMetric(), MFDCutCellMetric(nonsimple_cells = :lumped))
        wall = _mimetic_case(32, _mimetic_circle(32; y = 0.1); metric)
        velocity = _mimetic_velocity(wall, (phase, face) -> 0.0)
        @test_throws ArgumentError project!(velocity, CellField(wall.mesh), wall.solver, 1.0, wall.workspace)
    end
end

@testset "Mimetic apertures at corner clips next to pure cells" begin
    # The mean of the two PLIC traces halves a corner clip of a cut cell next
    # to a pure cell, which gives the cut cell a second micro-chord or a
    # zero-area polygon (circle n=64, wavy n=128, zero crossings of a cosine).
    # The mimetic interface takes the pure cell's aperture there instead: all
    # cut cells are simple and a uniform acceleration is projected to rest.
    cases = (
        (64, _mimetic_circle(64)),
        (128, (x, y) -> y - 0.5 - 0.37 / 128 - 0.05cos(2pi * x)),
        (32, (x, y) -> y - 0.5 - 0.01cos(2pi * x)),
    )
    for (n, levelset) in cases
        case = _mimetic_case(n, levelset)
        @test case.interface.phase_aperture_rule[] === :pure_neighbour
        velocity = _mimetic_velocity(case, (phase, face) -> face.axis == 2 ? -1.0 : 0.0)
        stats = project!(
            velocity, CellField(case.mesh), case.solver, 1.0, case.workspace;
            tolerance = 1.0e-10,
        )
        @test stats.converged
        @test stats.lumped_polygons == 0
        @test _mimetic_maximum(velocity) <= 1.0e-12
        # Only faces next to a pure cell change, to that cell's value; the
        # default rule is the mean of the traces.
        rule = copy(case.interface.phase_liquid_aperture)
        TwoPhaseAMR._set_phase_aperture_rule!(case.interface, :average)
        mean, _ = TwoPhaseAMR._current_phase_face_apertures!(case.interface)
        changed = findall(rule .!= mean)
        @test !isempty(changed)
        for index in changed
            face = case.mesh.faces[index]
            fractions = TwoPhaseAMR._conditional_fraction.(Ref(case.interface), (face.minus, face.plus))
            @test rule[index] in fractions
            @test count(c -> c == 0 || c == 1, fractions) == 1
        end
    end
    # The rule follows the projection of the two-phase model on the interface.
    mesh = BlockTreeMesh(block_size = (16, 16), periodic = (true, false))
    interface = InterfaceField(mesh, _mimetic_circle(16); lipschitz = 1)
    for (projection, rule) in ((MimeticProjection(), :pure_neighbour), (CMDGFMProjection(), :average))
        FlowSimulation(
            mesh, TwoPhase(interface; density = (1.0, 1.0), dynamic_viscosity = (0.0, 0.0));
            velocity_model = TwoVelocityModel(pressure_projection = projection),
            boundary = FreeSlip(),
        )
        @test interface.phase_aperture_rule[] === rule
    end
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
    @test stats.cut_cells == 0 && stats.lumped_polygons == 0
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
