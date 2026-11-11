using Test, TwoPhaseAMR, LinearAlgebra, SparseArrays
include("cmdgfm_reference_helpers.jl")

const CPR = TwoPhaseAMR

@testset "CMDGFM compact connectivity and disconnected pressure gauges" begin
    for adaptive in (false, true), periodic in ((true, true), (false, false))
        mesh = BlockTreeMesh(
            block_size = (4, 4), max_level = 2, periodic = periodic,
            refine = (key, origin, extent) -> adaptive &&
                (key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 1))
        )
        cache = CPR.CMDGFMPressureCache(Float64, typeof(mesh))
        cache.incidence = _cmdgfm_incidence(mesh, Float64)
        graph = CPR._cmdgfm_vertex_connectivity!(cache.vertex_patches, mesh)
        patches = CPR._vertex_patches(mesh, Float64)
        # Compare every endpoint's candidates, including periodic seams and
        # coarse/fine vertices, independently of the pressure equations.
        @test all(eachindex(mesh.faces)) do index
            face = mesh.faces[index]
            all(
                axis -> CPR._cmdgfm_vertex_candidates(graph, mesh, face, index, axis) ==
                    CPR._cmdgfm_vertex_candidates(patches, mesh, face, index, axis), 1:2
            )
        end
        columns, entries = graph.offsets, graph.faces
        @test CPR._cmdgfm_vertex_connectivity!(graph, mesh).offsets === columns
        @test graph.faces === entries
        for geometry in (
                nothing,
                (; active_cells = trues(ncells(mesh)), aperture = zeros(length(mesh.faces))),
                (;
                    active_cells = BitVector(cell % 7 != 0 for cell in 1:ncells(mesh)),
                    aperture = [
                        abs(face.center[1] - 0.5) < 1.0e-12 ? 0.0 : 1.0
                            for face in mesh.faces
                    ],
                ),
            )
            eligible = isnothing(geometry) ? trues(ncells(mesh)) : geometry.active_cells
            aperture = isnothing(geometry) ? ones(length(mesh.faces)) : geometry.aperture
            expected = _reference_gauge_partition(mesh, eligible, aperture)
            CPR._cmdgfm_gauge_partition!(cache, mesh, geometry)
            @test cache.pressure_indices == expected[1]
            @test cache.pressure_anchors == expected[2]
            for (local_index, cell) in pairs(cache.pressure_indices)
                @test cache.pressure_lookup[cell] == local_index
            end
        end
    end
end

@testset "CMDGFM pooled row-ordered CSC assembly" begin
    for T in (Float32, Float64)
        scratch = CPR.CMDGFMTriplets(T)
        matrix = spzeros(T, 0, 0)
        buffers = (matrix.colptr, rowvals(matrix), nonzeros(matrix))
        # Empty blocks, duplicate coefficients, cancellation and changing
        # dimensions/patterns must agree with generic sparse conversion.
        for (m, n, rows, columns, values) in (
                (0, 0, Int[], Int[], T[]),
                (3, 4, [1, 1, 1, 2, 2, 3], [3, 3, 1, 4, 4, 2], T[2, -2, 0, 1, 3, -4]),
                (2, 5, [1, 1, 2, 2, 2], [5, 3, 1, 1, 1], T[0.2, 3, 1, -1, 0.01]),
                (0, 3, Int[], Int[], T[]),
            )
            append!(empty!(scratch.rows), rows)
            append!(empty!(scratch.columns), columns)
            append!(empty!(scratch.values), values)
            matrix = CPR._cmdgfm_sorted_rows_csc!(matrix, scratch, m, n)
            reference = sparse(rows, columns, values, m, n)
            @test matrix == reference
            @test matrix.colptr == reference.colptr
            @test rowvals(matrix) == rowvals(reference)
            @test nonzeros(matrix) == nonzeros(reference)
            @test matrix.colptr === buffers[1]
            @test rowvals(matrix) === buffers[2]
            @test nonzeros(matrix) === buffers[3]
        end
    end
end

function _pressure_stencil_allocations(solver, work, data)
    cache = work.pressure_cache
    for _ in 1:3
        CPR._cmdgfm_jump_stencils(
            solver, data; incident_faces = cache.incidence,
            vertex_patches = cache.vertex_patches, storage = cache.stencil_storage
        )
    end
    return @allocated CPR._cmdgfm_jump_stencils(
        solver, data;
        incident_faces = cache.incidence, vertex_patches = cache.vertex_patches,
        storage = cache.stencil_storage
    )
end

@testset "CMDGFM cached stencil traversal has bounded allocation" begin
    mesh = BlockTreeMesh(block_size = (64, 64), periodic = (true, true))
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.453, y - 0.487) - 0.2637;
        lipschitz = 1
    )
    solver = CMDGFMProjectionSolver(interface; density = (1000.0, 1.0))
    work = CMDGFMProjectionWorkspace(solver)
    data = CPR._projection_face_data(solver, work)
    CPR._cmdgfm_pressure_cache!(work.pressure_cache, solver, data)
    # Exclude compilation and capacity growth. Allow small Julia-version-
    # dependent closure/sort overhead, but reject the former per-candidate
    # boxing (hundreds of kB on this mesh). ILU is not part of this gate.
    _pressure_stencil_allocations(solver, work, data)
    @test _pressure_stencil_allocations(solver, work, data) < 32768
end

@testset "CMDGFM direct incidence on periodic and bounded meshes" begin
    for dimensions in ((2, 2), (4, 2), (2, 4)), periodic in ((true, true), (false, false)),
            T in (Float32, Float64)
        mesh = BlockTreeMesh(; block_size = dimensions, periodic)
        actual, reference = _cmdgfm_incidence(mesh, T), _reference_projection_incidence(mesh, T)
        @test actual == reference
        @test actual.colptr == reference.colptr
        @test rowvals(actual) == rowvals(reference)
        @test nonzeros(actual) == nonzeros(reference)
    end
end

_pressure_reuse_velocity(mesh) = (
    FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            sinpi(2y) + 0.1cospi(2x) : -0.2sinpi(2x)
    ),
    FaceVelocity(
        mesh, (x, y, time, axis) -> axis == 1 ?
            0.3cospi(2y) : sinpi(2x)
    ),
)

function _cmdgfm_pressure_predictor!(velocity, pressure, solver, work, dt)
    cache = work.pressure_cache
    data = CPR._projection_face_data(solver, work; storage = cache.face_data)
    cache.face_data = data
    operator = CPR._cmdgfm_pressure_cache!(cache, solver, data).operator
    CPR._projection_weighted_gradient!(
        work.pressure_gradient, pressure, operator, cache.elliptic,
        solver.interface.geometry,
    )
    CPR._projection_correct_phase_velocities!(
        velocity, (work.pressure_gradient, work.pressure_gradient), dt, eltype(pressure),
    )
    return velocity
end
_pressure_model_interface(model) = model.interface

@testset "typed CMDGFM cache with Float32" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0),
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.273f0; lipschitz = 1)
    solver = CMDGFMProjectionSolver(interface; density = (10.0f0, 1.0f0))
    work = CMDGFMProjectionWorkspace(solver)
    velocity = _pressure_reuse_velocity(mesh)
    CPR._mark_inactive_phase_velocities!(velocity, interface)
    assembled = CPR._assemble_cmdgfm_system!(solver, work, 0.02f0, velocity)
    block = CPR._cmdgfm_block_preconditioner(solver, assembled, work)
    @test work.block_preconditioner === block
    @test typeof(block) <: fieldtype(typeof(work), :block_preconditioner)
    @test typeof(assembled.pressure_operator) <:
    fieldtype(typeof(work.pressure_cache), :operator)
    @test typeof(assembled.data) <: fieldtype(typeof(work.pressure_cache), :face_data)
    @test eltype(work.pressure_increment) === Float32
    pressure = CellField(mesh, Float32; init = 3.0f0)
    resting = (FaceVelocity(mesh, Float32), FaceVelocity(mesh, Float32))
    stats = project!(resting, pressure, solver, 0.02f0, work)
    @test stats.converged
    for values in pressure.blocks
        @test all(==(3.0f0), values)
    end
    @test all(block -> all(iszero, block), work.pressure_increment.blocks)
end

@testset "lazy two-velocity workspace types across AMR" begin
    for T in (Float32, Float64),
            transport in (FrozenInterfaceFluxes(), ProjectedLiquidVelocity()),
            cut in (false, true)
        mesh = BlockTreeMesh(
            base_mesh = (8, 4), block_size = (4, 4), max_level = 1,
            periodic = (true, true), origin = (zero(T), zero(T)),
            extent = (one(T), one(T)),
        )
        geometry = cut ? GeometryField(mesh, (x, y) -> y - T(0.08); lipschitz = 1) : nothing
        interface = InterfaceField(
            mesh, (x, y) -> x - T(0.273); geometry, lipschitz = 1,
        )
        flow = FlowSimulation(
            mesh, TwoPhase(
                interface; density = (T(10), one(T)),
                dynamic_viscosity = (zero(T), zero(T))
            );
            velocity_model = TwoVelocityModel(interface_transport = transport),
            velocity = _pressure_reuse_velocity(mesh),
        )
        phase = flow.workspace.two_phase
        @test (@inferred _pressure_model_interface(flow.system.two_phase)) === interface
        @test isnothing(phase.two_velocity[])
        @test all(isconcretetype, Base.uniontypes(eltype(typeof(phase.two_velocity))))
        state = CPR._two_velocity_flow_workspace!(phase, flow.system.two_phase)
        @test typeof(state) <: eltype(typeof(phase.two_velocity))
        @test (@inferred CPR._two_velocity_flow_workspace!(phase, flow.system.two_phase)) === state
        projection_workspace = state.projection_workspace

        adaptation = isnothing(geometry) ?
            AMRWorkspace(mesh, interface, flow.velocity, flow.pressure) :
            AMRWorkspace(mesh, interface, geometry, flow.velocity, flow.pressure)
        labels = [block.key.i == 0 ? :refine : :keep for block in mesh.blocks]
        @test adapt!(adaptation, labels; step = 1, force = true).changed
        CPR._rebuild_topology_dependent_state!(flow.system, flow.workspace; dt = T(0.001))
        phase = flow.workspace.two_phase
        @test all(isconcretetype, Base.uniontypes(eltype(typeof(phase.two_velocity))))
        @test phase.two_velocity[].projection_workspace === projection_workspace
        state = CPR._two_velocity_flow_workspace!(phase, flow.system.two_phase)
        @test typeof(state) <: eltype(typeof(phase.two_velocity))
        @test state.advection === phase.advection
    end
end

# Independent, deliberately allocation-heavy reference: retain the original
# full sparse products so cached/local assembly cannot validate itself.
function _cmdgfm_fresh_reference(solver, work, velocity, dt)
    mesh = solver.interface.mesh
    T = eltype(solver.interface)
    data = CPR._projection_face_data(solver, work)
    op = CPR._projection_pressure_operator(solver)
    incidence = _reference_projection_incidence(mesh, T)
    weights = isnothing(solver.interface.geometry) ? ones(T, length(mesh.faces)) :
        solver.interface.geometry.aperture
    active_cells = isnothing(solver.interface.geometry) ? trues(ncells(mesh)) :
        solver.interface.geometry.active_cells
    indices, anchors = _reference_gauge_partition(mesh, active_cells, weights)
    all_stencils = CPR._cmdgfm_jump_stencils(solver, data)
    stencils = filter(s -> s.resolved, all_stencils)
    mixed = [s.face for s in stencils]
    Ip = incidence[:, indices]
    PP = op.stiffness[indices, indices]
    PX = transpose(Ip[mixed, :]) * spdiagm(0 => data.coupling[mixed])
    G = _cmdgfm_common_gradient_matrix(op, incidence, solver.interface.geometry)
    XP, XX, rhs_jump, = CPR._cmdgfm_jump_blocks(
        stencils, mixed, data,
        G, indices, velocity, dt, T
    )
    mixture = CPR._phase_product.(data.liquid_aperture, velocity[1].interior) .+
        CPR._phase_product.(data.gas_aperture, velocity[2].interior)
    # The predictors are no-permeation, so boundary faces add no mixture flux.
    rhs = vcat((transpose(Ip) * mixture) ./ T(dt), rhs_jump)
    return (; matrix = [PP PX; XP XX], rhs, indices, anchors, mixed, op, data)
end

@testset "CMDGFM AMR schedule-type transitions retain pressure workspace" begin
    mesh = BlockTreeMesh(
        base_mesh = (8, 4), block_size = (4, 4),
        max_level = 1, periodic = (true, true)
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.273; lipschitz = 1)
    flow = FlowSimulation(
        mesh,
        TwoPhase(
            interface; density = (10.0, 1.0),
            dynamic_viscosity = (0.0, 0.0),
        );
        velocity_model = TwoVelocityModel(),
        velocity = _pressure_reuse_velocity(mesh),
    )
    phase = flow.workspace.two_phase
    @test isnothing(phase.two_velocity[])
    @test fieldtype(typeof(flow.system.two_phase), :interface) === typeof(interface)
    @test fieldtype(typeof(phase), :advection) === typeof(phase.advection)
    @test fieldtype(typeof(phase), :momentum) === typeof(phase.momentum)
    @test all(isconcretetype, Base.uniontypes(eltype(typeof(phase.two_velocity))))
    @test project!(flow).converged
    original_work = flow.workspace.two_phase.two_velocity[].projection_workspace
    increment_scratch = original_work.pressure_increment
    pressure_work = original_work.block_preconditioner.pressure_preconditioner
    pressure_buffers = (
        pressure_work.full_right_hand_side, pressure_work.full_solution,
        pressure_work.fine_coefficients, pressure_work.pressure_indices,
        pressure_work.pressure_anchors, pressure_work.mesh_keys,
    )
    @test flow.workspace.two_phase.two_velocity[].momentum.x.workspace.schedule isa
        CPR.UniformDualAdvectionSchedule
    adaptation = AMRWorkspace(mesh, interface, flow.velocity, flow.pressure)
    for (step, refine_left) in ((1, true), (100, false), (200, true))
        labels = refine_left ? [block.key.i == 0 ? :refine : :keep for block in mesh.blocks] :
            fill(:coarsen, nblocks(mesh))
        @test adapt!(adaptation, labels; step, force = true).changed
        CPR._rebuild_topology_dependent_state!(flow.system, flow.workspace; dt = 0.001)
        phase = flow.workspace.two_phase
        @test all(isconcretetype, Base.uniontypes(eltype(typeof(phase.two_velocity))))
        state = flow.workspace.two_phase.two_velocity[]
        @test typeof(state) <: eltype(typeof(phase.two_velocity))
        @test state.projection_workspace === original_work
        @test (state.momentum.x.workspace.schedule isa CPR.UniformDualAdvectionSchedule) == !refine_left
        @test !CPR._cmdgfm_same_topology(original_work.increment_keys, mesh)
        @test project!(flow).converged
        @test original_work.pressure_increment === increment_scratch
        @test length(increment_scratch.blocks) == nblocks(mesh)
        @test CPR._cmdgfm_same_topology(original_work.increment_keys, mesh)
        current_pressure = original_work.block_preconditioner.pressure_preconditioner
        for (actual, original) in zip(
                (
                    current_pressure.full_right_hand_side,
                    current_pressure.full_solution, current_pressure.fine_coefficients,
                    current_pressure.pressure_indices, current_pressure.pressure_anchors,
                    current_pressure.mesh_keys,
                ), pressure_buffers
            )
            @test actual === original
        end
        @test CPR._cmdgfm_same_topology(original_work.pressure_cache.keys, mesh)
    end
end

@testset "CMDGFM cached assembly equals fresh assembly" begin
    for adaptive in (false, true), cut in (false, true), density in ((1.0, 1.0), (1000.0, 1.0))
        mesh = adaptive ? BlockTreeMesh(
                block_size = (4, 4), max_level = 2,
                refine = (key, origin, extent) -> key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 1)
            ) :
            BlockTreeMesh(block_size = (12, 12), periodic = (true, false))
        geometry = cut ? GeometryField(
                mesh, (x, y) -> y - 0.09 - 0.04x;
                lipschitz = hypot(1, 0.04)
            ) : nothing
        interface = InterfaceField(
            mesh, (x, y) -> hypot(x - 0.453, y - 0.487) - 0.2637;
            geometry, lipschitz = 1
        )
        solver = CMDGFMProjectionSolver(interface; density)
        work = CMDGFMProjectionWorkspace(solver)
        for (owner, names) in (
                    (work, (:block_preconditioner, :krylov, :pressure_cache)),
                    (
                        work.pressure_cache, (
                            :operator, :face_data, :metric_source,
                            :gradient_metric_rows, :stiffness_source,
                        ),
                    ),
                ), name in names
            @test all(isconcretetype, Base.uniontypes(fieldtype(typeof(owner), name)))
        end
        @test isnothing(work.pressure_cache.operator)
        @test isnothing(work.pressure_cache.face_data)
        velocity = _pressure_reuse_velocity(mesh)
        CPR._mark_inactive_phase_velocities!(velocity, interface)
        first = CPR._assemble_cmdgfm_system!(solver, work, 0.02, velocity)
        @test first.data === work.pressure_cache.face_data
        reference_incidence = _reference_projection_incidence(mesh, Float64)
        @test first.incidence == reference_incidence
        @test first.incidence.colptr == reference_incidence.colptr
        @test rowvals(first.incidence) == rowvals(reference_incidence)
        cached_op = first.pressure_operator
        cached_incidence = first.incidence
        cached_metric = cached_op.gradient_operator.inverse_metric
        cached_elliptic = work.pressure_cache.elliptic
        block_buffers = map(
            (
                work.pressure_cache.pressure_jump, first.jump_pressure,
                first.jump_block,
            )
        ) do block
            (block.colptr, rowvals(block), nonzeros(block))
        end
        for shift in (0.0, 0.023)
            initialize!(interface, (x, y) -> hypot(x - 0.453 - shift, y - 0.487) - 0.2637)
            velocity = _pressure_reuse_velocity(mesh)
            CPR._mark_inactive_phase_velocities!(velocity, interface)
            reference = _cmdgfm_fresh_reference(solver, work, velocity, 0.02)
            actual = CPR._assemble_cmdgfm_system!(solver, work, 0.02, velocity)
            @test actual.pressure_operator === cached_op
            @test actual.incidence === cached_incidence
            @test actual.pressure_operator.gradient_operator.inverse_metric === cached_metric
            @test actual.pressure_indices == reference.indices
            @test actual.pressure_anchors == reference.anchors
            @test actual.mixed_faces == reference.mixed
            @test actual.matrix.colptr === first.matrix.colptr
            @test rowvals(actual.matrix) === rowvals(first.matrix)
            @test nonzeros(actual.matrix) === nonzeros(first.matrix)
            for (block, buffers) in zip(
                    (
                        work.pressure_cache.pressure_jump,
                        actual.jump_pressure, actual.jump_block,
                    ), block_buffers
                )
                @test block.colptr === buffers[1]
                @test rowvals(block) === buffers[2]
                @test nonzeros(block) === buffers[3]
            end
            @test norm(actual.matrix - reference.matrix, Inf) <=
                2.0e-12 * max(norm(reference.matrix, Inf), 1.0)
            @test actual.right_hand_side ≈ reference.rhs rtol = 2.0e-13 atol = 2.0e-13
            vector = sin.(1:size(actual.matrix, 2))
            @test actual.matrix * vector ≈ reference.matrix * vector rtol = 2.0e-12 atol = 2.0e-12
            p = CellField(mesh, (x, y) -> sinpi(2x) + 0.3cospi(2y))
            work.jump .= cos.(1:length(mesh.faces))
            fresh_gradient = (FaceVelocity(mesh), FaceVelocity(mesh))
            cached_gradient = (FaceVelocity(mesh), FaceVelocity(mesh))
            cmdgfm_gradient!(fresh_gradient, p, solver, work, reference.op, reference.data)
            cmdgfm_gradient!(cached_gradient, p, solver, work, actual.pressure_operator, actual.data)
            @test work.pressure_cache.elliptic === cached_elliptic
            for phase in 1:2
                @test cached_gradient[phase].interior ≈ fresh_gradient[phase].interior rtol = 2.0e-12 atol = 2.0e-12
            end
        end
        if cut
            @test_throws ArgumentError CPR._assemble_cmdgfm_system!(
                solver, work, 0.02, velocity;
                shared_operator = cached_op
            )
        else
            stale = CPR._projection_pressure_operator(solver)
            initialize!(interface, (x, y) -> hypot(x - 0.593, y - 0.487) - 0.2637)
            if density[1] == density[2]
                # Equal-density coefficients do not become stale when the
                # interface moves; the same operator is still compatible.
                @test CPR._assemble_cmdgfm_system!(
                    solver, work, 0.02, velocity;
                    shared_operator = stale
                ).pressure_operator === stale
            else
                @test_throws ArgumentError CPR._assemble_cmdgfm_system!(
                    solver, work, 0.02, velocity;
                    shared_operator = stale
                )
            end
        end
    end
end

@testset "CMDGFM shares only compatible scalar pressure data" begin
    for coarse_size in (64, 16)
        mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, false))
        interface = InterfaceField(
            mesh, (x, y) -> 0.503 + 0.01cospi(2x) - y;
            lipschitz = 1.01
        )
        flow = FlowSimulation(
            mesh,
            TwoPhase(
                interface; density = (1000.0, 1.0),
                dynamic_viscosity = (0.0, 0.0),
            );
            velocity_model = TwoVelocityModel(
                pressure_projection = CMDGFMProjection(),
            ),
            velocity = _pressure_reuse_velocity(mesh),
            numerics = NavierStokesOptions(
                multigrid = MultigridOptions(coarse_size = coarse_size),
            ),
        )
        stats = project!(flow; tolerance = 1.0e-8, max_cycles = 2000)
        @test stats.converged
        work = flow.workspace.two_phase.two_velocity[].projection_workspace
        pc = work.block_preconditioner.pressure_preconditioner
        @test work.pressure_cache.operator === flow.system.pressure_solver.operator
        @test (pc.multigrid === flow.system.pressure_solver.preconditioner) == (coarse_size == 64)
    end
end

@testset "CMDGFM initial guesses preserve the converged projection" begin
    for pure in (false, true), density in ((1.0, 1.0), (1000.0, 1.0), (1.0, 1000.0))
        mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
        interface = InterfaceField(
            mesh, pure ? ((x, y) -> -1.0) :
                ((x, y) -> hypot(x - 0.453, y - 0.487) - 0.2637); lipschitz = 1
        )
        solver = CMDGFMProjectionSolver(interface; density)
        cold_work = CMDGFMProjectionWorkspace(solver)
        warm_work = CMDGFMProjectionWorkspace(solver)
        cold_pressure, warm_pressure = CellField(mesh), CellField(mesh)
        tolerance = 1.0e-11
        # The retained guess is a previous increment, not physical pressure.
        # Both paths form the old-pressure predictor on the current interface.
        for shift in (0.0, 0.029)
            pure || initialize!(
                interface,
                (x, y) -> hypot(x - 0.453 - shift, y - 0.487) - 0.2637
            )
            for block in eachindex(cold_pressure.blocks)
                copyto!(warm_pressure.blocks[block], cold_pressure.blocks[block])
            end
            cold_velocity = _pressure_reuse_velocity(mesh)
            warm_velocity = _pressure_reuse_velocity(mesh)
            _cmdgfm_pressure_predictor!(cold_velocity, cold_pressure, solver, cold_work, 0.02)
            _cmdgfm_pressure_predictor!(warm_velocity, warm_pressure, solver, warm_work, 0.02)
            cold = project!(
                cold_velocity, cold_pressure, solver, 0.02, cold_work;
                tolerance, max_iterations = 4000, warm_start = false
            )
            warm = project!(
                warm_velocity, warm_pressure, solver, 0.02, warm_work;
                tolerance, max_iterations = 4000
            )
            @test cold.converged && warm.converged
            @test !cold_work.initial_guess_used
            @test cold_work.mixed_faces == warm_work.mixed_faces
            @test warm.mixture_volume_defect <= tolerance
            @test warm.normal_velocity_defect <= tolerance
            @test warm.momentum_identity_error <= 1024eps() * maximum(density)
            @test maximum(abs, warm.momentum_error) <= 1024eps() * maximum(density)
            @test warm_work.solution ≈ cold_work.solution rtol = 2.0e-8 atol = 2.0e-8
            for phase in 1:2
                a, b = cold_velocity[phase].interior, warm_velocity[phase].interior
                @test isnan.(a) == isnan.(b)
                active = findall(isfinite, a)
                @test b[active] ≈ a[active] rtol = 2.0e-8 atol = 2.0e-8
            end
        end
    end
end

@testset "CMDGFM initial-guess residual gate and zero fallback" begin
    # With no jump unknowns, the previous increment alone is a candidate.
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    interface = InterfaceField(mesh, (x, y) -> -1.0; lipschitz = 1)
    solver = CMDGFMProjectionSolver(interface)
    work = CMDGFMProjectionWorkspace(solver)
    p = CellField(mesh)
    velocity = _pressure_reuse_velocity(mesh)
    stats = project!(
        velocity, p, solver, 0.02, work;
        tolerance = 1.0e-11, max_iterations = 2000, warm_start = false
    )
    @test stats.converged
    previous_increment = copy(work.pressure_increment)
    assembled = CPR._assemble_cmdgfm_system!(
        solver, work, 0.02,
        _pressure_reuse_velocity(mesh)
    )
    pc = CPR._cmdgfm_preconditioner(solver, assembled, work)
    fill!(work.solution, 0.0)
    @test CPR._cmdgfm_initial_solution!(work, assembled, previous_increment, 0.02, pc)
    @test norm(work.correction_rhs) < 1.0e-8 * norm(assembled.right_hand_side)
    for invalid_guess in (NaN, 1.0e30)
        # An invalid or grossly inaccurate increment guess must not enter the
        # solve. The physical pressure and active phase velocities stay valid.
        for block in eachindex(mesh.blocks)
            values = interior(previous_increment, block)
            values .= invalid_guess .* reshape(1:length(values), size(values))
        end
        fill!(work.solution, 0.0)
        @test !CPR._cmdgfm_initial_solution!(work, assembled, previous_increment, 0.02, pc)
        @test all(iszero, work.solution)
        @test work.correction_rhs == assembled.right_hand_side
    end
    # No current jump ILU: do not guess a mixed-face jump at all.
    initialize!(interface, (x, y) -> x - 0.453)
    assembled = CPR._assemble_cmdgfm_system!(
        solver, work, 0.02,
        _pressure_reuse_velocity(mesh)
    )
    resize!(work.solution, length(assembled.right_hand_side))
    resize!(work.correction_rhs, length(assembled.right_hand_side))
    fill!(work.solution, 0.0)
    @test !CPR._cmdgfm_initial_solution!(work, assembled, previous_increment, 0.02, nothing)
    @test all(iszero, work.solution)
    @test work.correction_rhs == assembled.right_hand_side
end

@testset "CMDGFM jump guess uses the current residual" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.27; lipschitz = 1,
    )
    solver = CMDGFMProjectionSolver(interface; density = (10.0, 1.0))
    work = CMDGFMProjectionWorkspace(solver)
    velocity = _pressure_reuse_velocity(mesh)
    CPR._mark_inactive_phase_velocities!(velocity, interface)
    assembled = CPR._assemble_cmdgfm_system!(solver, work, 0.02, velocity)
    pc = CPR._cmdgfm_preconditioner(solver, assembled, work)
    np, nx = length(assembled.pressure_indices), length(assembled.mixed_faces)
    @test nx > 0

    # Manufacture a current system whose solution has only a jump. Its jump
    # RHS must be solved with the current ILU, not the old stored face jump.
    target = zeros(np + nx)
    target[(np + 1):end] .= sin.(1:nx)
    mul!(assembled.right_hand_side, assembled.matrix, target)
    resize!(work.solution, length(target))
    resize!(work.correction_rhs, length(target))
    fill!(work.solution, 0)
    fill!(work.jump, 1.0e6)
    zero_defect = maximum(CPR._cmdgfm_residual_defects(assembled.right_hand_side, assembled, 0.02))
    zero_l2 = norm(assembled.right_hand_side)
    @test CPR._cmdgfm_initial_solution!(work, assembled, nothing, 0.02, pc)
    residual = assembled.right_hand_side - assembled.matrix * work.solution
    @test work.correction_rhs ≈ residual rtol = 1.0e-12 atol = 1.0e-12
    @test maximum(CPR._cmdgfm_residual_defects(residual, assembled, 0.02)) < 0.95zero_defect
    @test norm(residual) < zero_l2
    candidate = copy(work.solution)
    fill!(work.solution, 0)
    fill!(work.jump, -1.0e6)
    @test CPR._cmdgfm_initial_solution!(work, assembled, nothing, 0.02, pc)
    @test work.solution ≈ candidate rtol = 1.0e-12 atol = 1.0e-12
end

@testset "CMDGFM reuses an increment rather than the physical pressure" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    interface = InterfaceField(mesh, (x, y) -> -1.0; lipschitz = 1)
    solver = CMDGFMProjectionSolver(interface)
    work = CMDGFMProjectionWorkspace(solver)
    first = project!(
        _pressure_reuse_velocity(mesh), CellField(mesh), solver, 0.02, work;
        tolerance = 1.0e-11, max_iterations = 2000, warm_start = false,
    )
    @test first.converged
    first_increment = copy(work.pressure_increment)

    # A constant physical pressure has zero predictor gradient. Its gauge
    # differs from the nonzero cached increment and cannot itself be a guess.
    pressure = CellField(mesh; init = 17.0)
    second = project!(
        _pressure_reuse_velocity(mesh), pressure, solver, 0.02, work;
        tolerance = 1.0e-11, max_iterations = 2000,
    )
    @test second.converged
    @test work.initial_guess_used
    for block in eachindex(pressure.blocks)
        @test interior(work.pressure_increment, block) ≈
            interior(first_increment, block) rtol = 1.0e-10 atol = 1.0e-10
        @test interior(pressure, block) ≈
            17.0 .+ interior(first_increment, block) rtol = 1.0e-10 atol = 1.0e-10
    end
end

@testset "CMDGFM incremental projection satisfies the coupled system" begin
    dt = 0.02
    for adaptive in (false, true), cut in (false, true),
            density in ((1.0, 1.0), (1000.0, 1.0))
        mesh = adaptive ? BlockTreeMesh(
                block_size = (4, 4), max_level = 2,
                refine = (key, origin, extent) -> key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 1),
            ) : BlockTreeMesh(
                block_size = (8, 8), periodic = cut ? (true, false) : (true, true),
            )
        geometry = cut ? GeometryField(
                mesh, (x, y) -> y - 0.09 - 0.04x;
                lipschitz = hypot(1, 0.04),
            ) : nothing
        interface = InterfaceField(
            mesh, (x, y) -> hypot(x - 0.453, y - 0.487) - 0.2637;
            geometry, lipschitz = 1,
        )
        solver = CMDGFMProjectionSolver(interface; density)
        reference_work = CMDGFMProjectionWorkspace(solver)
        increment_work = CMDGFMProjectionWorkspace(solver)
        increment_velocity = _pressure_reuse_velocity(mesh)
        enforce_no_permeation!(increment_velocity)
        pressure = CellField(mesh, (x, y) -> 2.3 + 0.2sinpi(2x) + 0.3cospi(2y))
        old_pressure = copy(pressure)
        _cmdgfm_pressure_predictor!(increment_velocity, pressure, solver, increment_work, dt)
        predictor = (copy(increment_velocity[1]), copy(increment_velocity[2]))
        reference = _cmdgfm_fresh_reference(solver, reference_work, predictor, dt)
        direct_solution = reference.matrix \ reference.rhs
        np = length(reference.indices)

        incremental = project!(
            increment_velocity, pressure, solver, dt, increment_work;
            tolerance = 1.0e-11, max_iterations = 4000, warm_start = false,
        )
        @test incremental.converged
        @test incremental.mixture_volume_defect <= 1.0e-11
        @test incremental.normal_velocity_defect <= 1.0e-11
        @test incremental.momentum_identity_error <= 1024eps() * maximum(density)
        @test increment_work.pressure_indices == reference.indices
        @test increment_work.mixed_faces == reference.mixed
        @test norm(increment_work.matrix - reference.matrix, Inf) <=
            2.0e-12 * max(norm(reference.matrix, Inf), 1.0)
        @test increment_work.right_hand_side ≈ reference.rhs rtol = 2.0e-12 atol = 2.0e-12
        residual = reference.rhs - reference.matrix * increment_work.solution
        residual_data = (;
            pressure_indices = reference.indices,
            pressure_operator = reference.op,
            mixed_faces = reference.mixed,
        )
        defects = CPR._cmdgfm_residual_defects(residual, residual_data, dt)
        @test maximum(defects) <= 1.0e-11
        @test increment_work.solution ≈ direct_solution rtol = 1.0e-6 atol = 1.0e-4

        gradients = (FaceVelocity(mesh), FaceVelocity(mesh))
        cmdgfm_gradient!(gradients, increment_work.pressure_increment, solver, increment_work)
        direct_increment = CellField(mesh)
        CPR._copy_projection_cell_solution!(
            direct_increment, zip(reference.indices, view(direct_solution, 1:np)),
        )
        fill!(reference_work.jump, 0)
        reference_work.jump[reference.mixed] .= view(direct_solution, (np + 1):length(direct_solution))
        direct_gradients = (FaceVelocity(mesh), FaceVelocity(mesh))
        cmdgfm_gradient!(direct_gradients, direct_increment, solver, reference_work)
        for phase in 1:2
            b = increment_velocity[phase].interior
            active = findall(isfinite, b)
            @test !isempty(active)
            name = phase == 1 ? :liquid : :gas
            @test all(eachindex(mesh.faces)) do index
                isfinite(b[index]) ==
                    CPR._phase_face_is_active(interface, index, mesh.faces[index], name)
            end
            @test b[active] ≈
                predictor[phase].interior[active] .- dt .* direct_gradients[phase].interior[active] rtol = 1.0e-6 atol = 1.0e-7
            @test b[active] ≈
                predictor[phase].interior[active] .- dt .* gradients[phase].interior[active] rtol = 1.0e-12 atol = 1.0e-12
        end

        # Accumulation includes the retained correction only; the original
        # gauge and inactive values survive, and halos are rebuilt.
        expected = copy(old_pressure)
        for block in eachindex(expected.blocks)
            interior(expected, block) .+= interior(increment_work.pressure_increment, block)
        end
        fill_halo!(expected)
        for block in eachindex(expected.blocks)
            @test pressure.blocks[block] ≈ expected.blocks[block] rtol = 0 atol = 1.0e-12
        end
        refs = reference.op.references
        for (index, cell) in pairs(reference.indices)
            anchor = reference.anchors[index]
            @test CPR._get(increment_work.pressure_increment, refs[cell]) ==
                increment_work.solution[index]
            @test iszero(CPR._get(increment_work.pressure_increment, refs[anchor]))
            @test isapprox(
                CPR._get(pressure, refs[cell]),
                CPR._get(old_pressure, refs[cell]) + direct_solution[index];
                rtol = 1.0e-6, atol = 1.0e-4,
            )
        end
    end
end

@testset "CMDGFM zero increment preserves disconnected pressure gauges" begin
    # A periodic x seam would reconnect the two sides of the solid strip.
    mesh = BlockTreeMesh(block_size = (16, 8), periodic = (false, false))
    geometry = GeometryField(mesh, (x, y) -> abs(x - 0.5) - 0.09; lipschitz = 1)
    interface = InterfaceField(mesh, (x, y) -> x - 0.27; geometry, lipschitz = 1)
    solver = CMDGFMProjectionSolver(interface; density = (10.0, 1.0))
    work = CMDGFMProjectionWorkspace(solver)
    pressure = CellField(mesh, (x, y) -> 5 + 3x - 2y)
    expected = copy(pressure)
    fill_halo!(expected)
    # A zero predictor can arise when the old-pressure gradient is exactly
    # balanced by the other predictor forces; it requires a zero increment.
    velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
    stats = project!(velocity, pressure, solver, 0.02, work)
    @test stats.converged
    @test iszero(stats.initial_residual)
    @test length(unique(work.pressure_cache.pressure_anchors)) >= 2
    @test all(iszero, work.solution)
    @test all(iszero, work.jump)
    for block in eachindex(expected.blocks)
        @test pressure.blocks[block] == expected.blocks[block]
        @test all(iszero, work.pressure_increment.blocks[block])
    end
end

@testset "CMDGFM shortened-step inherited defect RHS" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.453, y - 0.487) - 0.2637; lipschitz = 1,
    )
    solver = CMDGFMProjectionSolver(interface; density = (10.0, 1.0))
    work = CMDGFMProjectionWorkspace(solver)
    old_velocity = _pressure_reuse_velocity(mesh)
    CPR._mark_inactive_phase_velocities!(old_velocity, interface)
    old_data = CPR._projection_face_data(solver, work)
    inherited_flux = zeros(ncells(mesh))
    CPR._cmdgfm_cell_flux!(inherited_flux, old_data, interface, old_velocity)
    @test maximum(abs, inherited_flux) > 0

    # Transport can activate a phase that had no owned velocity on the old
    # geometry; the saved cell flux must still use the old apertures.
    initialize!(interface, (x, y) -> hypot(x - 0.573, y - 0.487) - 0.2637)
    reconstruct!(interface)
    @test any(((1, :liquid), (2, :gas))) do pair
        phase, name = pair
        any(eachindex(mesh.faces)) do index
            isnan(old_velocity[phase].interior[index]) &&
                CPR._phase_face_is_active(interface, index, mesh.faces[index], name)
        end
    end

    predictor = _pressure_reuse_velocity(mesh)
    CPR._mark_inactive_phase_velocities!(predictor, interface)
    dt, reference_dt = 2.0e-5, 0.02
    baseline_system = CPR._assemble_cmdgfm_system!(solver, work, dt, predictor)
    baseline = copy(baseline_system.right_hand_side)
    indices = copy(baseline_system.pressure_indices)
    np = length(indices)
    @test np < length(baseline)
    @test maximum(abs, view(baseline, (np + 1):length(baseline))) > 0
    without_flux = CPR._assemble_cmdgfm_system!(
        solver, work, dt, predictor; _reference_dt = reference_dt,
    )
    @test without_flux.right_hand_side == baseline
    corrected = CPR._assemble_cmdgfm_system!(
        solver, work, dt, predictor;
        _inherited_cell_flux = inherited_flux, _reference_dt = reference_dt,
    )
    weight = (reference_dt - dt) / (dt * reference_dt)
    @test corrected.pressure_indices == indices
    @test corrected.right_hand_side[1:np] ≈
        baseline[1:np] .- weight .* inherited_flux[indices] rtol = 2.0e-13 atol = 2.0e-10
    @test corrected.right_hand_side[(np + 1):end] == baseline[(np + 1):end]

    # An ordinary step must retain the exact old assembly, not merely an
    # approximately equal result after cancellation.
    ordinary = copy(
        CPR._assemble_cmdgfm_system!(
            solver, work, reference_dt, predictor,
        ).right_hand_side
    )
    same_step = CPR._assemble_cmdgfm_system!(
        solver, work, reference_dt, predictor;
        _inherited_cell_flux = inherited_flux, _reference_dt = reference_dt,
    )
    @test same_step.right_hand_side == ordinary

    # The reference must be finite, positive and no smaller than dt, whether
    # or not an inherited flux is supplied.
    for invalid_reference in (0.0, -dt, NaN, Inf, dt / 2, prevfloat(dt))
        @test_throws ArgumentError CPR._assemble_cmdgfm_system!(
            solver, work, dt, predictor;
            _inherited_cell_flux = inherited_flux, _reference_dt = invalid_reference,
        )
        @test_throws ArgumentError CPR._assemble_cmdgfm_system!(
            solver, work, dt, predictor; _reference_dt = invalid_reference,
        )
        @test_throws ArgumentError project!(
            predictor, CellField(mesh), solver, dt, work;
            _inherited_cell_flux = inherited_flux, _reference_dt = invalid_reference,
        )
        @test_throws ArgumentError project!(
            predictor, CellField(mesh), solver, dt, work;
            _reference_dt = invalid_reference,
        )
    end

    # Check the original arguments, not just their rounded solver-precision
    # conversions: a smaller Float64 reference can round up to a Float32 dt.
    mesh32 = BlockTreeMesh(
        block_size = (4, 4), origin = (0.0f0, 0.0f0),
        extent = (1.0f0, 1.0f0), periodic = (true, true),
    )
    interface32 = InterfaceField(mesh32, (x, y) -> x - 0.273f0; lipschitz = 1)
    solver32 = CMDGFMProjectionSolver(interface32; density = (10.0f0, 1.0f0))
    work32 = CMDGFMProjectionWorkspace(solver32)
    velocity32 = (FaceVelocity(mesh32, Float32), FaceVelocity(mesh32, Float32))
    dt32 = 0.02f0
    smaller_reference = prevfloat(Float64(dt32))
    @test smaller_reference < dt32 && Float32(smaller_reference) == dt32
    @test_throws ArgumentError project!(
        velocity32, CellField(mesh32, Float32), solver32, dt32, work32;
        _inherited_cell_flux = zeros(Float32, ncells(mesh32)),
        _reference_dt = smaller_reference,
    )
    @test_throws ArgumentError project!(
        velocity32, CellField(mesh32, Float32), solver32, dt32, work32;
        _reference_dt = smaller_reference,
    )
end

@testset "CMDGFM shortened projection retains only the mixture target" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.453, y - 0.487) - 0.2637; lipschitz = 1,
    )
    solver = CMDGFMProjectionSolver(interface; density = (10.0, 1.0))
    work = CMDGFMProjectionWorkspace(solver)
    velocity = _pressure_reuse_velocity(mesh)
    CPR._mark_inactive_phase_velocities!(velocity, interface)
    old_data = CPR._projection_face_data(solver, work)
    inherited_flux = zeros(ncells(mesh))
    CPR._cmdgfm_cell_flux!(inherited_flux, old_data, interface, velocity)
    dt, reference_dt, tolerance = 0.002, 0.02, 1.0e-9
    stats = project!(
        velocity, CellField(mesh), solver, dt, work;
        tolerance, max_iterations = 4000, warm_start = false,
        _inherited_cell_flux = inherited_flux, _reference_dt = reference_dt,
    )
    @test stats.converged
    # The retained mixture defect remains visible, while the normal-velocity
    # condition is imposed on the actual projected state.
    @test stats.mixture_volume_defect > tolerance
    @test stats.normal_velocity_defect <= tolerance
    actual_flux = zeros(ncells(mesh))
    CPR._cmdgfm_cell_flux!(actual_flux, work.pressure_cache.face_data, interface, velocity)
    beta = 1 - dt / reference_dt
    volumes = work.pressure_cache.operator.volumes
    @test dt * CPR._projection_strong_maximum(actual_flux .- beta .* inherited_flux, volumes) <= tolerance
    @test stats.mixture_volume_defect ≈
        dt * CPR._projection_strong_maximum(actual_flux, volumes)
    @test !isempty(work.stencils)
    for stencil in work.stencils
        h = CPR._projection_face_spacing(mesh, stencil.face)
        @test dt * abs(CPR._cmdgfm_normal_velocity_row(velocity, stencil)) / h <= tolerance
    end
end

@testset "CMDGFM moving interface enforces the actual short-step jump" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.453, y - 0.487) - 0.2637; lipschitz = 1,
    )
    solver = CMDGFMProjectionSolver(interface; density = (10.0, 1.0))
    work = CMDGFMProjectionWorkspace(solver)
    old_velocity = _pressure_reuse_velocity(mesh)
    CPR._mark_inactive_phase_velocities!(old_velocity, interface)
    initial_data = CPR._projection_face_data(solver, work)
    initial_flux = zeros(ncells(mesh))
    CPR._cmdgfm_cell_flux!(initial_flux, initial_data, interface, old_velocity)

    # Start with an old state satisfying the normal constraint and a nonzero
    # mixture defect, as can occur following an earlier shortened step.
    reference_dt, tolerance = 0.02, 1.0e-9
    old_stats = project!(
        old_velocity, CellField(mesh), solver, reference_dt / 2, work;
        tolerance, max_iterations = 4000, warm_start = false,
        _inherited_cell_flux = initial_flux, _reference_dt = reference_dt,
    )
    @test old_stats.converged
    @test old_stats.normal_velocity_defect <= tolerance
    @test old_stats.mixture_volume_defect > tolerance
    @test !isempty(work.stencils)
    for stencil in work.stencils
        h = CPR._projection_face_spacing(mesh, stencil.face)
        @test (reference_dt / 2) *
            abs(CPR._cmdgfm_normal_velocity_row(old_velocity, stencil)) / h <= tolerance
    end
    saved_flux = zeros(ncells(mesh))
    CPR._cmdgfm_cell_flux!(
        saved_flux, work.pressure_cache.face_data, interface, old_velocity,
    )

    # Extend the old phase velocities as momentum transport does, then move
    # the geometry by one cell so that newly active faces need that extension.
    momentum = MomentumAdvectionWorkspace(interface, old_velocity)
    CPR._prepare_momentum_advection!(momentum, old_velocity)
    predictor = (FaceVelocity(mesh), FaceVelocity(mesh))
    CPR._copy_momentum_velocity!(predictor, momentum)
    initialize!(interface, (x, y) -> hypot(x - 0.573, y - 0.487) - 0.2637)
    reconstruct!(interface)
    @test any(((1, :liquid), (2, :gas))) do pair
        phase, name = pair
        any(eachindex(mesh.faces)) do index
            isnan(old_velocity[phase].interior[index]) &&
                CPR._phase_face_is_active(interface, index, mesh.faces[index], name)
        end
    end
    CPR._mark_inactive_phase_velocities!(predictor, interface)

    dt = 0.002
    before = CPR._assemble_cmdgfm_system!(solver, work, dt, predictor)
    np = length(before.pressure_indices)
    jump_rhs = view(before.right_hand_side, (np + 1):length(before.right_hand_side))
    @test !isempty(jump_rhs)
    @test maximum(
        dt^2 * abs(jump_rhs[index]) /
            CPR._projection_face_spacing(mesh, face)
            for (index, face) in pairs(before.mixed_faces)
    ) > 100tolerance
    stats = project!(
        predictor, CellField(mesh), solver, dt, work;
        tolerance, max_iterations = 4000, warm_start = false,
        _inherited_cell_flux = saved_flux, _reference_dt = reference_dt,
    )
    @test stats.converged
    @test stats.mixture_volume_defect > tolerance
    @test stats.normal_velocity_defect <= tolerance
    actual_flux = zeros(ncells(mesh))
    CPR._cmdgfm_cell_flux!(actual_flux, work.pressure_cache.face_data, interface, predictor)
    volumes = work.pressure_cache.operator.volumes
    target_flux = (1 - dt / reference_dt) .* saved_flux
    @test dt * CPR._projection_strong_maximum(actual_flux .- target_flux, volumes) <= tolerance
    for stencil in work.stencils
        h = CPR._projection_face_spacing(mesh, stencil.face)
        @test dt * abs(CPR._cmdgfm_normal_velocity_row(predictor, stencil)) / h <= tolerance
    end
end

@testset "Standalone two-velocity projection preserves physical pressure" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.453, y - 0.487) - 0.2637; lipschitz = 1,
    )
    flow = FlowSimulation(
        mesh, TwoPhase(
            interface; density = (10.0, 1.0), dynamic_viscosity = (0.0, 0.0),
        );
        velocity_model = TwoVelocityModel(),
        velocity = _pressure_reuse_velocity(mesh),
        pressure = (x, y) -> 2.3 + sinpi(2x) + 0.3cospi(2y),
    )
    old_pressure = copy(flow.pressure)
    stats = project!(flow; dt = 0.02, tolerance = 1.0e-9, max_cycles = 4000)
    @test stats.initial_residual > 0
    @test stats.converged
    @test stats.mixture_volume_defect <= 1.0e-9
    @test stats.normal_velocity_defect <= 1.0e-9
    for block in eachindex(flow.pressure.blocks)
        @test flow.pressure.blocks[block] == old_pressure.blocks[block]
    end
end
