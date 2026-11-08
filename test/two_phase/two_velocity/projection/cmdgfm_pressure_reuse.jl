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
end

@testset "lazy two-velocity workspace types across AMR" begin
    for T in (Float32, Float64),
            projection in (CMDGFMProjection(), MimeticProjection()),
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
            velocity_model = TwoVelocityModel(
                interface_transport = transport, pressure_projection = projection,
            ),
            velocity = _pressure_reuse_velocity(mesh),
        )
        phase = flow.workspace.two_phase
        @test (@inferred _pressure_model_interface(flow.solver.two_phase)) === interface
        @test isnothing(phase.two_velocity[])
        @test all(isconcretetype, Base.uniontypes(eltype(typeof(phase.two_velocity))))
        state = CPR._two_velocity_flow_workspace!(phase, flow.solver.two_phase)
        @test typeof(state) <: eltype(typeof(phase.two_velocity))
        @test (@inferred CPR._two_velocity_flow_workspace!(phase, flow.solver.two_phase)) === state
        projection_workspace = state.projection_workspace

        adaptation = isnothing(geometry) ?
            AMRWorkspace(mesh, interface, flow.velocity, flow.pressure) :
            AMRWorkspace(mesh, interface, geometry, flow.velocity, flow.pressure)
        labels = [block.key.i == 0 ? :refine : :keep for block in mesh.blocks]
        @test adapt!(adaptation, labels; step = 1, force = true).changed
        CPR._rebuild_topology_dependent_state!(flow.solver, flow.workspace; dt = T(0.001))
        phase = flow.workspace.two_phase
        @test all(isconcretetype, Base.uniontypes(eltype(typeof(phase.two_velocity))))
        @test phase.two_velocity[].projection_workspace === projection_workspace
        state = CPR._two_velocity_flow_workspace!(phase, flow.solver.two_phase)
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
    boundary, = CPR._mimetic_boundary_constraints(solver.interface, velocity)
    rhs = vcat((transpose(Ip) * mixture .- boundary[indices]) ./ T(dt), rhs_jump)
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
    @test fieldtype(typeof(flow.solver.two_phase), :interface) === typeof(interface)
    @test fieldtype(typeof(phase), :advection) === typeof(phase.advection)
    @test fieldtype(typeof(phase), :momentum) === typeof(phase.momentum)
    @test all(isconcretetype, Base.uniontypes(eltype(typeof(phase.two_velocity))))
    @test project!(flow).converged
    original_work = flow.workspace.two_phase.two_velocity[].projection_workspace
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
        CPR._rebuild_topology_dependent_state!(flow.solver, flow.workspace; dt = 0.001)
        phase = flow.workspace.two_phase
        @test all(isconcretetype, Base.uniontypes(eltype(typeof(phase.two_velocity))))
        state = flow.workspace.two_phase.two_velocity[]
        @test typeof(state) <: eltype(typeof(phase.two_velocity))
        @test state.projection_workspace === original_work
        @test (state.momentum.x.workspace.schedule isa CPR.UniformDualAdvectionSchedule) == !refine_left
        @test project!(flow).converged
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
        @test work.pressure_cache.operator === flow.solver.pressure_solver.operator
        @test (pc.multigrid === flow.solver.pressure_solver.preconditioner) == (coarse_size == 64)
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
        # Reuse pressure through interface movement and a change in the jump
        # unknown set. Both paths always receive identical fresh predictors.
        for shift in (0.0, 0.029)
            pure || initialize!(
                interface,
                (x, y) -> hypot(x - 0.453 - shift, y - 0.487) - 0.2637
            )
            cold_velocity = _pressure_reuse_velocity(mesh)
            warm_velocity = _pressure_reuse_velocity(mesh)
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
    # With no jump unknowns, the previous pressure alone is a candidate.
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
    assembled = CPR._assemble_cmdgfm_system!(
        solver, work, 0.02,
        _pressure_reuse_velocity(mesh)
    )
    pc = CPR._cmdgfm_preconditioner(solver, assembled, work)
    fill!(work.solution, 0.0)
    @test CPR._cmdgfm_initial_solution!(work, assembled, p, 0.02, pc)
    @test norm(work.correction_rhs) < 1.0e-8 * norm(assembled.right_hand_side)
    for invalid_guess in (NaN, 1.0e30)
        # An invalid or grossly inaccurate *pressure guess* must not enter
        # the solve. This does not supply invalid active phase velocities.
        for block in eachindex(mesh.blocks)
            values = interior(p, block)
            values .= invalid_guess .* reshape(1:length(values), size(values))
        end
        fill!(work.solution, 0.0)
        @test !CPR._cmdgfm_initial_solution!(work, assembled, p, 0.02, pc)
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
    @test !CPR._cmdgfm_initial_solution!(work, assembled, p, 0.02, nothing)
    @test all(iszero, work.solution)
    @test work.correction_rhs == assembled.right_hand_side
end

@testset "Mimetic cache follows interface and solid geometry" begin
    for variant in (:joint_smoothed, :joint_face), cut in (false, true)
        mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, false))
        geometry = cut ? GeometryField(
                mesh, (x, y) -> y - 0.09 - 0.04x;
                lipschitz = hypot(1, 0.04)
            ) : nothing
        interface = InterfaceField(
            mesh,
            (x, y) -> hypot(x - 0.453, y - 0.487) - 0.2637;
            geometry, lipschitz = 1
        )
        solver = MimeticProjectionSolver(interface; density = (10.0, 1.0), variant)
        work = MimeticProjectionWorkspace(solver)
        cache = work.cache
        for shift in (0.0, 0.037, 0.13, 0.0)
            initialize!(
                interface,
                (x, y) -> hypot(x - 0.453 - shift, y - 0.487) - 0.2637
            )
            velocity = _pressure_reuse_velocity(mesh)
            CPR._mark_inactive_phase_velocities!(velocity, interface)
            actual = CPR._assemble_mimetic_system!(solver, work, 0.02, velocity)
            fresh_work = MimeticProjectionWorkspace(solver)
            fresh = CPR._assemble_mimetic_system!(solver, fresh_work, 0.02, velocity)
            @test actual.pressure_indices == fresh.pressure_indices
            @test actual.active_faces == fresh.active_faces
            @test actual.inactive_faces == fresh.inactive_faces
            # Coefficient refresh changes floating-point evaluation only at
            # roundoff level compared with constructing a new operator.
            @test actual.matrix ≈ fresh.matrix rtol = 2.0e-12 atol = 2.0e-12
            @test actual.candidate_constraint ≈ fresh.candidate_constraint rtol = 2.0e-12 atol = 2.0e-12
            @test actual.candidate_scales ≈ fresh.candidate_scales rtol = 2.0e-12 atol = 2.0e-12
            @test actual.right_hand_side ≈ fresh.right_hand_side rtol = 2.0e-12 atol = 2.0e-12
            operator = cache.pressure.operator
            incidence = cache.pressure.incidence
            pressure_slice = cache.incidence_pressure
            active_slice = cache.active_incidence
            CPR._assemble_mimetic_system!(solver, work, 0.02, velocity)
            @test cache.pressure.operator === operator
            @test cache.pressure.incidence === incidence
            @test cache.incidence_pressure === pressure_slice
            @test cache.active_incidence === active_slice
        end
        if cut
            operator = cache.pressure.operator
            old_revision = cache.pressure.geometry_revision
            CPR.refresh_geometry!(geometry)
            velocity = _pressure_reuse_velocity(mesh)
            CPR._mark_inactive_phase_velocities!(velocity, interface)
            actual = CPR._assemble_mimetic_system!(solver, work, 0.02, velocity)
            fresh = CPR._assemble_mimetic_system!(
                solver, MimeticProjectionWorkspace(solver), 0.02, velocity
            )
            @test cache.pressure.geometry_revision != old_revision
            @test cache.pressure.operator !== operator
            @test actual.matrix ≈ fresh.matrix rtol = 2.0e-12 atol = 2.0e-12
        end
    end
end

@testset "Mimetic MINRES workspace is reused and resized" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    interface = InterfaceField(
        mesh,
        (x, y) -> hypot(x - 0.453, y - 0.487) - 0.2637; lipschitz = 1
    )
    solver = MimeticProjectionSolver(interface; density = (10.0, 1.0))
    work = MimeticProjectionWorkspace(solver)
    velocity = _pressure_reuse_velocity(mesh)
    stats = project!(
        velocity, CellField(mesh), solver, 0.02, work;
        tolerance = 1.0e-8, max_iterations = 5000, check_symmetry = true
    )
    @test stats.converged
    @test stats.matrix_symmetry_error == 0
    krylov = work.cache.krylov
    @test !isnothing(krylov)
    stats = project!(
        _pressure_reuse_velocity(mesh), CellField(mesh), solver, 0.02, work;
        tolerance = 1.0e-8, max_iterations = 5000
    )
    @test stats.converged
    @test isnothing(stats.matrix_symmetry_error)
    @test work.cache.krylov === krylov
    # Exercise the AMR/active-set resizing path independently of solver
    # convergence and verify that its preconditioned optional vector follows.
    matrix = spdiagm(0 => ones(7))
    @test CPR._mimetic_krylov_workspace!(work.cache, matrix, ones(7)) === krylov
    @test krylov.m == krylov.n == 7
    @test length(krylov.x) == length(krylov.r1) == length(krylov.y) == 7
    @test isempty(krylov.v) || length(krylov.v) == 7
end

@testset "Mimetic workspace survives AMR topology changes" begin
    for variant in (:joint_smoothed, :joint_face)
        mesh = BlockTreeMesh(
            base_mesh = (8, 4), block_size = (4, 4),
            max_level = 1, periodic = (true, true)
        )
        interface = InterfaceField(mesh, (x, y) -> x - 0.273; lipschitz = 1)
        flow = FlowSimulation(
            mesh,
            TwoPhase(interface; density = (10.0, 1.0), dynamic_viscosity = (0.0, 0.0));
            velocity_model = TwoVelocityModel(
                pressure_projection = MimeticProjection(; variant)
            ),
            velocity = _pressure_reuse_velocity(mesh)
        )
        @test project!(flow; tolerance = 1.0e-8, max_cycles = 4000).converged
        work = flow.workspace.two_phase.two_velocity[].projection_workspace
        krylov = work.cache.krylov
        @test !isnothing(krylov)
        adaptation = AMRWorkspace(mesh, interface, flow.velocity, flow.pressure)
        labels = [block.key.i == 0 ? :refine : :keep for block in mesh.blocks]
        @test adapt!(adaptation, labels; step = 1, force = true).changed
        CPR._rebuild_topology_dependent_state!(flow.solver, flow.workspace; dt = 0.001)
        @test flow.workspace.two_phase.two_velocity[].projection_workspace === work
        @test project!(flow; tolerance = 1.0e-8, max_cycles = 4000).converged
        @test work.cache.krylov === krylov
        @test CPR._cmdgfm_same_topology(work.cache.pressure.keys, mesh)
        @test work.cache.pressure.incidence == _reference_projection_incidence(mesh, Float64)
    end
end

@testset "Mimetic cache recovers from an interrupted rebuild" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, true))
    levelset(x, y) = hypot(x - 0.453, y - 0.487) - 0.2637
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    solver = MimeticProjectionSolver(interface; density = (10.0, 1.0))
    work = MimeticProjectionWorkspace(solver)
    reconstruct!(interface)
    # Fail after the shared pressure cache has started its topology rebuild,
    # then repair the state and ensure no partially built slice is reused.
    for block in eachindex(interface.conditional_fraction.blocks)
        fill!(interior(interface.conditional_fraction, block), NaN)
    end
    @test_throws ArgumentError CPR._mimetic_pressure_cache!(work.cache, solver, nothing)
    @test !work.cache.initialized
    reconstruct!(interface; force = true)
    velocity = _pressure_reuse_velocity(mesh)
    CPR._mark_inactive_phase_velocities!(velocity, interface)
    recovered = CPR._assemble_mimetic_system!(solver, work, 0.02, velocity)
    fresh = CPR._assemble_mimetic_system!(
        solver, MimeticProjectionWorkspace(solver), 0.02, velocity
    )
    @test work.cache.initialized
    @test recovered.pressure_indices == fresh.pressure_indices
    @test recovered.active_faces == fresh.active_faces
    @test recovered.matrix ≈ fresh.matrix rtol = 2.0e-12 atol = 2.0e-12
end

@testset "Mimetic cache distinguishes solid geometry identities" begin
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, false))
    first_geometry = GeometryField(mesh, (x, y) -> y - 0.09; lipschitz = 1)
    second_geometry = GeometryField(mesh, (x, y) -> y - 0.14; lipschitz = 1)
    @test first_geometry.revision == second_geometry.revision
    levelset(x, y) = hypot(x - 0.453, y - 0.487) - 0.2637
    first_interface = InterfaceField(mesh, levelset; geometry = first_geometry, lipschitz = 1)
    second_interface = InterfaceField(mesh, levelset; geometry = second_geometry, lipschitz = 1)
    first_solver = MimeticProjectionSolver(first_interface; density = (10.0, 1.0))
    second_solver = MimeticProjectionSolver(second_interface; density = (10.0, 1.0))
    work = MimeticProjectionWorkspace(first_solver)
    first_velocity = _pressure_reuse_velocity(mesh)
    CPR._mark_inactive_phase_velocities!(first_velocity, first_interface)
    CPR._assemble_mimetic_system!(first_solver, work, 0.02, first_velocity)
    first_operator = work.cache.pressure.operator
    second_velocity = _pressure_reuse_velocity(mesh)
    CPR._mark_inactive_phase_velocities!(second_velocity, second_interface)
    actual = CPR._assemble_mimetic_system!(second_solver, work, 0.02, second_velocity)
    fresh = CPR._assemble_mimetic_system!(
        second_solver, MimeticProjectionWorkspace(second_solver), 0.02, second_velocity
    )
    @test work.cache.geometry === second_geometry
    @test work.cache.pressure.operator !== first_operator
    @test actual.pressure_indices == fresh.pressure_indices
    @test actual.active_faces == fresh.active_faces
    @test actual.matrix ≈ fresh.matrix rtol = 2.0e-12 atol = 2.0e-12
end

@testset "Mimetic cache refreshes the embedded mask when the fraction floor changes" begin
    for variant in (:joint_smoothed, :joint_face)
        mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, false))
        geometry = GeometryField(mesh, (x, y) -> y - 0.14; lipschitz = 1)
        levelset(x, y) = hypot(x - 0.453, y - 0.487) - 0.2637
        first_interface = InterfaceField(mesh, levelset; geometry, lipschitz = 1)
        second_interface = InterfaceField(
            mesh, levelset; geometry, lipschitz = 1, fraction_floor = 0.2,
        )
        first_solver = MimeticProjectionSolver(first_interface; density = (10.0, 1.0), variant)
        second_solver = MimeticProjectionSolver(second_interface; density = (10.0, 1.0), variant)
        work = MimeticProjectionWorkspace(first_solver)
        first_velocity = _pressure_reuse_velocity(mesh)
        CPR._mark_inactive_phase_velocities!(first_velocity, first_interface)
        CPR._assemble_mimetic_system!(first_solver, work, 0.02, first_velocity)
        previous_mask = copy(work.cache.embedded_cells)
        incidence = work.cache.pressure.incidence
        second_velocity = _pressure_reuse_velocity(mesh)
        CPR._mark_inactive_phase_velocities!(second_velocity, second_interface)
        actual = CPR._assemble_mimetic_system!(second_solver, work, 0.02, second_velocity)
        fresh_work = MimeticProjectionWorkspace(second_solver)
        fresh = CPR._assemble_mimetic_system!(second_solver, fresh_work, 0.02, second_velocity)
        @test work.cache.embedded_cells != previous_mask
        @test work.cache.embedded_cells == fresh_work.cache.embedded_cells
        @test work.cache.pressure.incidence === incidence
        @test actual.active_faces == fresh.active_faces
        @test actual.matrix ≈ fresh.matrix rtol = 2.0e-12 atol = 2.0e-12
    end
end

@testset "Mimetic cache rebuilds on cut-cell AMR" begin
    mesh = BlockTreeMesh(
        base_mesh = (8, 4), block_size = (4, 4),
        max_level = 1, periodic = (true, true)
    )
    geometry = GeometryField(
        mesh, (x, y) -> y - 0.08 - 0.03x;
        lipschitz = hypot(1, 0.03)
    )
    interface = InterfaceField(mesh, (x, y) -> x - 0.273; geometry, lipschitz = 1)
    flow = FlowSimulation(
        mesh,
        TwoPhase(interface; density = (10.0, 1.0), dynamic_viscosity = (0.0, 0.0));
        velocity_model = TwoVelocityModel(pressure_projection = MimeticProjection()),
        velocity = _pressure_reuse_velocity(mesh)
    )
    state = CPR._two_velocity_flow_workspace!(flow.workspace.two_phase, flow.solver.two_phase)
    work, solver = state.projection_workspace, state.projection_solver
    velocity = _pressure_reuse_velocity(mesh)
    CPR._mark_inactive_phase_velocities!(velocity, interface)
    CPR._assemble_mimetic_system!(solver, work, 0.02, velocity)
    previous_operator = work.cache.pressure.operator
    adaptation = AMRWorkspace(mesh, interface, geometry, flow.velocity, flow.pressure)
    labels = [block.key.i == 0 ? :refine : :keep for block in mesh.blocks]
    @test adapt!(adaptation, labels; step = 1, force = true).changed
    CPR._rebuild_topology_dependent_state!(flow.solver, flow.workspace; dt = 0.001)
    @test flow.workspace.two_phase.two_velocity[].projection_workspace === work
    velocity = _pressure_reuse_velocity(mesh)
    CPR._mark_inactive_phase_velocities!(velocity, interface)
    actual = CPR._assemble_mimetic_system!(solver, work, 0.02, velocity)
    fresh = CPR._assemble_mimetic_system!(
        solver, MimeticProjectionWorkspace(solver), 0.02, velocity
    )
    @test work.cache.pressure.operator !== previous_operator
    @test work.cache.pressure.geometry_revision == geometry.revision
    @test actual.pressure_indices == fresh.pressure_indices
    @test actual.active_faces == fresh.active_faces
    @test actual.matrix ≈ fresh.matrix rtol = 2.0e-12 atol = 2.0e-12
end
