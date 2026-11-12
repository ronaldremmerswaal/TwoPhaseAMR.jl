using LinearAlgebra: diag
using SparseArrays: sparse
using TwoPhaseAMR: @_parallel, _parallel_for, _parallel_block_columns, _PARALLEL_GRAIN

# Run `f` with every loop above `grain` work units threaded.
function with_parallel_grain(f, grain)
    previous = _PARALLEL_GRAIN[]
    _PARALLEL_GRAIN[] = grain
    try
        return f()
    finally
        _PARALLEL_GRAIN[] = previous
    end
end

# Grain high enough that every loop takes its serial branch.
const SERIAL_GRAIN = typemax(Int) ÷ 4

# Exception `f` throws and its message, or `nothing`. A Polyester worker's
# exception would arrive wrapped as a `TaskFailedException`, so callers assert
# the type as well as the text; a diagnostic that names the offending cell,
# block or face is part of the documented behaviour.
function thrown_exception(f)
    try
        f()
    catch err
        return (err, sprint(showerror, err))
    end
    return nothing
end

# Message of a failure returned by `thrown_exception`, or "" when `f` threw
# nothing.
message_of(failure) = isnothing(failure) ? "" : failure[2]

# Two-level jump of an adaptive workspace whose only field is a cell field:
# the old buffers are staged and the mesh is refined twice without a transfer.
function two_level_jump_workspace()
    mesh = BlockTreeMesh(block_size = (8, 8), max_level = 3, periodic = (true, true))
    field = CellField(mesh, (x, y) -> x)
    workspace = AMRWorkspace(mesh, field)
    buffer = only(workspace.field_buffers)
    TwoPhaseAMR._copy_geometry_snapshot!(workspace)
    TwoPhaseAMR._stage_field!(buffer, workspace)
    for step in (1, 2)
        TwoPhaseAMR._collect_active_nodes!(workspace)
        TwoPhaseAMR._apply_refinement_actions!(
            workspace, fill(:refine, length(workspace.leaf_nodes)), step,
        )
    end
    TwoPhaseAMR._collect_active_nodes!(workspace)
    TwoPhaseAMR._build_face_transfer_geometry!(workspace)
    TwoPhaseAMR._rebuild_mesh_schedules!(workspace)
    return workspace, buffer, field
end

function parallel_loop_features(values)
    @_parallel for index in eachindex(values)
        iseven(index) && continue
        total = 0
        for term in 1:index
            term > 5 && break
            isodd(term) || continue
            total += term
        end
        values[index] = total
    end
    return values
end

function scaled_copy!(destination, source)
    @_parallel for index in eachindex(destination)
        @inbounds destination[index] = 2source[index]
    end
    return destination
end

expand_parallel(loop) = macroexpand(TwoPhaseAMR, :(@_parallel $loop))

@testset "parallel loop launch" begin
    @test _PARALLEL_GRAIN[] == 1024
    expected = [isodd(index) ? sum(filter(isodd, 1:min(index, 5))) : -1 for index in 1:37]
    for grain in (1, _PARALLEL_GRAIN[])
        @test with_parallel_grain(() -> parallel_loop_features(fill(-1, 37)), grain) == expected
    end

    @test_throws "cannot break" expand_parallel(
        :(
            for index in 1:3
                break
            end
        )
    )
    @test_throws "cannot return" expand_parallel(
        :(
            for index in 1:3
                for inner in 1:2
                    return inner
                end
            end
        )
    )
    @test_throws "only `work = expression`" macroexpand(
        TwoPhaseAMR, Meta.parse("@_parallel grain = 4 for index in 1:3\nend"),
    )

    visited = zeros(Int, 100)
    with_parallel_grain(1) do
        _parallel_for(length(visited); work = 3) do index
            visited[index] += index
        end
    end
    @test visited == 1:100
    @test_throws ArgumentError _parallel_for(identity, -1)

    # The loop form launches its body without a closure argument, so a warm
    # launch allocates nothing on either path.
    destination, source = zeros(4096), rand(4096)
    for grain in (1, _PARALLEL_GRAIN[])
        with_parallel_grain(grain) do
            scaled_copy!(destination, source)
            @test @allocated(scaled_copy!(destination, source)) == 0
        end
    end
    @test destination == 2source

    for mesh in (BlockTreeMesh(block_size = (8, 6)), BlockTreeMesh(base_mesh = (32, 16), block_size = (8, 8)))
        nx, ny = mesh.block_size
        owners = zeros(Int, nx * ny * nblocks(mesh))
        with_parallel_grain(1) do
            _parallel_block_columns(mesh) do block, first_j, last_j
                for j in first_j:last_j, i in 1:nx
                    owners[(block - 1) * nx * ny + (j - 1) * nx + i] += 1
                end
            end
        end
        @test all(==(1), owners)
    end
end

@testset "parallel failures and user callbacks" begin
    # Data-dependent checks are fused into the threaded loops and recorded in
    # owned flags; the caller throws after the launch, so the failure is the
    # package's own exception rather than a Polyester worker's
    # `TaskFailedException`.
    mesh = BlockTreeMesh(base_mesh = (32, 32), block_size = (8, 8), periodic = (false, false))
    interface = InterfaceField(mesh, (x, y) -> y - 0.5; lipschitz = 1)
    interface.fraction.blocks[end][3, 3] = 2.0
    failure = with_parallel_grain(1) do
        thrown_exception() do
            TwoPhaseAMR._refresh_conditional_fraction!(interface)
        end
    end
    @test isa(failure, Tuple{ArgumentError, String})
    @test occursin("liquid volume fraction", message_of(failure))
    @test occursin("CellRef(", message_of(failure))

    active = trues(ncells(mesh))
    volumes = TwoPhaseAMR._cell_volumes(mesh, Float64)
    operator = EllipticOperator(
        mesh; face_coefficient = ones(length(mesh.faces)), reaction = 0.75,
        active_cells = active, volumes, inactive_condition = :dirichlet,
    )
    constraints = copy(operator.constraints)
    operator_volumes = copy(operator.volumes)
    inconsistent = copy(volumes)
    inconsistent[end] = 0.0
    failure = with_parallel_grain(1) do
        thrown_exception() do
            TwoPhaseAMR._update_elliptic_activity!(operator, active, inconsistent)
        end
    end
    @test isa(failure, Tuple{ArgumentError, String})
    @test occursin("positive volume", message_of(failure))
    @test occursin("CellRef(", message_of(failure))
    @test operator.constraints == constraints
    @test operator.volumes == operator_volumes

    # Initialization stays transactional now that the level set runs serially.
    fraction = copy.(interface.fraction.blocks)
    failing_levelset(x, y) = x > 0.7 ? error("level set failed") : y - 0.5
    @test_throws "level set failed" initialize!(interface, failing_levelset; lipschitz = 1)
    @test interface.fraction.blocks == fraction

    # User callbacks run serially on the calling task, so a callback that is
    # not thread-safe (here: push! to a shared vector) is still correct.
    calls = Int[]
    field = CellField(mesh)
    velocity = FaceVelocity(mesh)
    with_parallel_grain(1) do
        initialize!(field, (x, y) -> (push!(calls, Threads.threadid()); x))
        set_velocities!(velocity, (x, y, time, axis) -> (push!(calls, Threads.threadid()); y))
        fill_halo!(field, Dirichlet((x, y, time) -> (push!(calls, Threads.threadid()); 0.0)))
        @test_throws ArgumentError PhaseScalar(interface; liquid = (x, y) -> x > 0.9 ? NaN : 1.0)
    end
    physical_halos = count(entry -> entry.physical_boundary, mesh.halo_entries)
    @test physical_halos > 0
    @test length(calls) ==
        ncells(mesh) + length(mesh.faces) + length(mesh.boundary_faces) + physical_halos
    @test all(==(Threads.threadid()), calls)
end

@testset "recorded failures in threaded kernels" begin
    # A check whose outcome depends on the data is fused into the threaded
    # loop: the body records the failure in an owned, preallocated flag and the
    # caller throws after the launch, so the error is the package's own and the
    # diagnostic names the offending cell, block or face.

    # WY conditional-fraction bounds, checked in the increment sweep.
    mesh = BlockTreeMesh(base_mesh = (16, 16), block_size = (8, 8), periodic = (true, true))
    interface = InterfaceField(
        mesh, (x, y) -> 0.25 + 0.1sinpi(2x) * sinpi(2y); lipschitz = 1,
    )
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> axis == 1 ? 1.0 : 0.5)
    advection = WYAdvectionWorkspace(interface)
    dt = stable_timestep(velocity, advection)
    interface.conditional_fraction.blocks[2][3, 3] = 5.0
    failure = with_parallel_grain(1) do
        thrown_exception() do
            advect!(
                interface, velocity, dt, advection;
                check_timestep = false, initial_geometry_prepared = true,
            )
        end
    end
    @test isa(failure, Tuple{ErrorException, String})
    @test occursin("WY conditional fraction out of bounds", message_of(failure))
    @test occursin("CellRef(2, 2, 2)", message_of(failure))

    # AMR cell transfer rejects a two-level jump before the buffers are taken
    # apart, so the field is intact.
    adaptation, buffer, field = two_level_jump_workspace()
    failure = with_parallel_grain(1) do
        thrown_exception() do
            TwoPhaseAMR._transfer_cell_field!(buffer, adaptation)
        end
    end
    @test isa(failure, Tuple{ArgumentError, String})
    @test occursin("more than one level", message_of(failure))
    @test occursin("BlockKey", message_of(failure))
    @test isempty(field.blocks)

    # AMR face transfer records the faces without old data and throws after
    # the launch, naming the first one.
    face_mesh = BlockTreeMesh(block_size = (8, 8), max_level = 2, periodic = (true, true))
    face_velocity = FaceVelocity(face_mesh)
    face_adaptation = AMRWorkspace(face_mesh, face_velocity)
    face_buffer = only(face_adaptation.field_buffers)
    TwoPhaseAMR._copy_geometry_snapshot!(face_adaptation)
    TwoPhaseAMR._stage_field!(face_buffer, face_adaptation)
    TwoPhaseAMR._build_face_transfer_geometry!(face_adaptation)
    TwoPhaseAMR._rebuild_mesh_schedules!(face_adaptation)
    empty!(face_adaptation.old_faces)
    empty!(face_adaptation.old_boundary_faces)
    TwoPhaseAMR._build_face_transfer_geometry!(face_adaptation)
    failure = with_parallel_grain(1) do
        thrown_exception() do
            TwoPhaseAMR._transfer_face_field!(face_buffer, face_adaptation)
        end
    end
    @test isa(failure, Tuple{ArgumentError, String})
    @test occursin(
        "could not locate old face data for a new face: face ",
        message_of(failure),
    )

    # The reaction update uses the diagonal slots located at assembly, so a
    # warm refresh still assembles `diffusion*stiffness + reaction*volumes`.
    reaction_mesh = BlockTreeMesh(base_mesh = (16, 16), block_size = (8, 8), periodic = (true, true))
    reaction_volumes = TwoPhaseAMR._cell_volumes(reaction_mesh, Float64)
    operator = EllipticOperator(
        reaction_mesh; face_coefficient = ones(length(reaction_mesh.faces)),
        reaction = 0.75, active_cells = trues(ncells(reaction_mesh)),
        volumes = reaction_volumes,
    )
    assembled = Matrix(operator.matrix)
    with_parallel_grain(1) do
        TwoPhaseAMR._update_elliptic_face_coefficient!(
            operator, copy(operator.face_coefficient),
        )
    end
    @test Matrix(operator.matrix) == assembled
    @test diag(operator.matrix) ==
        operator.diffusion .* diag(operator.stiffness) .+
        operator.reaction .* reaction_volumes

    # A stiffness pattern without a diagonal entry is rejected when the matrix
    # is built, naming the column, instead of in the per-step refresh.
    without_diagonal = (
        volumes = [1.0, 1.0],
        stiffness = sparse([1, 2], [1, 1], [2.0, -1.0], 2, 2),
    )
    failure = thrown_exception() do
        TwoPhaseAMR._variable_diffusion_matrix(without_diagonal, [1.0, 1.0], 0.5)
    end
    @test isa(failure, Tuple{ErrorException, String})
    @test occursin("no diagonal at index 2", message_of(failure))

    # `fill_halo!` with a `Dirichlet` callback gives exactly the serially
    # filled halos, and every physical halo holds the callback value.
    halo_mesh = BlockTreeMesh(base_mesh = (32, 32), block_size = (8, 8), periodic = (false, false))
    threaded_field = CellField(halo_mesh)
    serial_field = CellField(halo_mesh)
    initialize!(threaded_field, (x, y) -> x + 2y)
    initialize!(serial_field, (x, y) -> x + 2y)
    boundary = Dirichlet((x, y, time) -> x - y + time)
    with_parallel_grain(1) do
        fill_halo!(threaded_field, boundary; time = 0.5)
    end
    with_parallel_grain(SERIAL_GRAIN) do
        fill_halo!(serial_field, boundary; time = 0.5)
    end
    @test threaded_field.blocks == serial_field.blocks
    for entry_index in TwoPhaseAMR._physical_halo_entries(halo_mesh)
        entry = halo_mesh.halo_entries[entry_index]
        @test TwoPhaseAMR._get(threaded_field, entry.destination) ==
            entry.center[1] - entry.center[2] + 0.5
    end
end

@testset "threaded batch exceptions" begin
    # Polyester only hands a batch to a worker thread when there are two or
    # more, so with one thread the recorded-failure tests above exercise the
    # serial expansion of the same body. State that instead of passing
    # silently, and check the recorded-failure contract on the threaded path.
    if Threads.nthreads() < 2
        @test_skip Threads.nthreads() >= 2
    else
        values = zeros(8)
        flags = falses(8)
        function record_failure!(values, flags)
            fill!(flags, false)
            with_parallel_grain(1) do
                @_parallel for index in eachindex(values)
                    values[index] = index
                    index == 8 && (flags[index] = true)
                end
            end
            index = findfirst(flags)
            isnothing(index) && return nothing
            throw(ArgumentError("recorded failure at index $index"))
        end
        failure = thrown_exception() do
            record_failure!(values, flags)
        end
        @test isa(failure, Tuple{ArgumentError, String})
        @test occursin("recorded failure at index 8", message_of(failure))
        @test values == 1:8
    end
end

@testset "shared-memory kernel scheduling" begin
    mesh = BlockTreeMesh(block_size = (64, 64))
    field = CellField(mesh)
    initialize!(field, (x, y) -> x + 2y)
    field_sum = integrate(field)
    @test field_sum ≈ 1.5 atol = 32eps()

    scalar = CellField(mesh, (x, y) -> 1 + 0.1sinpi(2x) * cospi(2y))
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> axis == 1 ?
            0.1sinpi(2x)^2 * sinpi(2y) :
            -0.1sinpi(2x) * sinpi(2y)^2,
    )
    workspace = AdvectionWorkspace(scalar)
    initial_mass = integrate(scalar)
    advect!(
        scalar, velocity,
        stable_timestep(velocity, workspace; cfl = 0.2), workspace,
    )
    @test integrate(scalar) ≈ initial_mass atol = 64eps()
end
