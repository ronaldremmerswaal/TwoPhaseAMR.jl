using LinearAlgebra: diag, mul!
using SparseArrays: sparse
using TwoPhaseAMR: @_parallel, _parallel_for, _parallel_block_columns, _PARALLEL_GRAIN
using SparseArrays: SparseMatrixCSC, nnz, nonzeros, nzrange, rowvals, sprand, sprandn, spzeros

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

# The strong residual maximum over unit volumes, with and without partials.
threaded_maximum(values, partials) =
    TwoPhaseAMR._strong_residual_maximum(values, ones(length(values)), partials)

# The former serial scatter form of the factorized roundoff floor.
function scatter_roundoff_floor(action, solution, right_hand_side, volumes)
    operator = action.operator
    T = eltype(solution)
    nx, ny = operator.mesh.block_size
    square_root = operator.square_root_face_coefficient
    face_input = similar(action.face_input)
    for (face_index, face) in pairs(operator.mesh.faces)
        minus = TwoPhaseAMR._cell_index(face.minus, nx, ny)
        plus = TwoPhaseAMR._cell_index(face.plus, nx, ny)
        face_input[face_index] = square_root[face_index] * face.area *
            (abs(solution[minus]) + abs(solution[plus]))
    end
    metric = operator.gradient_operator.inverse_metric
    face_output = zeros(T, length(face_input))
    if metric isa TwoPhaseAMR.DiagonalInverseFaceMetric
        face_output .= abs.(metric.diagonal) .* face_input
    else
        rows = TwoPhaseAMR.rowvals(metric.matrix)
        values = nonzeros(metric.matrix)
        for column in axes(metric.matrix, 2), pointer in TwoPhaseAMR.nzrange(metric.matrix, column)
            face_output[rows[pointer]] += abs(values[pointer]) * face_input[column]
        end
    end
    temporary = (
        abs(operator.reaction) .* volumes .+
            abs(operator.diffusion) .* operator.boundary_diagonal .+ operator.constraints
    ) .* abs.(solution)
    for (face_index, face) in pairs(operator.mesh.faces)
        bound = abs(operator.diffusion) * face.area *
            square_root[face_index] * face_output[face_index]
        temporary[TwoPhaseAMR._cell_index(face.minus, nx, ny)] += bound
        temporary[TwoPhaseAMR._cell_index(face.plus, nx, ny)] += bound
    end
    scale = one(T)
    for index in eachindex(temporary)
        volumes[index] > zero(T) || continue
        scale = max(scale, temporary[index] / volumes[index], abs(right_hand_side[index] / volumes[index]))
    end
    return T(128) * eps(T) * scale, temporary
end

# Grain 1 threads every loop; `typemax(Int)` runs every loop serially.
const BOTH_PATHS = (1, typemax(Int))

@testset "threaded reductions and transfers are bitwise serial" begin
    # Fixed-chunk maximum: the serial maximum, and canonical NaN bits
    # whichever NaN terms occur, with partials (threaded) and without.
    values = randn(5000)
    values[1:7:end] .= -0.0
    withnan = copy(values)
    withnan[3777] = -NaN
    withnan[11] = reinterpret(Float64, reinterpret(UInt64, NaN) | 0x01)
    partials = TwoPhaseAMR._reduction_partials!(Float64[], 5000)
    @test length(partials) == 5
    for grain in BOTH_PATHS, scratch in (partials, nothing)
        @test with_parallel_grain(() -> threaded_maximum(values, scratch), grain) === maximum(abs, values)
        @test with_parallel_grain(() -> threaded_maximum(withnan, scratch), grain) === NaN
        @test with_parallel_grain(() -> threaded_maximum(fill(-0.0, 3000), scratch), grain) === 0.0
    end
    # Undersized reduction scratch is rejected, not written past its end.
    failure = thrown_exception(() -> threaded_maximum(values, Float64[]))
    @test isa(failure, Tuple{ArgumentError, String})
    @test occursin("reduction scratch has 0 entries", message_of(failure))

    # Fixed-chunk slot sums: one chunk is the index-order sum; more chunks
    # are summed per chunk and folded in chunk order, at every grain.
    for items in (1000, 5000)
        slots = rand(0:3, items)
        terms, weights = randn(items), rand(items)
        expected = zeros(3)
        for chunk in 1:cld(items, 1024)
            partial = zeros(3)
            for item in ((chunk - 1) * 1024 + 1):min(items, chunk * 1024)
                iszero(slots[item]) || (partial[slots[item]] += weights[item] * terms[item])
            end
            expected .+= partial
        end
        for grain in BOTH_PATHS
            sums = with_parallel_grain(grain) do
                TwoPhaseAMR._chunked_slot_sums!(zeros(3), Float64[], slots, terms, weights, Val(:weighted))
            end
            @test sums == expected
        end
    end

    # Three slots: the padded chunk stride is layout only, so the serial and
    # threaded sums are bitwise equal, and the scratch holds whole cache
    # lines per chunk (8 Float64) so that no two chunks share one.
    slots = rand(0:3, 3000)
    terms, weights = randn(3000), rand(3000)
    serial_sums = with_parallel_grain(SERIAL_GRAIN) do
        TwoPhaseAMR._chunked_slot_sums!(
            zeros(3), Float64[], slots, terms, weights, Val(:weighted),
        )
    end
    scratch = Float64[]
    threaded_sums = with_parallel_grain(1) do
        TwoPhaseAMR._chunked_slot_sums!(zeros(3), scratch, slots, terms, weights, Val(:weighted))
    end
    @test isequal(threaded_sums, serial_sums)
    @test length(scratch) == (cld(3000, 1024) - 1) * 8 + 3

    # Multigrid transfers: row-scheduled prolongation and column-dot
    # restriction equal their serial references, signed zeros included. The
    # references repeat the documented arithmetic (`muladd` in stored order,
    # zero start, and for restriction the added zero that maps -0.0 to 0.0);
    # they are not SparseArrays' `mul!`, whose fusion depends on the Julia
    # version and which the kernels therefore never use.
    prolongation = sprand(3000, 800, 0.004)
    nonzeros(prolongation)[1:3:end] .*= -1
    nonzeros(prolongation)[2:7:end] .= -0.0
    coarse = randn(800)
    coarse[1:5:end] .= -0.0
    fine = randn(3000)
    fine[1:4:end] .= -0.0
    level = (;
        prolongation,
        prolongation_rows = TwoPhaseAMR.TransferRowSchedule(typeof(prolongation)),
    )
    # Serial reference of the prolongation: the documented row schedule
    # (entries of each row in increasing column order, values read fresh from
    # the matrix), gathered row by row with `muladd` from a zero start.
    function scheduled_prolongation(matrix, source)
        schedule = TwoPhaseAMR.TransferRowSchedule(typeof(matrix))
        TwoPhaseAMR._transfer_row_schedule!(schedule, matrix)
        offsets = schedule.offsets
        columns = schedule.columns
        entries = schedule.entries
        values = nonzeros(matrix)
        reference = zeros(size(matrix, 1))
        for row in 1:size(matrix, 1)
            value = 0.0
            for slot in offsets[row]:(offsets[row + 1] - 1)
                value = muladd(values[entries[slot]], source[columns[slot]], value)
            end
            reference[row] = value
        end
        return reference
    end
    function column_order_restriction(matrix, source)
        reference = zeros(size(matrix, 2))
        for column in 1:size(matrix, 2)
            value = 0.0
            for pointer in nzrange(matrix, column)
                value = muladd(
                    nonzeros(matrix)[pointer], source[rowvals(matrix)[pointer]], value,
                )
            end
            reference[column] = value + 0.0
        end
        return reference
    end
    for grain in BOTH_PATHS
        prolongated = with_parallel_grain(() -> TwoPhaseAMR._prolongate!(zeros(3000), level, coarse), grain)
        @test isequal(prolongated, scheduled_prolongation(prolongation, coarse))
        restricted = with_parallel_grain(() -> TwoPhaseAMR._restrict!(zeros(800), prolongation, fine), grain)
        @test isequal(restricted, column_order_restriction(prolongation, fine))
    end
    # One thread and the threaded path agree bitwise (the kernels have a
    # single arithmetic, so no `mul!` fast path can differ by version).
    serial_prolongated = with_parallel_grain(
        () -> TwoPhaseAMR._prolongate!(zeros(3000), level, coarse), SERIAL_GRAIN,
    )
    serial_restricted = with_parallel_grain(
        () -> TwoPhaseAMR._restrict!(zeros(800), prolongation, fine), SERIAL_GRAIN,
    )
    @test isequal(
        with_parallel_grain(() -> TwoPhaseAMR._prolongate!(zeros(3000), level, coarse), 1),
        serial_prolongated,
    )
    @test isequal(
        with_parallel_grain(() -> TwoPhaseAMR._restrict!(zeros(800), prolongation, fine), 1),
        serial_restricted,
    )
    # The schedule reads refreshed values from the matrix it was built for.
    nonzeros(prolongation) .*= 3
    @test isequal(
        TwoPhaseAMR._prolongate!(zeros(3000), level, coarse),
        scheduled_prolongation(prolongation, coarse),
    )

    # A multigrid level's stored transpose: refreshed values on the recorded
    # pattern, a pattern changed in place at constant nnz and a new matrix
    # object all give `copy(transpose(A))` (pattern and value bits, signed
    # zeros included), on the serial and the threaded gather.
    for grain in BOTH_PATHS
        matrix = sprand(1800, 700, 0.004)
        nonzeros(matrix)[2:5:end] .= -0.0
        matrix_transpose = copy(transpose(matrix))
        transpose_map = TwoPhaseAMR.SparseTransposeMap()
        refresh!(source) = with_parallel_grain(
            () -> TwoPhaseAMR._refresh_matrix_transpose!(matrix_transpose, transpose_map, source),
            grain,
        )
        function same_transpose(source)
            expected = copy(transpose(source))
            return matrix_transpose.colptr == expected.colptr &&
                rowvals(matrix_transpose) == rowvals(expected) &&
                isequal(nonzeros(matrix_transpose), nonzeros(expected))
        end
        refresh!(matrix)
        @test same_transpose(matrix)
        nonzeros(matrix) .= randn(nnz(matrix))
        nonzeros(matrix)[3:7:end] .= -0.0
        refresh!(matrix)
        @test same_transpose(matrix)
        # Move the last stored entry of a column to the last row (constant
        # nnz, rows stay sorted).
        last_row = size(matrix, 1)
        moved = findfirst(
            column -> !isempty(nzrange(matrix, column)) &&
                rowvals(matrix)[last(nzrange(matrix, column))] < last_row,
            axes(matrix, 2),
        )
        rowvals(matrix)[last(nzrange(matrix, moved))] = last_row
        refresh!(matrix)
        @test same_transpose(matrix)
        other = sprand(1800, 700, 0.003)
        refresh!(other)
        @test same_transpose(other)
        @test_throws DimensionMismatch refresh!(sprand(700, 1800, 0.003))
    end

    # A stored transpose built with its map is `copy(transpose(A))`, also for
    # rectangular and empty matrices, and the map is recorded: the next
    # refresh on the same pattern is a hit.
    for grain in BOTH_PATHS, (m, n) in ((1800, 700), (700, 1800), (0, 0), (5, 0), (0, 5))
        matrix = sprand(m, n, 0.004)
        nonzeros(matrix)[1:3:end] .= -0.0
        stored, transpose_map = with_parallel_grain(() -> TwoPhaseAMR._stored_transpose(matrix), grain)
        expected = copy(transpose(matrix))
        @test size(stored) == size(expected) && stored.colptr == expected.colptr &&
            rowvals(stored) == rowvals(expected) && isequal(nonzeros(stored), nonzeros(expected))
        @test with_parallel_grain(() -> TwoPhaseAMR._recorded_pattern(transpose_map, matrix), grain)
    end

    # A level refresh with another matrix size is rejected before the level
    # changes.
    level_matrix = sparse([1, 2, 3, 1, 2, 3], [1, 2, 3, 2, 3, 1], [4.0, 5.0, 6.0, -1.0, -1.0, -1.0], 3, 3)
    level = TwoPhaseAMR._multigrid_level(level_matrix, ones(3), nothing)
    @test level.matrix_transpose == copy(transpose(level_matrix))
    @test TwoPhaseAMR._recorded_pattern(level.transpose_map, level_matrix)
    transpose_before = copy(level.matrix_transpose)
    failure = thrown_exception() do
        TwoPhaseAMR._refresh_multigrid_level!(level, sparse(1:4, 1:4, ones(4)), ones(4), nothing)
    end
    @test failure[1] isa ArgumentError &&
        failure[2] == "ArgumentError: multigrid row-form size changed on fixed topology"
    @test level.matrix === level_matrix
    @test level.matrix_transpose == transpose_before

    # The gathered factorized roundoff floor equals the former face scatter,
    # with the diagonal (uniform) and the sparse (two-level) Riesz map.
    for mesh in (
            BlockTreeMesh(base_mesh = (32, 32), block_size = (8, 8), periodic = (false, false)),
            BlockTreeMesh(
                block_size = (8, 8), max_level = 2, periodic = (false, false),
                refine = (key, origin, extent) -> key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
            ),
        )
        volumes = TwoPhaseAMR._cell_volumes(mesh, Float64)
        operator = EllipticOperator(
            mesh; face_coefficient = 0.5 .+ rand(length(mesh.faces)), reaction = 0.75,
            active_cells = trues(ncells(mesh)), volumes,
        )
        action = TwoPhaseAMR.FactorizedEllipticOperator(operator)
        solution, rhs = randn(ncells(mesh)), randn(ncells(mesh))
        solution[1:9:end] .= -0.0
        floor, temporary = scatter_roundoff_floor(action, solution, rhs, operator.volumes)
        for grain in BOTH_PATHS
            gathered = zeros(ncells(mesh))
            @test with_parallel_grain(grain) do
                TwoPhaseAMR._roundoff_residual_floor!(
                    action, solution, rhs, operator.volumes, gathered,
                    TwoPhaseAMR._reduction_partials!(Float64[], ncells(mesh)),
                )
            end === floor
            @test isequal(gathered, temporary)
            @test TwoPhaseAMR._roundoff_residual_floor!(
                action, solution, rhs, operator.volumes, zeros(ncells(mesh)),
            ) === floor
        end
        # Undersized reduction scratch is rejected, not written past its end.
        failure = thrown_exception() do
            TwoPhaseAMR._roundoff_residual_floor!(
                action, solution, rhs, operator.volumes, zeros(ncells(mesh)), Float64[],
            )
        end
        @test isa(failure, Tuple{ArgumentError, String})
        @test occursin("reduction scratch has 0 entries", message_of(failure))
    end
end

# The two fields of `CMDGFMProjectionWorkspace` that `_cmdgfm_row_form!` uses.
mutable struct RowFormWorkspace
    matrix_transpose::SparseMatrixCSC{Float64, Int}
    transpose_map::TwoPhaseAMR.SparseTransposeMap
end

@testset "CMDGFM row-form products are bitwise SparseArrays" begin
    # Rectangular, empty and CMDGFM-shaped matrices (a five-point pressure
    # block with entries over six decades, coupled to a jump block with
    # large and small couplings): the threaded row product equals
    # SparseArrays' serial `mul!` bit for bit, signed zeros included, on the
    # serial and the threaded path of every supported Julia version.
    function cmdgfm_shaped(np)
        rows, columns, entries = Int[], Int[], Float64[]
        for row in 1:np, (column, value) in (
                    (row, 4.0 + rand()), (row - 1, -1.0), (row + 1, -1.0 + 1.0e-3randn()),
                    (row - 37, -1.0), (row + 37, -0.999),
                )
            1 <= column <= np || continue
            push!(rows, row), push!(columns, column), push!(entries, value * exp10(rand(-3:3)))
        end
        for jump in 1:(np ÷ 10)
            row = np + jump
            push!(rows, row), push!(columns, row), push!(entries, 3.0 + randn())
            for cell in rand(1:np, 4)
                push!(rows, row), push!(columns, cell), push!(entries, 1.0e5randn())
                push!(rows, cell), push!(columns, row), push!(entries, 1.0e-5randn())
            end
        end
        return sparse(rows, columns, entries, np + np ÷ 10, np + np ÷ 10)
    end
    rectangular = sprandn(3000, 1700, 0.002)
    nonzeros(rectangular)[2:9:end] .= -0.0
    matrices = [
        spzeros(0, 0), spzeros(0, 4), spzeros(4, 0), sprandn(1, 1, 1.0), sprandn(7, 3, 0.4),
        rectangular, sprandn(2500, 2500, 0.0), cmdgfm_shaped(4000),
    ]
    for matrix in matrices, grain in BOTH_PATHS
        source = randn(size(matrix, 2)) .* exp10.(rand(-8:8, size(matrix, 2)))
        source[1:7:end] .= -0.0
        expected = mul!(fill(NaN, size(matrix, 1)), matrix, source)
        row_form = TwoPhaseAMR._RowForm(copy(transpose(matrix)))
        @test size(row_form) == size(matrix)
        product = with_parallel_grain(() -> mul!(fill(NaN, size(matrix, 1)), row_form, source), grain)
        @test isequal(product, expected)
    end

    # The workspace's row form follows a size change, a pattern change at
    # the same size and new values on the recorded pattern; the asymmetry is
    # the serial lookup maximum; the chunked increment update matches the
    # serial update and its change flag.
    for grain in BOTH_PATHS
        workspace = RowFormWorkspace(spzeros(0, 0), TwoPhaseAMR.SparseTransposeMap())
        same_pattern = cmdgfm_shaped(2000)
        renewed = copy(same_pattern)
        nonzeros(renewed) .= randn(nnz(renewed))
        nonzeros(renewed)[1:11:end] .= -0.0
        for matrix in (cmdgfm_shaped(3000), cmdgfm_shaped(2000), same_pattern, renewed)
            form = with_parallel_grain(() -> TwoPhaseAMR._cmdgfm_row_form!(workspace, matrix), grain)
            @test form.matrix_transpose === workspace.matrix_transpose
            @test form.matrix_transpose == copy(transpose(matrix))
            @test isequal(nonzeros(form.matrix_transpose), nonzeros(copy(transpose(matrix))))
            reference = 0.0
            for column in axes(matrix, 2), pointer in nzrange(matrix, column)
                row = rowvals(matrix)[pointer]
                reference = max(reference, abs(nonzeros(matrix)[pointer] - matrix[column, row]))
            end
            asymmetry = with_parallel_grain(grain) do
                TwoPhaseAMR._cmdgfm_matrix_asymmetry(matrix, form.matrix_transpose, Float64[])
            end
            @test asymmetry === reference
        end
    end
    for grain in BOTH_PATHS, unchanged in (false, true)
        solution, columns, increment = randn(5000), rand(5000), randn(5000)
        unchanged && fill!(increment, 0.0)
        expected = solution .+ columns .* (0.25 .* increment)
        flags = Bool[]
        changed = with_parallel_grain(grain) do
            TwoPhaseAMR._cmdgfm_apply_increment!(solution, columns, 0.25, increment, flags)
        end
        @test changed == !unchanged
        @test isequal(solution, expected)
        @test length(flags) == cld(5000, 1024)
    end
end

@testset "two-phase face kernels are bitwise serial" begin
    # More than one fixed reduction chunk of faces, so the chunked marking,
    # checks and maxima are threaded at grain 1.
    mesh = BlockTreeMesh(base_mesh = (32, 48), block_size = (8, 8), periodic = (true, false))
    interface = InterfaceField(mesh, (x, y) -> y - 0.5 - 0.05cospi(2x); lipschitz = 1)
    @test TwoPhaseAMR._reduction_chunks(length(mesh.faces)) > 1
    field(x, y, time, axis) = axis == 1 ? sinpi(2y) * cospi(x) : 0.3sinpi(2x) * y
    phase_velocity() = (FaceVelocity(mesh, field), FaceVelocity(mesh, field))
    enforce(velocity) = (enforce_no_permeation!(velocity); velocity)
    reference = enforce(phase_velocity())
    with_parallel_grain(SERIAL_GRAIN) do
        TwoPhaseAMR._mark_inactive_phase_velocities!(reference, interface, Int[])
    end
    threaded = enforce(phase_velocity())
    failures = Int[]
    with_parallel_grain(1) do
        TwoPhaseAMR._mark_inactive_phase_velocities!(threaded, interface, failures)
    end
    for phase in 1:2
        @test isequal(threaded[phase].interior, reference[phase].interior)
        @test isequal(threaded[phase].boundary, reference[phase].boundary)
    end
    @test any(isnan, threaded[1].interior) && any(isnan, threaded[2].interior)

    # The first active non-finite face is reported, whatever the thread count.
    active = findall(isfinite, threaded[2].interior)
    broken = map(copy, threaded)
    broken[2].interior[active[end]] = NaN
    broken[1].interior[findfirst(isfinite, broken[1].interior)] = Inf
    expected = findfirst(isfinite, threaded[1].interior)
    messages = map((1, SERIAL_GRAIN)) do grain
        with_parallel_grain(grain) do
            message_of(
                thrown_exception() do
                    TwoPhaseAMR._mark_inactive_phase_velocities!(map(copy, broken), interface, Int[])
                end
            )
        end
    end
    @test messages[1] == messages[2]
    @test occursin("phase-1 velocity on face $expected", messages[1])

    # Interior and boundary failures at the same time: the interior error wins
    # (its chunk flags are filled before the serial boundary pass) and names
    # the first failing face, liquid before gas. Every thread count reports
    # the same face and phase.
    both_phases = filter(
        index -> isfinite(threaded[1].interior[index]) &&
            isfinite(threaded[2].interior[index]),
        eachindex(threaded[1].interior),
    )
    active_boundary = [
        index for (index, face) in pairs(mesh.boundary_faces) if
            TwoPhaseAMR._phase_cell_is_active(interface, face.cell, :liquid)
    ]
    @test !isempty(both_phases) && !isempty(active_boundary)
    simultaneous = map(copy, threaded)
    simultaneous[1].interior[first(both_phases)] = NaN
    simultaneous[2].interior[first(both_phases)] = Inf
    simultaneous[1].boundary[first(active_boundary)] = NaN
    simultaneous[2].boundary[first(active_boundary)] = Inf
    messages = map((1, SERIAL_GRAIN)) do grain
        with_parallel_grain(grain) do
            message_of(
                thrown_exception() do
                    TwoPhaseAMR._mark_inactive_phase_velocities!(
                        map(copy, simultaneous), interface, Int[],
                    )
                end
            )
        end
    end
    @test messages[1] == messages[2] ==
        "ArgumentError: non-finite active phase-1 velocity on face $(first(both_phases))"
    # With the interior values valid the boundary pass reports its own first
    # failing face, liquid before gas, at every thread count.
    boundary_only = map(copy, threaded)
    boundary_only[1].boundary[first(active_boundary)] = NaN
    boundary_only[2].boundary[first(active_boundary)] = Inf
    messages = map((1, SERIAL_GRAIN)) do grain
        with_parallel_grain(grain) do
            message_of(
                thrown_exception() do
                    TwoPhaseAMR._mark_inactive_phase_velocities!(
                        map(copy, boundary_only), interface, Int[],
                    )
                end
            )
        end
    end
    @test messages[1] == messages[2] ==
        "ArgumentError: non-finite active phase-1 boundary velocity on face $(first(active_boundary))"

    # WY frozen-interface bound: exact fixed-chunk maxima, also with the
    # velocity checks threaded; allocation-free face maxima.
    work = WYAdvectionWorkspace(interface)
    TwoPhaseAMR._freeze_phase_apertures!(interface, work)
    bounds = map((SERIAL_GRAIN, 1)) do grain
        with_parallel_grain(grain) do
            TwoPhaseAMR._frozen_interface_stable_timestep(threaded, work)
        end
    end
    @test isfinite(bounds[1]) && bounds[1] === bounds[2]
    # A warm launch allocates nothing per face on either path. The threaded
    # path additionally pays whatever the parallel runtime packs for the
    # launch: Polyester boxes the closure argument on Julia 1.11 (16-32 B
    # measured per launch) and not from 1.12 on, while a per-face allocation
    # would cost tens of kilobytes here.
    for grain in (SERIAL_GRAIN, 1)
        with_parallel_grain(grain) do
            TwoPhaseAMR._frozen_face_maximum(threaded, work, nothing)
            allocated = @allocated TwoPhaseAMR._frozen_face_maximum(
                threaded, work, nothing,
            )
            if grain == SERIAL_GRAIN || VERSION >= v"1.12"
                @test allocated == 0
            else
                @test allocated <= 64
            end
        end
    end
    gas_failure = map(copy, threaded)
    gas_failure[2].interior[active[1]] = NaN
    messages = map((1, SERIAL_GRAIN)) do grain
        with_parallel_grain(grain) do
            message_of(
                thrown_exception() do
                    TwoPhaseAMR._frozen_interface_stable_timestep(gas_failure, work)
                end
            )
        end
    end
    @test messages[1] == messages[2] == "ArgumentError: active gas velocity must be finite"
    # Both phases invalid: the liquid error wins, as in the serial order.
    both = map(copy, gas_failure)
    both[1].interior[findlast(isfinite, both[1].interior)] = Inf
    messages = map((1, SERIAL_GRAIN)) do grain
        with_parallel_grain(grain) do
            message_of(
                thrown_exception() do
                    TwoPhaseAMR._frozen_interface_stable_timestep(both, work)
                end
            )
        end
    end
    @test messages[1] == messages[2] == "ArgumentError: active liquid velocity must be finite"
    # Boundary and interior failures at the same time: the errors follow the
    # serial phase order (liquid interior, liquid boundary, gas interior, gas
    # boundary), the same at every thread count. A liquid boundary failure
    # therefore wins over a gas interior failure...
    active_gas_boundary = [
        index for (index, face) in pairs(mesh.boundary_faces) if
            TwoPhaseAMR._phase_cell_is_active(interface, face.cell, :gas)
    ]
    @test !isempty(active_gas_boundary)
    mixed = map(copy, threaded)
    mixed[2].interior[active[1]] = NaN
    mixed[1].boundary[first(active_boundary)] = one(eltype(mixed[1].boundary))
    messages = map((1, SERIAL_GRAIN)) do grain
        with_parallel_grain(grain) do
            message_of(
                thrown_exception() do
                    TwoPhaseAMR._frozen_interface_stable_timestep(mixed, work)
                end
            )
        end
    end
    @test messages[1] == messages[2] ==
        "ArgumentError: WY requires zero active phase boundary velocity"
    # ...while within one phase the interior failure comes first.
    interior_first = map(copy, mixed)
    interior_first[1].boundary[first(active_boundary)] = zero(eltype(mixed[1].boundary))
    interior_first[2].boundary[first(active_gas_boundary)] = one(eltype(mixed[2].boundary))
    messages = map((1, SERIAL_GRAIN)) do grain
        with_parallel_grain(grain) do
            message_of(
                thrown_exception() do
                    TwoPhaseAMR._frozen_interface_stable_timestep(interior_first, work)
                end
            )
        end
    end
    @test messages[1] == messages[2] == "ArgumentError: active gas velocity must be finite"
end

@testset "phase face average is bitwise serial" begin
    # Mixed-level periodic/wall mesh with a cut solid: both axes, plain and
    # embedded, more faces than the grain-1 threaded path needs.
    mesh = BlockTreeMesh(
        base_mesh = (24, 24), block_size = (4, 4), extent = (12.0, 12.0),
        periodic = (true, false), max_level = 2,
        refine = (key, origin, extent) -> key.level < 2 && origin[1] < 6 && origin[2] < 7,
    )
    geometry = GeometryField(mesh, (x, y) -> hypot(x - 8.07, y - 6.33) - 1.55; lipschitz = 1)
    source = CellField(mesh)
    for block in 1:nblocks(mesh)
        cells = interior(source, block)
        for j in axes(cells, 2), i in axes(cells, 1)
            x, y = cellcenter(mesh, block, i, j)
            cells[i, j] = sin(1.3x + 0.1block) * cos(0.7y) + 0.01i * j
        end
    end
    for (Axis, solid) in Iterators.product((1, 2), (nothing, geometry))
        averages = map((SERIAL_GRAIN, 1)) do grain
            destination = FaceField{Axis}(mesh)
            with_parallel_grain(grain) do
                TwoPhaseAMR._phase_face_average!(destination, source, solid)
            end
            destination
        end
        @test isequal(averages[1].interior, averages[2].interior)
        @test isequal(averages[1].boundary, averages[2].boundary)
        isnothing(solid) || continue
        # The plain average is the spacing-weighted mean of the two cells.
        axis_faces = filter(face -> face.axis == Axis, mesh.faces)
        expected = map(axis_faces) do face
            hm = mesh.blocks[face.minus.block].spacing[Axis]
            hp = mesh.blocks[face.plus.block].spacing[Axis]
            (hm * TwoPhaseAMR._get(source, face.minus) + hp * TwoPhaseAMR._get(source, face.plus)) / (hm + hp)
        end
        @test isequal(averages[2].interior, expected)
    end
end

@testset "face component copies are range copies" begin
    # The serial face-filter mapping the range copies replace.
    function filtered_component(velocity, Axis)
        interior_values = [velocity.interior[k] for (k, face) in pairs(velocity.mesh.faces) if face.axis == Axis]
        boundary_values = [velocity.boundary[k] for (k, face) in pairs(velocity.mesh.boundary_faces) if face.axis == Axis]
        return interior_values, boundary_values
    end
    for periodic in ((true, true), (true, false), (false, false))
        mesh = BlockTreeMesh(
            base_mesh = (24, 24), block_size = (4, 4), extent = (12.0, 12.0),
            periodic = periodic, max_level = 2,
            refine = (key, origin, extent) -> key.level < 2 && origin[1] < 6 && origin[2] < 7,
        )
        velocity = FaceVelocity(mesh, (x, y, time, axis) -> axis * sin(x) + cos(1.7y))
        for Axis in (1, 2), grain in (1, SERIAL_GRAIN)
            expected_interior, expected_boundary = filtered_component(velocity, Axis)
            component = FaceField{Axis}(mesh)
            with_parallel_grain(() -> copyto!(component, velocity), grain)
            @test isequal(component.interior, expected_interior)
            @test isequal(component.boundary, expected_boundary)
            all(periodic) && @test isempty(component.boundary)
            # Back into a velocity: only the faces of `Axis` change.
            component.interior .*= -2
            component.boundary .*= -2
            target = copy(velocity)
            with_parallel_grain(() -> copyto!(target, component), grain)
            @test isequal(filtered_component(target, Axis), (component.interior, component.boundary))
            @test isequal(filtered_component(target, 3 - Axis), filtered_component(velocity, 3 - Axis))
        end
    end
    destination = zeros(3000)
    source = collect(1.0:3000.0)
    for grain in (1, SERIAL_GRAIN)
        fill!(destination, 0)
        with_parallel_grain(() -> TwoPhaseAMR._parallel_copyto!(destination, 10, source, 500, 2400), grain)
        @test destination[11:2410] == source[501:2900]
        @test iszero(destination[1:10]) && iszero(destination[2411:end])
        @test with_parallel_grain(() -> TwoPhaseAMR._parallel_copyto!(destination, 3000, source, 3000, 0), grain) === destination
        @test_throws BoundsError TwoPhaseAMR._parallel_copyto!(destination, 2000, source, 0, 1001)
        @test_throws BoundsError TwoPhaseAMR._parallel_copyto!(destination, 0, source, 2001, 1000)
        @test_throws ArgumentError TwoPhaseAMR._parallel_copyto!(destination, 0, source, 0, -1)
    end
end
