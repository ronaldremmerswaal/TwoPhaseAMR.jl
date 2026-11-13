using LinearAlgebra: Diagonal
using SparseArrays: SparseMatrixCSC, dropzeros!, nnz, nonzeros, nzrange, rowvals, sparse, spdiagm
using Random: MersenneTwister

isdefined(@__MODULE__, :test_cell_values) ||
    include(joinpath(@__DIR__, "..", "test_utils.jl"))

# Transcription of the SparseArrays assembly that `_integrated_stiffness`
# replaced: sparse(J), transpose(J) * M * J, (A + A') / 2, dropzeros!, then the
# pattern completion with every cell pair of a metric entry and the diagonal.
function reference_elliptic_stiffness(mesh, metric, coefficient::Vector{T}) where {T}
    nx, ny = mesh.block_size
    number_of_faces = length(mesh.faces)
    number_of_cells = TwoPhaseAMR.ncells(mesh)
    rows, columns, values = Int[], Int[], T[]
    for (face_index, face) in pairs(mesh.faces)
        append!(rows, (face_index, face_index))
        append!(
            columns, (
                TwoPhaseAMR._cell_index(face.minus, nx, ny),
                TwoPhaseAMR._cell_index(face.plus, nx, ny),
            ),
        )
        append!(values, (-T(face.area), T(face.area)))
    end
    jump = sparse(rows, columns, values, number_of_faces, number_of_cells)
    face_metric = if metric isa TwoPhaseAMR.DiagonalInverseFaceMetric
        spdiagm(0 => coefficient .* metric.diagonal)
    else
        square_root = spdiagm(0 => sqrt.(coefficient))
        square_root * metric.matrix * square_root
    end
    stiffness = transpose(jump) * face_metric * jump
    stiffness = (stiffness + transpose(stiffness)) / 2
    dropzeros!(stiffness)
    pattern_rows, pattern_columns = collect(1:number_of_cells), collect(1:number_of_cells)
    entries = metric isa TwoPhaseAMR.DiagonalInverseFaceMetric ?
        [(face, face) for face in eachindex(metric.diagonal)] :
        [
            (rowvals(metric.matrix)[pointer], column) for column in axes(metric.matrix, 2)
            for pointer in nzrange(metric.matrix, column)
        ]
    for (left_face, right_face) in entries
        left, right = mesh.faces[left_face], mesh.faces[right_face]
        for a in (left.minus, left.plus), b in (right.minus, right.plus)
            push!(pattern_rows, TwoPhaseAMR._cell_index(a, nx, ny))
            push!(pattern_columns, TwoPhaseAMR._cell_index(b, nx, ny))
        end
    end
    completed = sparse(
        pattern_rows, pattern_columns, ones(T, length(pattern_rows)),
        number_of_cells, number_of_cells,
    )
    fill!(nonzeros(completed), zero(T))
    for column in axes(stiffness, 2), pointer in nzrange(stiffness, column)
        row = rowvals(stiffness)[pointer]
        slot = TwoPhaseAMR._csc_stored_pointer(completed, row, column)
        nonzeros(completed)[slot] = nonzeros(stiffness)[pointer]
    end
    return completed
end

bit_pattern(values::AbstractVector{Float32}) = reinterpret(UInt32, values)
bit_pattern(values::AbstractVector{Float64}) = reinterpret(UInt64, values)

@testset "Elliptic stiffness is bitwise the SparseArrays assembly" begin
    rng = MersenneTwister(20261113)
    for T in (Float32, Float64), periodic in ((true, true), (false, false), (true, false)),
            mixed in (false, true)
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), max_level = 2, periodic = periodic,
            refine = (key, origin, extent) -> mixed && (
                (key.level == 0 && key.i <= 1 && key.j == 0) ||
                    (key.level == 1 && key.i == 1 && key.j == 1)
            ),
        )
        base = GradientOperator(mesh, T)
        number_of_faces = length(mesh.faces)
        # Zero scales give explicitly stored zeros in the metric.
        scale = T[isodd(index) ? 0 : 1 + rand(rng) for index in 1:number_of_faces]
        scaled = TwoPhaseAMR._scaled_inverse_metric(base.inverse_metric, scale)
        operators = (
            base,
            TwoPhaseAMR.GradientOperator(
                mesh, scaled, zeros(T, number_of_faces), base.construction_cache,
            ),
        )
        coefficients = (
            T[rand(rng) for _ in 1:number_of_faces],
            T[10.0^rand(rng, -8:8) * rand(rng) for _ in 1:number_of_faces],
            T[rand(rng) < 0.3 ? 0 : rand(rng) for _ in 1:number_of_faces],
            # Equal coefficients cancel exactly in regular interior rows.
            fill(T(0.1), number_of_faces),
        )
        for operator in operators, coefficient in coefficients
            actual = TwoPhaseAMR._integrated_stiffness(mesh, operator, coefficient)
            expected = reference_elliptic_stiffness(mesh, operator.inverse_metric, coefficient)
            @test actual.colptr == expected.colptr
            @test rowvals(actual) == rowvals(expected)
            @test bit_pattern(nonzeros(actual)) == bit_pattern(nonzeros(expected))
        end
        # The refresh map is coefficient-independent; its reference is the
        # sortperm grouping of "Fixed-pattern elliptic refresh" in poisson.jl.
        stiffness = TwoPhaseAMR._integrated_stiffness(mesh, base, ones(T, number_of_faces))
        refresh = TwoPhaseAMR._elliptic_coefficient_refresh_map(mesh, base, stiffness)
        @test refresh.offsets[end] == length(refresh.stiffness_pointers) + 1
        @test issorted(refresh.stiffness_pointers)
    end
    # Diagonal metric: a uniform periodic mesh.
    mesh = BlockTreeMesh(base_mesh = (8, 8), block_size = (4, 4), max_level = 0)
    operator = GradientOperator(mesh, Float64)
    @test operator.inverse_metric isa TwoPhaseAMR.DiagonalInverseFaceMetric
    coefficient = [10.0^rand(rng, -8:8) for _ in eachindex(mesh.faces)]
    actual = TwoPhaseAMR._integrated_stiffness(mesh, operator, coefficient)
    expected = reference_elliptic_stiffness(mesh, operator.inverse_metric, coefficient)
    @test (actual.colptr, rowvals(actual)) == (expected.colptr, rowvals(expected))
    @test bit_pattern(nonzeros(actual)) == bit_pattern(nonzeros(expected))
end

@testset "Inverse-metric assembly does not scale with the refinement ceiling" begin
    function metric_cost(max_level)
        mesh = BlockTreeMesh(
            base_mesh = (16, 16), block_size = (4, 4), max_level = max_level,
            refine = (key, origin, extent) -> key.level == 0 && key.i <= 1 && key.j <= 1,
        )
        cache = TwoPhaseAMR.GradientConstructionCache(Float64)
        metric = TwoPhaseAMR._assemble_inverse_metric(mesh, Float64, cache)
        bytes = @allocated TwoPhaseAMR._assemble_inverse_metric(mesh, Float64, cache)
        return metric, bytes
    end
    shallow, shallow_bytes = metric_cost(6)
    deep, deep_bytes = metric_cost(20)
    @test (shallow.colptr, rowvals(shallow)) == (deep.colptr, rowvals(deep))
    @test nonzeros(shallow) == nonzeros(deep)
    # The same active mesh: allocation independent of max_level (it grew
    # with 2^max_level before).
    @test deep_bytes <= 1.05 * shallow_bytes + 4096
end

# Transcription of the per-ghost `_height_cache!` that the flat donor array
# replaced: halo cells in block, row, column order, donors by block index.
function reference_height_ghosts(mesh, ::Type{T}) where {T}
    nx, ny = mesh.block_size
    indices_by_key = Dict(block.key => index for (index, block) in pairs(mesh.blocks))
    lattice = TwoPhaseAMR._finest_lattice_size(mesh)
    ghosts = []
    for (b, block) in pairs(mesh.blocks)
        scale = TwoPhaseAMR._block_scale(mesh, block.key.level)
        for j in -3:(ny + 4), i in -3:(nx + 4)
            1 <= i <= nx && 1 <= j <= ny && continue
            xlo = (block.key.i * nx + i - 1) * scale
            ylo = (block.key.j * ny + j - 1) * scale
            if (!mesh.periodic[1] && !(0 <= xlo && xlo + scale <= lattice[1])) ||
                    (!mesh.periodic[2] && !(0 <= ylo && ylo + scale <= lattice[2]))
                continue
            end
            mesh.periodic[1] && (xlo = mod(xlo, lattice[1]))
            mesh.periodic[2] && (ylo = mod(ylo, lattice[2]))
            xhi, yhi = xlo + scale, ylo + scale
            nodes = TwoPhaseAMR._tree_node_type(mesh.root)[]
            TwoPhaseAMR._collect_leaf_nodes_in_window!(
                nodes, mesh.root, (xlo, ylo), (xhi, yhi), mesh.max_level, mesh.block_size,
            )
            donors = []
            for db in sort!([indices_by_key[node.data.key] for node in nodes])
                other = mesh.blocks[db]
                other_scale = TwoPhaseAMR._block_scale(mesh, other.key.level)
                ox, oy = other.key.i * nx * other_scale, other.key.j * ny * other_scale
                for dj in 1:ny, di in 1:nx
                    cx, cy = ox + (di - 1) * other_scale, oy + (dj - 1) * other_scale
                    left, right = max(xlo, cx), min(xhi, cx + other_scale)
                    low, high = max(ylo, cy), min(yhi, cy + other_scale)
                    right > left && high > low || continue
                    bounds = (
                        T((left - cx) / other_scale - 0.5), T((right - cx) / other_scale - 0.5),
                        T((low - cy) / other_scale - 0.5), T((high - cy) / other_scale - 0.5),
                    )
                    weight = T((right - left) * (high - low) / (scale * scale))
                    push!(donors, (TwoPhaseAMR.CellRef(db, di, dj), bounds, weight))
                end
            end
            push!(ghosts, (b, i + 4, j + 4, donors))
        end
    end
    return ghosts
end

@testset "Height-function halo cache matches the per-ghost construction" begin
    # The ghost and donor lists are mesh topology only: a solid changes which
    # donors `_fill_height_values!` accepts, not the lists themselves.
    for periodic in ((true, true), (false, false), (true, false)), mixed in (false, true)
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), max_level = 2, periodic = periodic,
            refine = (key, origin, extent) -> mixed && (
                (key.level == 0 && key.i <= 1 && key.j == 0) ||
                    (key.level == 1 && key.i == 1 && key.j == 1)
            ),
        )
        field = InterfaceField(mesh; T = Float64)
        cache = TwoPhaseAMR._height_cache!(field)
        actual = [
            (
                ghost.block, ghost.i, ghost.j,
                [
                    (donor.cell, donor.bounds, donor.weight) for donor in
                        cache.donors[ghost.first_donor .+ (0:(ghost.donor_count - 1))]
                ],
            ) for ghost in cache.ghosts
        ]
        @test actual == reference_height_ghosts(mesh, Float64)
        @test cache.keys == [block.key for block in mesh.blocks]
    end
end
