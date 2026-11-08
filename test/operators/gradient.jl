using LinearAlgebra: Symmetric, eigvals, isposdef
using SparseArrays: dropzeros!, nonzeros, sparse

function cell_inner_product(a::CellField, b::CellField)
    result = zero(promote_type(eltype(a), eltype(b)))
    nx, ny = a.mesh.block_size
    for (block_index, block) in pairs(a.mesh.blocks), j in 1:ny, i in 1:nx
        ref = TwoPhaseAMR.CellRef(block_index, i, j)
        result += block.volume * TwoPhaseAMR._get(a, ref) * TwoPhaseAMR._get(b, ref)
    end
    return result
end

function wrapped_face_displacement(mesh, face)
    minus = cellcenter(mesh, face.minus.block, face.minus.i, face.minus.j)
    plus = cellcenter(mesh, face.plus.block, face.plus.i, face.plus.j)
    return ntuple(2) do axis
        delta = plus[axis] - minus[axis]
        delta - round(delta / mesh.extent[axis]) * mesh.extent[axis]
    end
end

@testset "vertex grouping preserves gradient metric assembly" begin
    function dictionary_metric(mesh, ::Type{T}) where {T}
        patches = Dict{Tuple{Int, Int}, Vector{TwoPhaseAMR.MetricFace{T}}}()
        for (index, face) in pairs(mesh.faces)
            displacement = TwoPhaseAMR._face_displacement(mesh, face)
            metric_face = TwoPhaseAMR.MetricFace(
                index, face.axis, T(face.area),
                (T(displacement[1]), T(displacement[2])),
            )
            for endpoint in 1:2
                key = TwoPhaseAMR._vertex_key(mesh, face, endpoint)
                patch = get!(patches, key) do
                    TwoPhaseAMR.MetricFace{T}[]
                end
                push!(patch, metric_face)
            end
        end
        rows, columns, values = Int[], Int[], T[]
        cache = TwoPhaseAMR.GradientConstructionCache(T)
        for key in sort!(collect(keys(patches)))
            faces = patches[key]
            if TwoPhaseAMR._physical_vertex(mesh, key) || length(faces) <= 2
                TwoPhaseAMR._append_boundary_patch_metric!(rows, columns, values, faces)
            else
                metric = TwoPhaseAMR._cached_local_inverse_metric!(cache, faces)
                for (column, right) in pairs(faces), (row, left) in pairs(faces)
                    push!(rows, left.index)
                    push!(columns, right.index)
                    push!(values, metric[row, column])
                end
            end
        end
        result = sparse(rows, columns, values, length(mesh.faces), length(mesh.faces))
        dropzeros!(result)
        return result
    end

    for periodic in ((true, true), (true, false), (false, false)),
            T in (Float32, Float64)
        mesh = BlockTreeMesh(
            base_mesh = (8, 8), block_size = (4, 4), max_level = 2,
            origin = (-0.3, 0.2), extent = (1.2, 0.8), periodic = periodic,
            refine = (key, origin, extent) ->
            key.level == 0 && key.i == 0 && key.j == 0,
        )
        actual = TwoPhaseAMR._assemble_inverse_metric(
            mesh, T, TwoPhaseAMR.GradientConstructionCache(T),
        )
        reference = dictionary_metric(mesh, T)
        @test actual.colptr == reference.colptr
        @test actual.rowval == reference.rowval
        @test isequal(nonzeros(actual), nonzeros(reference))
    end
end

@testset "gradient templates use typed variable-length keys" begin
    for T in (Float32, Float64)
        cache = TwoPhaseAMR.GradientConstructionCache(T)
        @test keytype(cache.local_metrics) === TwoPhaseAMR.GradientMetricKey{T}
        @test isconcretetype(keytype(cache.local_metrics))
        @test !ismutabletype(keytype(cache.local_metrics))
        descriptors = [
            (1, one(T), one(T), zero(T)), (2, one(T), zero(T), one(T)),
        ]
        first_key = TwoPhaseAMR.GradientMetricKey(copy(descriptors))
        equal_key = TwoPhaseAMR.GradientMetricKey(copy(descriptors))
        reversed_key = TwoPhaseAMR.GradientMetricKey(reverse(descriptors))
        shorter_key = TwoPhaseAMR.GradientMetricKey(descriptors[1:1])
        different_key = TwoPhaseAMR.GradientMetricKey(
            [
                descriptors[1], (2, T(2), zero(T), one(T)),
            ]
        )
        @test first_key == equal_key
        @test isequal(first_key, equal_key)
        @test hash(first_key) == hash(equal_key)
        @test !isequal(first_key, reversed_key)
        @test !isequal(first_key, shorter_key)
        @test !isequal(first_key, different_key)
        lookup = Dict(first_key => 1)
        @test lookup[equal_key] == 1
        @test !haskey(lookup, reversed_key)
        @test !haskey(lookup, shorter_key)
        @test !haskey(lookup, different_key)

        patches = [
            [
                TwoPhaseAMR.MetricFace{T}(
                    i, isodd(i) ? 1 : 2, one(T),
                    isodd(i) ? (one(T), zero(T)) : (zero(T), one(T)),
                ) for i in 1:count
            ] for count in (3, 4, 5)
        ]
        for faces in patches
            result = TwoPhaseAMR._cached_local_inverse_metric!(cache, faces)
            reverse!(faces)
            @test TwoPhaseAMR._cached_local_inverse_metric!(cache, faces) == result
        end
        @test cache.misses == 3
        @test cache.hits == 3
        @test sort!([length(key.descriptors) for key in keys(cache.local_metrics)]) == [3, 4, 5]
        @test all(key -> key.descriptors !== cache.key_scratch, keys(cache.local_metrics))
        TwoPhaseAMR._cached_local_inverse_metric!(cache, reverse(patches[1]))
        @test cache.misses == 3
        @test cache.hits == 4
    end
end

@testset "compatible pressure gradient" begin
    uniform = BlockTreeMesh(block_size = (8, 6), extent = (2.0, 1.5))
    pressure = CellField(uniform)
    initialize!(pressure, (x, y) -> sinpi(x) + 0.2cospi(4y / 3))
    gradient = FaceVelocity(uniform, (x, y, t, axis) -> 0.0)
    operator = GradientOperator(pressure)
    gradient!(gradient, pressure, operator)

    @test operator.inverse_metric isa TwoPhaseAMR.DiagonalInverseFaceMetric
    for (index, face) in pairs(uniform.faces)
        displacement = wrapped_face_displacement(uniform, face)
        distance = displacement[face.axis]
        expected = (
            TwoPhaseAMR._get(pressure, face.plus) -
                TwoPhaseAMR._get(pressure, face.minus)
        ) / distance
        @test gradient.interior[index] ≈ expected atol = 8eps() rtol = 8eps()
    end

    constant_pressure = CellField(uniform; init = 3.25)
    gradient!(gradient, constant_pressure, operator)
    @test all(iszero, gradient.interior)

    uniform32 = BlockTreeMesh(
        block_size = (4, 4), origin = (0.0f0, 0.0f0), extent = (1.0f0, 1.0f0)
    )
    pressure32 = CellField(uniform32, Float32; init = 1)
    operator32 = GradientOperator(pressure32)
    gradient32 = FaceVelocity(
        uniform32, (x, y, t, axis) -> 0.0f0; T = Float32
    )
    gradient!(gradient32, pressure32, operator32)
    @test eltype(operator32.pressure_jump) == Float32
    @test eltype(gradient32) == Float32

    physical = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    physical_operator = GradientOperator(physical)
    physical_pressure = CellField(physical)
    initialize!(physical_pressure, (x, y) -> cospi(x) + 0.25cospi(2y))
    physical_gradient = FaceVelocity(
        physical, (x, y, time, axis) -> one(x)
    )
    gradient!(physical_gradient, physical_pressure, physical_operator)
    @test all(iszero, physical_gradient.boundary)
    @test physical_operator.inverse_metric isa
        TwoPhaseAMR.DiagonalInverseFaceMetric
    stretched_amr = BlockTreeMesh(
        block_size = (4, 4), extent = (2.0, 1.0), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    stretched_operator = GradientOperator(stretched_amr)
    @test stretched_operator.inverse_metric isa
        TwoPhaseAMR.SparseInverseFaceMetric
    @test isposdef(Symmetric(stretched_operator.inverse_metric.matrix))

    function stretched_gradient_error(n)
        mesh = BlockTreeMesh(
            block_size = (2n, n), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 ||
                (key.level == 1 && key.i == 0 && key.j == 0),
        )
        pressure = CellField(
            mesh, (x, y) -> sinpi(2x) + 0.2cospi(2y),
        )
        numerical = FaceVelocity(mesh)
        gradient!(numerical, pressure, GradientOperator(mesh))
        squared_error = total_area = 0.0
        for (index, face) in pairs(mesh.faces)
            exact = face.axis == 1 ?
                2pi * cospi(2face.center[1]) :
                -0.4pi * sinpi(2face.center[2])
            squared_error += face.area * (numerical.interior[index] - exact)^2
            total_area += face.area
        end
        return sqrt(squared_error / total_area)
    end
    stretched_errors = collect(stretched_gradient_error.((4, 8, 16)))
    stretched_rates = log2.(
        stretched_errors[1:(end - 1)] ./ stretched_errors[2:end],
    )
    @test all(diff(stretched_errors) .< 0)
    @test minimum(stretched_rates) > 1.0
end

@testset "AMR inverse face metric" begin
    mesh = BlockTreeMesh(
        block_size = (4, 4), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    operator = GradientOperator(mesh)
    @test operator.inverse_metric isa TwoPhaseAMR.SparseInverseFaceMetric
    @test isposdef(Symmetric(operator.inverse_metric.matrix))

    # The vertex-patch Riesz map exactly maps the area-weighted pressure jumps
    # of either affine coordinate field to its face-normal gradient. Working
    # directly with jumps avoids the artificial discontinuity of an affine
    # field across a periodic seam.
    jump = similar(operator.pressure_jump)
    result = similar(jump)
    for coordinate_axis in 1:2
        for (index, face) in pairs(mesh.faces)
            displacement = wrapped_face_displacement(mesh, face)
            jump[index] = face.area * displacement[coordinate_axis]
        end
        TwoPhaseAMR._mul_inverse_metric!(
            result, operator.inverse_metric, jump
        )
        for (index, face) in pairs(mesh.faces)
            expected = face.axis == coordinate_axis ? 1.0 : 0.0
            @test result[index] ≈ expected atol = 2.0e-13 rtol = 2.0e-13
        end
    end

    # Symmetry and definiteness of -D*G are the directly usable consequences
    # of gradient--divergence duality for the pressure projection.
    p = CellField(mesh)
    q = CellField(mesh)
    initialize!(p, (x, y) -> sinpi(2x) * cospi(4y) + 0.1cospi(6x - 2y))
    initialize!(q, (x, y) -> cospi(4x) * sinpi(2y) - 0.2sinpi(2x + 6y))
    gp = FaceVelocity(mesh, (x, y, t, axis) -> 0.0)
    gq = copy(gp)
    dgp = CellField(mesh)
    dgq = CellField(mesh)
    gradient!(gp, p, operator)
    gradient!(gq, q, operator)
    divergence!(dgp, gp)
    divergence!(dgq, gq)
    @test cell_inner_product(p, dgq) ≈ cell_inner_product(q, dgp) atol = 3.0e-12 rtol = 3.0e-12
    @test cell_inner_product(p, dgp) <= 64eps()

    constant_pressure = CellField(mesh; init = -2)
    gradient!(gp, constant_pressure, operator)
    @test maximum(abs, gp.interior) <= 32eps()

    # The mixed-level pressure operator has exactly the constant null mode;
    # no refinement-interface checkerboard mode is admitted.
    nx, ny = mesh.block_size
    refs = [
        TwoPhaseAMR.CellRef(block, i, j)
            for block in eachindex(mesh.blocks) for j in 1:ny for i in 1:nx
    ]
    pressure_matrix = zeros(length(refs), length(refs))
    basis = CellField(mesh)
    for (column, source) in pairs(refs)
        for block in eachindex(mesh.blocks)
            fill!(interior(basis, block), 0)
        end
        TwoPhaseAMR._set!(basis, 1.0, source)
        gradient!(gp, basis, operator)
        divergence!(dgp, gp)
        for (row, target) in pairs(refs)
            volume = mesh.blocks[target.block].volume
            pressure_matrix[row, column] =
                -volume * TwoPhaseAMR._get(dgp, target)
        end
    end
    @test pressure_matrix ≈ transpose(pressure_matrix) atol = 2.0e-13 rtol = 2.0e-13
    eigenvalues = eigvals(Symmetric(pressure_matrix))
    spectral_tolerance = 2.0e-12 * maximum(eigenvalues)
    @test first(eigenvalues) >= -spectral_tolerance
    @test count(value -> abs(value) <= spectral_tolerance, eigenvalues) == 1
end

@testset "gradient is independent of uniform block partitioning" begin
    oneblock = BlockTreeMesh(block_size = (8, 8))
    fourblocks = BlockTreeMesh(
        block_size = (4, 4), max_level = 1,
        refine = (key, origin, extent) -> true,
    )
    fourroots = BlockTreeMesh(base_mesh = (8, 8), block_size = (4, 4))
    pressure_function = (x, y) -> sinpi(2x) + 0.3cospi(4y)
    pa = CellField(oneblock)
    pb = CellField(fourblocks)
    pc = CellField(fourroots)
    initialize!(pa, pressure_function)
    initialize!(pb, pressure_function)
    initialize!(pc, pressure_function)
    ga = FaceVelocity(oneblock, (x, y, t, axis) -> 0.0)
    gb = FaceVelocity(fourblocks, (x, y, t, axis) -> 0.0)
    gc = FaceVelocity(fourroots, (x, y, t, axis) -> 0.0)
    gradient!(ga, pa, GradientOperator(pa))
    gradient!(gb, pb, GradientOperator(pb))
    gradient!(gc, pc, GradientOperator(pc))

    function values_by_coordinate(velocity)
        Dict(
            (Float64(face.center[1]), Float64(face.center[2]), face.axis) =>
                velocity.interior[index] for (index, face) in pairs(velocity.mesh.faces)
        )
    end
    @test values_by_coordinate(ga) == values_by_coordinate(gb)
    @test values_by_coordinate(ga) == values_by_coordinate(gc)
end

@testset "mixed-level gradient convergence" begin
    function gradient_error(block_size)
        mesh = BlockTreeMesh(
            block_size = (block_size, block_size), max_level = 2,
            refine = (key, origin, extent) ->
            key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
        )
        pressure = CellField(mesh)
        initialize!(pressure, (x, y) -> sinpi(2x) * cospi(2y))
        numerical = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
        gradient!(numerical, pressure, GradientOperator(pressure))

        squared_error = 0.0
        total_weight = 0.0
        for (index, face) in pairs(mesh.faces)
            x, y = face.center
            exact = face.axis == 1 ?
                2pi * cospi(2x) * cospi(2y) :
                -2pi * sinpi(2x) * sinpi(2y)
            displacement = wrapped_face_displacement(mesh, face)
            weight = face.area * abs(displacement[face.axis])
            squared_error += weight * (numerical.interior[index] - exact)^2
            total_weight += weight
        end
        return sqrt(squared_error / total_weight)
    end

    errors = gradient_error.((4, 8, 16))
    @test log2(errors[1] / errors[2]) > 1.4
    @test log2(errors[2] / errors[3]) > 1.4
end
