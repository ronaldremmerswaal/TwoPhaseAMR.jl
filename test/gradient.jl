using LinearAlgebra: Symmetric, eigvals, isposdef

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
    @test_throws ArgumentError GradientOperator(stretched_amr)
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
    pressure_function = (x, y) -> sinpi(2x) + 0.3cospi(4y)
    pa = CellField(oneblock)
    pb = CellField(fourblocks)
    initialize!(pa, pressure_function)
    initialize!(pb, pressure_function)
    ga = FaceVelocity(oneblock, (x, y, t, axis) -> 0.0)
    gb = FaceVelocity(fourblocks, (x, y, t, axis) -> 0.0)
    gradient!(ga, pa, GradientOperator(pa))
    gradient!(gb, pb, GradientOperator(pb))

    function values_by_coordinate(velocity)
        Dict(
            (Float64(face.center[1]), Float64(face.center[2]), face.axis) =>
                velocity.interior[index] for (index, face) in pairs(velocity.mesh.faces)
        )
    end
    @test values_by_coordinate(ga) == values_by_coordinate(gb)
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
