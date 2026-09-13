"""
    harmonic_face_coefficients(coefficient::CellField)

Interpolate a positive cell-centred diffusion coefficient to owned interior
faces with the distance-weighted harmonic mean.  This is the appropriate
default for discontinuous material coefficients: the normal flux is continuous
and the two half-cell resistances are added in series.
"""
function harmonic_face_coefficients(coefficient::CellField{T}) where {T}
    mesh = coefficient.mesh
    result = Vector{T}(undef, length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        minus_value = _get(coefficient, face.minus)
        plus_value = _get(coefficient, face.plus)
        minus_value > zero(T) && plus_value > zero(T) ||
            throw(ArgumentError("cell coefficients must be strictly positive"))
        minus_distance = mesh.blocks[face.minus.block].spacing[face.axis] / 2
        plus_distance = mesh.blocks[face.plus.block].spacing[face.axis] / 2
        @inbounds result[index] = (minus_distance + plus_distance) /
            (minus_distance / minus_value + plus_distance / plus_value)
    end
    return result
end

@inline function _elliptic_cell_index(ref::CellRef, nx::Int, ny::Int)
    return (ref.block - 1) * nx * ny + (ref.j - 1) * nx + ref.i
end

function _cell_references(mesh::BlockTreeMesh)
    nx, ny = mesh.block_size
    return [CellRef(block, i, j)
            for block in eachindex(mesh.blocks) for j in 1:ny for i in 1:nx]
end

function _cell_volumes(mesh::BlockTreeMesh, ::Type{T}) where {T}
    nx, ny = mesh.block_size
    result = Vector{T}(undef, ncells(mesh))
    for (block_index, block) in pairs(mesh.blocks)
        offset = (block_index - 1) * nx * ny
        fill!(@view(result[(offset + 1):(offset + nx * ny)]), T(block.volume))
    end
    return result
end

function _face_coefficients(mesh::BlockTreeMesh, coefficient, ::Type{T}) where {T}
    number_of_faces = length(mesh.faces)
    values = if coefficient isa Number
        fill(T(coefficient), number_of_faces)
    elseif coefficient isa AbstractVector
        length(coefficient) == number_of_faces ||
            throw(DimensionMismatch("one coefficient is required per interior face"))
        T.(coefficient)
    elseif coefficient isa FaceVelocity
        coefficient.mesh === mesh ||
            throw(ArgumentError("face coefficient and operator must share a mesh"))
        T.(coefficient.interior)
    elseif applicable(coefficient, mesh.faces[1].center[1],
                      mesh.faces[1].center[2], mesh.faces[1].axis)
        [T(coefficient(face.center[1], face.center[2], face.axis))
         for face in mesh.faces]
    else
        throw(ArgumentError(
            "face_coefficient must be a number, vector, FaceVelocity, or (x, y, axis) function"))
    end
    all(isfinite, values) || throw(ArgumentError("face coefficients must be finite"))
    all(>(zero(T)), values) ||
        throw(ArgumentError("face coefficients must be strictly positive"))
    return values
end

function _weighted_inverse_metric(metric::DiagonalInverseFaceMetric{T},
                                  coefficient::Vector{T}) where {T}
    return spdiagm(0 => coefficient .* metric.diagonal)
end

function _weighted_inverse_metric(metric::SparseInverseFaceMetric{T},
                                  coefficient::Vector{T}) where {T}
    square_root = spdiagm(0 => sqrt.(coefficient))
    return square_root * metric.matrix * square_root
end

function _integrated_stiffness(mesh::BlockTreeMesh,
                               gradient_operator::GradientOperator{T},
                               coefficient::Vector{T}) where {T}
    nx, ny = mesh.block_size
    number_of_faces = length(mesh.faces)
    rows = Vector{Int}(undef, 2number_of_faces)
    columns = Vector{Int}(undef, 2number_of_faces)
    values = Vector{T}(undef, 2number_of_faces)
    for (face_index, face) in pairs(mesh.faces)
        entry = 2face_index - 1
        rows[entry] = face_index
        columns[entry] = _elliptic_cell_index(face.minus, nx, ny)
        values[entry] = -T(face.area)
        rows[entry + 1] = face_index
        columns[entry + 1] = _elliptic_cell_index(face.plus, nx, ny)
        values[entry + 1] = T(face.area)
    end
    jump = sparse(rows, columns, values, number_of_faces, ncells(mesh))
    face_metric = _weighted_inverse_metric(
        gradient_operator.inverse_metric, coefficient)
    stiffness = transpose(jump) * face_metric * jump
    # Sparse products can accumulate the two triangles in a different order.
    # Explicit symmetrisation makes the algebraic contract exact to roundoff.
    stiffness = (stiffness + transpose(stiffness)) / 2
    dropzeros!(stiffness)
    return stiffness
end

"""
    EllipticOperator(mesh; reaction=0, diffusion=1, face_coefficient=1)

Construct the positive cell-centred operator

`L(u) = reaction*u - diffusion*div(face_coefficient*grad(u))`.

The matrix is stored in integrated finite-volume form,
`reaction*M + diffusion*S`, where `M` is the diagonal cell-volume matrix and
`S` is symmetric positive semidefinite.  Keeping the reaction and diffusion
parts separate makes the same object usable for periodic pressure Poisson
problems (`reaction=0`) and later implicit diffusion solves (`reaction>0`).

At a mixed-level interface the inverse face metric is non-diagonal.  Variable
coefficients therefore enter through the symmetric product
`sqrt(kappa)*K*sqrt(kappa)`; this preserves gradient/divergence duality and
reduces to ordinary face multiplication wherever `K` is diagonal.
"""
struct EllipticOperator{T,M<:BlockTreeMesh,G<:GradientOperator{T}} <:
       AbstractEllipticOperator{T}
    mesh::M
    gradient_operator::G
    face_coefficient::Vector{T}
    square_root_face_coefficient::Vector{T}
    stiffness::SparseMatrixCSC{T,Int}
    matrix::SparseMatrixCSC{T,Int}
    volumes::Vector{T}
    reaction::T
    diffusion::T
    references::Vector{CellRef}
end

function EllipticOperator(mesh::M; reaction=0, diffusion=1,
                          face_coefficient=1,
                          T=eltype(mesh.origin)) where {M<:BlockTreeMesh}
    T <: Union{Float32,Float64} ||
        throw(ArgumentError("elliptic operators require Float32 or Float64"))
    reaction_value = T(reaction)
    diffusion_value = T(diffusion)
    reaction_value >= zero(T) ||
        throw(ArgumentError("reaction must be nonnegative"))
    diffusion_value > zero(T) ||
        throw(ArgumentError("diffusion must be strictly positive"))
    gradient_operator = GradientOperator(mesh, T)
    coefficients = _face_coefficients(mesh, face_coefficient, T)
    stiffness = _integrated_stiffness(mesh, gradient_operator, coefficients)
    volumes = _cell_volumes(mesh, T)
    matrix = diffusion_value * stiffness
    if !iszero(reaction_value)
        matrix += spdiagm(0 => reaction_value .* volumes)
    end
    dropzeros!(matrix)
    return EllipticOperator{T,M,typeof(gradient_operator)}(
        mesh, gradient_operator, coefficients, sqrt.(coefficients),
        stiffness, matrix, volumes,
        reaction_value, diffusion_value, _cell_references(mesh))
end

"""Construct the periodic positive Poisson operator `-div(kappa*grad(u))`."""
PoissonOperator(mesh::BlockTreeMesh; face_coefficient=1,
                T=eltype(mesh.origin)) =
    EllipticOperator(mesh; reaction=0, diffusion=1,
                     face_coefficient, T)

struct EllipticWorkspace{T}
    input::Vector{T}
    output::Vector{T}
    face_input::Vector{T}
    face_output::Vector{T}
end

EllipticWorkspace(operator::EllipticOperator{T}) where {T} =
    EllipticWorkspace(zeros(T, length(operator.volumes)),
                      zeros(T, length(operator.volumes)),
                      zeros(T, length(operator.face_coefficient)),
                      zeros(T, length(operator.face_coefficient)))

function _copy_from_field!(destination::AbstractVector, source::CellField,
                           references::Vector{CellRef})
    length(destination) == length(references) ||
        throw(DimensionMismatch("field and vector sizes differ"))
    for (index, reference) in pairs(references)
        @inbounds destination[index] = _get(source, reference)
    end
    return destination
end

function _copy_to_field!(destination::CellField, source::AbstractVector,
                         references::Vector{CellRef})
    length(source) == length(references) ||
        throw(DimensionMismatch("field and vector sizes differ"))
    for (index, reference) in pairs(references)
        _set!(destination, @inbounds(source[index]), reference)
    end
    return destination
end

"""
    apply!(destination, operator, source, workspace)

Apply the strong-form elliptic operator to cell values without allocating.
"""
function apply!(destination::CellField{T}, operator::EllipticOperator{T},
                source::CellField{T}, workspace::EllipticWorkspace{T}) where {T}
    destination.mesh === source.mesh === operator.mesh ||
        throw(ArgumentError("fields and elliptic operator must share a mesh"))
    _copy_from_field!(workspace.input, source, operator.references)
    mul!(workspace.output, operator.matrix, workspace.input)
    for index in eachindex(workspace.output)
        @inbounds workspace.output[index] /= operator.volumes[index]
    end
    _copy_to_field!(destination, workspace.output, operator.references)
    return destination
end

"""
    weighted_gradient!(destination, pressure, operator, workspace)

Apply the coefficient-weighted face gradient used inside `operator`.  Calling
this routine followed by `-divergence!` is algebraically identical to the
diffusion part of `apply!`, including the non-diagonal AMR interface metric.
"""
function weighted_gradient!(destination::FaceVelocity{T},
                            pressure::CellField{T},
                            operator::EllipticOperator{T},
                            workspace::EllipticWorkspace{T}) where {T}
    destination.mesh === pressure.mesh === operator.mesh ||
        throw(ArgumentError("fields and elliptic operator must share a mesh"))
    _pressure_jump!(operator.gradient_operator.pressure_jump, pressure)
    for index in eachindex(workspace.face_input)
        @inbounds workspace.face_input[index] =
            operator.square_root_face_coefficient[index] *
            operator.gradient_operator.pressure_jump[index]
    end
    _mul_inverse_metric!(workspace.face_output,
                         operator.gradient_operator.inverse_metric,
                         workspace.face_input)
    for index in eachindex(destination.interior)
        @inbounds destination.interior[index] =
            operator.square_root_face_coefficient[index] * workspace.face_output[index]
    end
    fill!(destination.boundary, zero(T))
    return destination
end

struct StaggeredEllipticComponent{Axis,T}
    matrix::SparseMatrixCSC{T,Int}
    volumes::Vector{T}
    local_to_global::Vector{Int}
end

"""
    StaggeredEllipticOperator(diffusion_operator;
                              reaction=1, diffusion=1)

Construct the integrated staggered elliptic problem

`reaction*u - diffusion*Δ_h*u`.

The same abstract elliptic contract as the cell-centred pressure operator is
used, but with dual volumes and one matrix per velocity component. For an
implicit midpoint viscous step, `diffusion = dt*viscosity/2` and `reaction=1`.
Nonhomogeneous wall data belongs on the right-hand side and is not embedded in
these homogeneous matrices.
"""
struct StaggeredEllipticOperator{T,M<:BlockTreeMesh,X,Y,D} <:
       AbstractEllipticOperator{T}
    mesh::M
    x::X
    y::Y
    diffusion_operator::D
    reaction::T
    diffusion::T
end

function _staggered_elliptic_component(
        component::StaggeredDiffusionComponent{Axis,T}, reaction,
        diffusion) where {Axis,T}
    mass = spdiagm(0 => component.volumes)
    matrix = mass * (reaction *
        spdiagm(0 => ones(T, length(component.volumes))) -
        diffusion * component.laplacian)
    dropzeros!(matrix)
    return StaggeredEllipticComponent{Axis,T}(
        sparse(matrix), component.volumes, component.local_to_global)
end

"""Construct the symmetric conservative shifted matrix used for preconditioning."""
function _staggered_stable_matrix(
        component::StaggeredDiffusionComponent{Axis,T}, reaction,
        diffusion) where {Axis,T}
    matrix = diffusion * component.stiffness +
             spdiagm(0 => reaction .* component.volumes)
    dropzeros!(matrix)
    return matrix
end

function StaggeredEllipticOperator(
        diffusion_operator::StaggeredDiffusionOperator{T};
        reaction=1, diffusion=1) where {T}
    reaction_value = T(reaction)
    diffusion_value = T(diffusion)
    reaction_value > zero(T) ||
        throw(ArgumentError("staggered elliptic reaction must be positive"))
    diffusion_value >= zero(T) ||
        throw(ArgumentError("staggered elliptic diffusion must be nonnegative"))
    x = _staggered_elliptic_component(
        diffusion_operator.x, reaction_value, diffusion_value)
    y = _staggered_elliptic_component(
        diffusion_operator.y, reaction_value, diffusion_value)
    return StaggeredEllipticOperator{
        T,typeof(diffusion_operator.mesh),typeof(x),typeof(y),
        typeof(diffusion_operator)}(
        diffusion_operator.mesh, x, y, diffusion_operator,
        reaction_value, diffusion_value)
end
