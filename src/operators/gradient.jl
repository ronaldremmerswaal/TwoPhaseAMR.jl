abstract type AbstractInverseFaceMetric{T} end

struct DiagonalInverseFaceMetric{T} <: AbstractInverseFaceMetric{T}
    diagonal::Vector{T}
end

struct SparseInverseFaceMetric{T, Ti} <: AbstractInverseFaceMetric{T}
    matrix::SparseMatrixCSC{T, Ti}
end

# Stored keys own their descriptor vectors, which are never mutated after
# insertion. Equality and hashing use descriptor contents rather than identity.
struct GradientMetricKey{T}
    descriptors::Vector{Tuple{Int, T, T, T}}
end

Base.:(==)(left::GradientMetricKey, right::GradientMetricKey) =
    left.descriptors == right.descriptors
Base.isequal(left::GradientMetricKey, right::GradientMetricKey) =
    isequal(left.descriptors, right.descriptors)
Base.hash(key::GradientMetricKey, seed::UInt) = hash(key.descriptors, seed)

"""Persistent local geometry templates and scratch shared across AMR topologies."""
mutable struct GradientConstructionCache{T}
    local_metrics::Dict{GradientMetricKey{T}, Matrix{T}}
    key_scratch::Vector{Tuple{Int, T, T, T}}
    hits::Int
    misses::Int
end

GradientConstructionCache(::Type{T}) where {T} =
    GradientConstructionCache{T}(
        Dict{GradientMetricKey{T}, Matrix{T}}(), Tuple{Int, T, T, T}[], 0, 0,
    )

"""
    GradientOperator(mesh, [T=eltype(mesh.origin)])
    GradientOperator(pressure::CellField)

Precompute the geometric Riesz map used by the pressure gradient. The primary
divergence remains the oriented sum of fluxes through owned leaf-face
segments. The gradient is its negative adjoint in the face inner product
induced by the inverse Riesz map.

On a conforming Cartesian mesh the map is diagonal and recovers the ordinary
MAC difference. At a coarse--fine interface it is a sparse, symmetric positive
definite vertex-patch operator. Its non-diagonal entries provide the wider
stencil needed to reproduce affine pressure gradients without modifying the
cellwise Gauss divergence.

At a physical boundary the normal pressure correction is homogeneous Neumann:
the boundary face gradient is exactly zero. Vertex patches cut by a wall use
the diagonal half-face metric, while complete interior patches retain the AMR
Riesz map. Thus the cell divergence remains the primal operator and the
pressure operator is still symmetric positive semidefinite. An operator owns
one scratch vector and must not be shared by concurrent kernels.
"""
struct GradientOperator{
        T, M <: BlockTreeMesh, R <: AbstractInverseFaceMetric{T}, C,
    }
    mesh::M
    inverse_metric::R
    pressure_jump::Vector{T}
    construction_cache::C
end

struct MetricFace{T}
    index::Int
    axis::Int
    area::T
    displacement::NTuple{2, T}
end

@inline function _wrapped_delta(delta, extent, periodic)
    periodic || return delta
    return delta - round(delta / extent) * extent
end

@inline function _face_displacement(mesh::BlockTreeMesh, face::InteriorFace)
    minus = cellcenter(mesh, face.minus.block, face.minus.i, face.minus.j)
    plus = cellcenter(mesh, face.plus.block, face.plus.i, face.plus.j)
    return (
        _wrapped_delta(plus[1] - minus[1], mesh.extent[1], mesh.periodic[1]),
        _wrapped_delta(plus[2] - minus[2], mesh.extent[2], mesh.periodic[2]),
    )
end

@inline function _vertex_key(
        mesh::BlockTreeMesh, face::InteriorFace,
        endpoint::Int
    )
    sx, sy = _finest_lattice_size(mesh)
    dx = mesh.extent[1] / sx
    dy = mesh.extent[2] / sy
    tangent_sign = endpoint == 1 ? -1 : 1
    if face.axis == 1
        i = round(Int, (face.center[1] - mesh.origin[1]) / dx)
        j = round(
            Int, (
                face.center[2] - mesh.origin[2] +
                    tangent_sign * face.area / 2
            ) / dy
        )
    else
        i = round(
            Int, (
                face.center[1] - mesh.origin[1] +
                    tangent_sign * face.area / 2
            ) / dx
        )
        j = round(Int, (face.center[2] - mesh.origin[2]) / dy)
    end
    mesh.periodic[1] && (i = mod(i, sx))
    mesh.periodic[2] && (j = mod(j, sy))
    return (i, j)
end

function _vertex_patches(mesh::BlockTreeMesh, ::Type{T}) where {T}
    patches = Dict{Tuple{Int, Int}, Vector{MetricFace{T}}}()
    for (index, face) in pairs(mesh.faces)
        displacement = _face_displacement(mesh, face)
        metric_face = MetricFace(
            index,
            face.axis,
            T(face.area),
            (T(displacement[1]), T(displacement[2])),
        )
        for endpoint in 1:2
            key = _vertex_key(mesh, face, endpoint)
            push!(get!(patches, key, MetricFace{T}[]), metric_face)
        end
    end
    return patches
end

function _local_inverse_metric(faces::Vector{MetricFace{T}}) where {T}
    number_of_faces = length(faces)
    number_of_faces > 2 ||
        throw(ArgumentError("a periodic vertex patch must contain at least three faces"))

    # For an affine pressure p, `moments * grad(p)` is the vector of
    # area-weighted pressure jumps across the faces in this patch. `normals`
    # partitions the desired coordinate-normal gradient equally between the
    # two endpoints of each face.
    moments = zeros(T, number_of_faces, 2)
    normals = zeros(T, number_of_faces, 2)
    for (row, face) in pairs(faces)
        moments[row, 1] = face.area * face.displacement[1]
        moments[row, 2] = face.area * face.displacement[2]
        normals[row, face.axis] = T(1 // 2)
    end

    moment_normal = transpose(moments) * normals
    # On a stretched coarse--fine patch, exact affine reproduction by this
    # vertex-local map is incompatible with symmetry: M'N is nonsymmetric.
    # Its symmetric part is the energy-compatible least-squares choice. It
    # retains a symmetric positive face metric, is exact on uniform stretched
    # grids, and converges at coarse--fine interfaces. For square cells the
    # tensor is already symmetric and the original affine-exact map is kept.
    moment_normal = (moment_normal + transpose(moment_normal)) / 2
    isposdef(Symmetric(moment_normal)) ||
        throw(ArgumentError("degenerate vertex patch in gradient construction"))

    consistent = normals * (moment_normal \ transpose(normals))
    gram = transpose(moments) * moments
    projector = Matrix{T}(I, number_of_faces, number_of_faces) -
        moments * (gram \ transpose(moments))

    # Match the ordinary diagonal MAC metric on a regular Cartesian patch and
    # control all modes not fixed by affine consistency.
    stabilization = tr(consistent) / (number_of_faces - 2)
    local_metric = consistent + stabilization * projector
    return (local_metric + transpose(local_metric)) / 2
end

@inline _metric_face_descriptor(face::MetricFace, scale) = (
    face.axis,
    face.area / scale,
    face.displacement[1] / scale,
    face.displacement[2] / scale,
)

function _cached_local_inverse_metric!(
        cache::GradientConstructionCache{T}, faces::Vector{MetricFace{T}},
    ) where {T}
    scale = maximum(
        max(face.area, abs(face.displacement[1]), abs(face.displacement[2]))
            for face in faces
    )
    scale > zero(T) || throw(ArgumentError("degenerate vertex-patch scale"))
    # A stable geometric ordering makes the template independent of global face
    # numbering. Equal descriptors are interchangeable in the local metric.
    sort!(faces; by = face -> _metric_face_descriptor(face, scale))
    descriptors = cache.key_scratch
    resize!(descriptors, length(faces))
    for index in eachindex(faces)
        @inbounds descriptors[index] = _metric_face_descriptor(faces[index], scale)
    end
    key = GradientMetricKey(descriptors)
    normalized = get(cache.local_metrics, key, nothing)
    if isnothing(normalized)
        cache.misses += 1
        normalized = _local_inverse_metric(faces) .* (scale * scale)
        cache.local_metrics[GradientMetricKey(copy(descriptors))] = normalized
    else
        cache.hits += 1
    end
    return normalized ./ (scale * scale)
end

@inline function _physical_vertex(mesh::BlockTreeMesh, key::Tuple{Int, Int})
    sx, sy = _finest_lattice_size(mesh)
    return (!mesh.periodic[1] && (key[1] == 0 || key[1] == sx)) ||
        (!mesh.periodic[2] && (key[2] == 0 || key[2] == sy))
end

function _append_boundary_patch_metric!(
        rows, columns, coefficients,
        faces::Vector{MetricFace{T}}
    ) where {T}
    # Every interior face belongs to two endpoint patches. A wall endpoint is
    # closed with one half of the ordinary two-point face metric. This is the
    # reflected-pressure (zero normal jump) closure and remains SPD even at a
    # coarse--fine corner where a complete affine patch is unavailable.
    for face in faces
        normal_distance = abs(face.displacement[face.axis])
        normal_distance > zero(T) ||
            throw(ArgumentError("degenerate face beside a physical boundary"))
        push!(rows, face.index)
        push!(columns, face.index)
        push!(coefficients, inv(T(2) * face.area * normal_distance))
    end
    return nothing
end

function _assemble_inverse_metric(
        mesh::BlockTreeMesh, ::Type{T}, cache::GradientConstructionCache{T},
    ) where {T}
    number_of_faces = length(mesh.faces)
    rows = Int[]
    columns = Int[]
    coefficients = T[]
    patches = _vertex_patches(mesh, T)
    for key in sort!(collect(keys(patches)))
        faces = patches[key]
        if _physical_vertex(mesh, key) || length(faces) <= 2
            _append_boundary_patch_metric!(rows, columns, coefficients, faces)
            continue
        end
        local_metric = _cached_local_inverse_metric!(cache, faces)
        for (local_column, column_face) in pairs(faces)
            for (local_row, row_face) in pairs(faces)
                push!(rows, row_face.index)
                push!(columns, column_face.index)
                push!(coefficients, local_metric[local_row, local_column])
            end
        end
    end
    inverse_metric = sparse(
        rows, columns, coefficients,
        number_of_faces, number_of_faces
    )
    dropzeros!(inverse_metric)
    return inverse_metric
end

function _is_diagonal(matrix::SparseMatrixCSC)
    for column in axes(matrix, 2)
        for index in nzrange(matrix, column)
            row = rowvals(matrix)[index]
            row == column || iszero(nonzeros(matrix)[index]) || return false
        end
    end
    return true
end

function _metric_storage(matrix::SparseMatrixCSC{T, Ti}) where {T, Ti}
    if _is_diagonal(matrix)
        diagonal = zeros(T, size(matrix, 1))
        for column in axes(matrix, 2)
            for index in nzrange(matrix, column)
                rowvals(matrix)[index] == column || continue
                diagonal[column] = nonzeros(matrix)[index]
            end
        end
        all(>(zero(T)), diagonal) ||
            throw(ArgumentError("gradient Riesz map has a non-positive diagonal"))
        return DiagonalInverseFaceMetric(diagonal)
    end
    return SparseInverseFaceMetric(matrix)
end

function GradientOperator(
        mesh::M, ::Type{T} = eltype(mesh.origin);
        construction_cache = GradientConstructionCache(T),
    ) where
    {M <: BlockTreeMesh, T <: _SupportedReal}
    construction_cache isa GradientConstructionCache{T} || throw(
        ArgumentError("gradient construction cache has the wrong scalar type"),
    )
    inverse_metric = _assemble_inverse_metric(mesh, T, construction_cache)
    metric = _metric_storage(inverse_metric)
    return GradientOperator{T, M, typeof(metric), typeof(construction_cache)}(
        mesh, metric, zeros(T, length(mesh.faces)), construction_cache,
    )
end

GradientOperator(pressure::CellField{T}) where {T <: _SupportedReal} =
    GradientOperator(pressure.mesh, T)

@inline function _mul_inverse_metric!(
        destination::AbstractVector,
        metric::DiagonalInverseFaceMetric,
        source::AbstractVector
    )
    @_parallel for index in 1:length(destination)
        @inbounds destination[index] = metric.diagonal[index] * source[index]
    end
    return destination
end

@inline function _mul_inverse_metric!(
        destination::AbstractVector,
        metric::SparseInverseFaceMetric,
        source::AbstractVector
    )
    _parallel_csr_row_mul!(destination, metric.matrix, source)
    return destination
end

function _pressure_jump!(jump, pressure::CellField)
    faces = pressure.mesh.faces
    @_parallel for index in 1:length(faces)
        face = faces[index]
        @inbounds jump[index] = face.area *
            (_get(pressure, face.plus) - _get(pressure, face.minus))
    end
    return jump
end

"""
    gradient!(destination, pressure, operator)

Apply the pressure gradient derived from the primary finite-volume divergence.
The result is a coordinate-normal value on every owned interior face. Physical
boundary values are zero, imposing the homogeneous Neumann pressure-correction
condition associated with a prescribed no-permeation velocity.
"""
function gradient!(
        destination::FaceVelocity{T}, pressure::CellField{T},
        operator::GradientOperator{T}
    ) where {T}
    destination.mesh === pressure.mesh === operator.mesh ||
        throw(ArgumentError("destination, pressure, and operator must share a mesh"))
    _pressure_jump!(operator.pressure_jump, pressure)
    _mul_inverse_metric!(
        destination.interior, operator.inverse_metric,
        operator.pressure_jump
    )
    fill!(destination.boundary, zero(T))
    return destination
end
