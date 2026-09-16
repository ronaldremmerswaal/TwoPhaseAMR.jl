"""Coupled staggered `div(2μD(u))` with cell-centred normal stresses and vertex
shear stresses. The integrated stiffness is `E'WE`, so viscosity dissipates
kinetic energy for positive μ. Density enters the time-step mass matrix.
"""
mutable struct VariableStaggeredDiffusionOperator{T, M, B}
    mesh::M
    stiffness::SparseMatrixCSC{T, Int}
    volumes::Vector{T}
    boundary::B
    strain::SparseMatrixCSC{T, Int}
    stress_interpolation::SparseMatrixCSC{T, Int}
    stress_measures::Vector{T}
    stress_weights::Vector{T}
    cell_work::Vector{T}
    stress_work::Vector{T}
    stiffness_pointers::Vector{Int}
    stiffness_stresses::Vector{Int}
    stiffness_factors::Vector{T}
end

function _stiffness_refresh_map(strain, stiffness)
    T = eltype(stiffness)
    entries = [Tuple{Int, T}[] for _ in axes(strain, 1)]
    strain_rows = rowvals(strain)
    strain_values = nonzeros(strain)
    for column in axes(strain, 2), pointer in nzrange(strain, column)
        row = strain_rows[pointer]
        push!(entries[row], (column, strain_values[pointer]))
    end
    stiffness_rows = rowvals(stiffness)
    pointers = Int[]
    stresses = Int[]
    factors = T[]
    for (stress, row_entries) in pairs(entries)
        for (column, right_value) in row_entries
            for (row, left_value) in row_entries
                pointer = 0
                for candidate in nzrange(stiffness, column)
                    if stiffness_rows[candidate] == row
                        pointer = candidate
                        break
                    end
                end
                iszero(pointer) && error(
                    "stress refresh changed diffusion sparsity",
                )
                push!(pointers, pointer)
                push!(stresses, stress)
                push!(factors, left_value * right_value)
            end
        end
    end
    return pointers, stresses, factors
end

@inline function _phase_dynamic_viscosity(c, viscosity::NTuple{2})
    liquid, gas = viscosity
    return exp(muladd(c, log(liquid) - log(gas), log(gas)))
end

"""
    dynamic_viscosity(interface; viscosity)

Return the cell-centred dynamic viscosity used by the one-velocity diffusion
operator. Following the thesis, the coefficient is the phase-fraction-
weighted geometric mean

`mu = mu_liquid^c * mu_gas^(1-c)`.

At shear-stress vertices the operator volume-interpolates `c` before applying
the same formula. Equivalently, it geometrically averages this returned field,
because `log(mu)` is affine in `c`.
"""
function dynamic_viscosity(
        field::InterfaceField{T}; viscosity::NTuple{2}
    ) where {T}
    phases = T.(viscosity)
    all(mu -> isfinite(mu) && mu > zero(T), phases) ||
        throw(
        ArgumentError(
            "phase dynamic viscosities must be finite and positive"
        )
    )
    result = CellField(field.mesh, T)
    for ref in _cell_references(field.mesh)
        _set!(
            result,
            _phase_dynamic_viscosity(_get(field.fraction, ref), phases),
            ref
        )
    end
    return result
end

function _stress_vertices(mesh, ::Type{T}) where {T}
    # Cell-corner quadrature weights partition the domain volume exactly.
    lattice = mesh.block_size .* (1 << mesh.max_level)
    vertices = Dict{NTuple{2, Int}, Tuple{T, Vector{Tuple{CellRef, T}}}}()
    nx, ny = mesh.block_size
    for ref in _cell_references(mesh)
        block = mesh.blocks[ref.block]
        x, y = cellcenter(mesh, ref.block, ref.i, ref.j)
        hx, hy = block.spacing
        for sx in (-1, 1),sy in (-1, 1)
            key = (
                round(Int, (x + sx * hx / 2 - mesh.origin[1]) / mesh.extent[1] * lattice[1]),
                round(Int, (y + sy * hy / 2 - mesh.origin[2]) / mesh.extent[2] * lattice[2]),
            )
            key = ntuple(a -> mesh.periodic[a] ? mod(key[a], lattice[a]) : key[a], 2)
            volume, refs = get!(vertices, key) do
                (zero(T), Tuple{CellRef, T}[])
            end
            weight = T(block.volume / 4)
            push!(refs, (ref, weight))
            vertices[key] = (volume + weight, refs)
        end
    end
    return vertices, lattice
end

struct DiffusionFaceSearch{T}
    interior_indices::NTuple{2, Vector{Int}}
    interior_x::NTuple{2, Vector{T}}
    boundary_indices::NTuple{2, Vector{Int}}
    boundary_x::NTuple{2, Vector{T}}
end

function _sorted_face_axis(mesh, faces, axis, ::Type{T}) where {T}
    indices = [index for (index, face) in pairs(faces) if face.axis == axis]
    sort!(indices; by = index -> (faces[index].center[1], index))
    coordinates = T[faces[index].center[1] for index in indices]
    return indices, coordinates
end

function DiffusionFaceSearch(mesh, ::Type{T}) where {T}
    interior = ntuple(
        axis -> _sorted_face_axis(mesh, mesh.faces, axis, T), 2
    )
    boundary = ntuple(
        axis -> _sorted_face_axis(mesh, mesh.boundary_faces, axis, T), 2
    )
    return DiffusionFaceSearch(
        (interior[1][1], interior[2][1]),
        (interior[1][2], interior[2][2]),
        (boundary[1][1], boundary[2][1]),
        (boundary[1][2], boundary[2][2]),
    )
end

function _indices_in_x_window!(
        candidates, indices, coordinates, centre, radius
    )
    first = searchsortedfirst(coordinates, centre - radius)
    last = searchsortedlast(coordinates, centre + radius)
    first <= last && append!(candidates, @view(indices[first:last]))
    return candidates
end

function _nearby_diffusion_faces!(
        points::Vector{Tuple{Int, T, T}}, candidates,
        search::DiffusionFaceSearch{T}, mesh, point, component, scale
    ) where {T}
    empty!(points)
    empty!(candidates)
    radius = 3scale
    indices = search.interior_indices[component]
    coordinates = search.interior_x[component]
    _indices_in_x_window!(candidates, indices, coordinates, point[1], radius)
    if mesh.periodic[1]
        _indices_in_x_window!(
            candidates, indices, coordinates,
            point[1] - mesh.extent[1], radius,
        )
        _indices_in_x_window!(
            candidates, indices, coordinates,
            point[1] + mesh.extent[1], radius,
        )
    end
    sort!(candidates; alg = QuickSort)
    unique!(candidates)
    for index in candidates
        face = mesh.faces[index]
        dx = T(_wrapped_delta(
                face.center[1] - point[1], mesh.extent[1], mesh.periodic[1]
            ))
        dy = T(_wrapped_delta(
                face.center[2] - point[2], mesh.extent[2], mesh.periodic[2]
            ))
        hypot(dx, dy) <= radius && push!(points, (index, dx, dy))
    end

    empty!(candidates)
    indices = search.boundary_indices[component]
    coordinates = search.boundary_x[component]
    _indices_in_x_window!(candidates, indices, coordinates, point[1], radius)
    sort!(candidates; alg = QuickSort)
    for index in candidates
        face = mesh.boundary_faces[index]
        dx = T(face.center[1] - point[1])
        dy = T(face.center[2] - point[2])
        hypot(dx, dy) <= radius && push!(points, (0, dx, dy))
    end
    return points
end

function _vertex_derivative(
        mesh, point, component, scale, boundary, ::Type{T}, search,
        points, candidates, ids, stencil
    ) where {T}
    axis = 3 - component
    tolerance = 128eps(T) * maximum(mesh.extent)
    _nearby_diffusion_faces!(
        points, candidates, search, mesh, point, component, scale,
    )
    empty!(ids)
    empty!(stencil)
    # Standard MAC shear derivative on regular lines (including wall closure).
    wall_low = !mesh.periodic[axis] && abs(point[axis] - mesh.origin[axis]) <= tolerance
    wall_high = !mesh.periodic[axis] && abs(point[axis] - mesh.origin[axis] - mesh.extent[axis]) <= tolerance
    if (wall_low || wall_high) && boundary isa FreeSlip
        return ids, stencil
    end
    low = high = (0, zero(T), zero(T))
    has_low = wall_low
    has_high = wall_high
    for candidate in points
        abs(candidate[component + 1]) <= tolerance || continue
        coordinate = candidate[axis + 1]
        if coordinate < -tolerance &&
                (!has_low || coordinate > low[axis + 1])
            low = candidate
            has_low = true
        elseif coordinate > tolerance &&
                (!has_high || coordinate < high[axis + 1])
            high = candidate
            has_high = true
        end
    end
    if wall_low
        low = (0, zero(T), zero(T))
    elseif wall_high
        high = (0, zero(T), zero(T))
    end
    if has_low && has_high
        distance = high[axis + 1] - low[axis + 1]
        low[1] > 0 && (push!(ids, low[1]); push!(stencil, -inv(distance)))
        high[1] > 0 && (push!(ids, high[1]); push!(stencil, inv(distance)))
        return ids, stencil
    end
    # At hanging vertices a collocated pair can be absent. A weighted quadratic
    # fit reproduces affine strain while using the same owned face unknowns.
    sort!(points; by = p -> p[2]^2 + p[3]^2)
    length(points) > 16 && resize!(points, 16)
    for degree in (2, 1)
        columns = degree == 2 ? 6 : 3
        length(points) >= columns || continue
        matrix = Matrix{T}(undef, length(points), columns)
        weights = Vector{T}(undef, length(points))
        for (i, p) in pairs(points)
            x, y = p[2] / scale, p[3] / scale
            weights[i] = inv(1 + x * x + y * y)
            row = degree == 2 ? (one(T), x, y, x * x, x * y, y * y) : (one(T), x, y)
            for j in 1:columns
                matrix[i, j] = weights[i] * row[j]
            end
        end
        singular = svdvals(matrix)
        minimum(singular) > 1.0e-10maximum(singular) || continue
        target = zeros(T, columns);target[axis + 1] = inv(scale)
        # Minimal-norm solution of the polynomial reproduction constraints.
        coefficients = transpose(matrix) \ target
        for (i, p) in pairs(points)
            p[1] > 0 || continue
            push!(ids, p[1]);push!(stencil, weights[i] * coefficients[i])
        end
        return ids, stencil
    end
    throw(ArgumentError("insufficient faces for staggered shear reconstruction"))
end

function _variable_staggered_diffusion(mesh, viscosity::CellField, boundary, ::Type{T}) where {T}
    viscosity.mesh === mesh || throw(ArgumentError("viscosity field must share the operator mesh"))
    boundary isa FreeSlip || (boundary isa NoSlip && boundary.velocity isa StationaryWall) ||
        throw(ArgumentError("variable stress diffusion currently supports stationary NoSlip or FreeSlip walls"))
    all(c -> isfinite(c)&&c > 0, (v for b in eachindex(mesh.blocks) for v in interior(viscosity, b))) ||
        throw(ArgumentError("dynamic viscosity must be finite and positive"))
    refs = _cell_references(mesh);nx, ny = mesh.block_size
    rows = Int[];cols = Int[];vals = T[]
    interpolation_rows = Int[]
    interpolation_columns = Int[]
    interpolation_values = T[]
    measures = T[]
    face_search = DiffusionFaceSearch(mesh, T)
    derivative_points = Tuple{Int, T, T}[]
    derivative_candidates = Int[]
    derivative_indices = Int[]
    derivative_coefficients = T[]
    for ref in refs,axis in 1:2
        row = length(measures) + 1
        push!(measures, 2 * mesh.blocks[ref.block].volume)
        push!(interpolation_rows, row)
        push!(interpolation_columns, _cell_index(ref, nx, ny))
        push!(interpolation_values, one(T))
    end
    for (i, face) in pairs(mesh.faces)
        for (ref, sign) in ((face.minus, 1), (face.plus, -1))
            row = 2 * (_cell_index(ref, nx, ny) - 1) + face.axis
            push!(rows, row);push!(cols, i)
            push!(vals, sign * face.area / mesh.blocks[ref.block].volume)
        end
    end
    vertices, lattice = _stress_vertices(mesh, T)
    for key in sort!(collect(keys(vertices)))
        volume, cells = vertices[key]
        point = ntuple(a -> mesh.origin[a] + key[a] * mesh.extent[a] / lattice[a], 2)
        push!(measures, volume);row = length(measures)
        for (ref, weight) in cells
            push!(interpolation_rows, row)
            push!(interpolation_columns, _cell_index(ref, nx, ny))
            push!(interpolation_values, weight / volume)
        end
        scale = maximum(maximum(mesh.blocks[r.block].spacing) for (r, w) in cells)
        for component in 1:2
            indices, coefficients = _vertex_derivative(
                mesh, point, component, scale, boundary, T,
                face_search, derivative_points, derivative_candidates,
                derivative_indices, derivative_coefficients,
            )
            append!(rows, fill(row, length(indices)));append!(cols, indices);append!(vals, coefficients)
        end
    end
    strain = sparse(rows, cols, vals, length(measures), length(mesh.faces))
    interpolation = sparse(
        interpolation_rows, interpolation_columns, interpolation_values,
        length(measures), length(refs)
    )
    cell_work = T[log(_get(viscosity, ref)) for ref in refs]
    stress_work = interpolation * cell_work
    weights = measures .* exp.(stress_work)
    stiffness = transpose(strain) * spdiagm(0 => weights) * strain
    stiffness_pointers, stiffness_stresses, stiffness_factors =
        _stiffness_refresh_map(strain, stiffness)
    volumes = T[face.area * (mesh.blocks[face.minus.block].spacing[face.axis] + mesh.blocks[face.plus.block].spacing[face.axis]) / 2 for face in mesh.faces]
    return VariableStaggeredDiffusionOperator(
        mesh, stiffness, volumes, boundary, strain, interpolation,
        measures, weights, cell_work, stress_work,
        stiffness_pointers, stiffness_stresses, stiffness_factors,
    )
end

"""Refresh the stress-location geometric mean without rebuilding geometric
strain and interpolation stencils. This is used after every VOF update."""
function _update_dynamic_viscosity!(
        operator::VariableStaggeredDiffusionOperator{T},
        field::InterfaceField{T}, viscosity::NTuple{2}
    ) where {T}
    operator.mesh === field.mesh ||
        throw(ArgumentError("interface and diffusion operator must share a mesh"))
    phases = T.(viscosity)
    all(mu -> isfinite(mu) && mu > zero(T), phases) ||
        throw(
        ArgumentError(
            "phase dynamic viscosities must be finite and positive"
        )
    )
    log_gas = log(phases[2])
    log_jump = log(phases[1]) - log_gas
    for (index, ref) in pairs(_cell_references(field.mesh))
        @inbounds operator.cell_work[index] =
            muladd(_get(field.fraction, ref), log_jump, log_gas)
    end
    mul!(
        operator.stress_work, operator.stress_interpolation,
        operator.cell_work
    )
    for index in eachindex(operator.stress_weights)
        @inbounds operator.stress_weights[index] =
            operator.stress_measures[index] * exp(operator.stress_work[index])
    end
    stiffness_values = nonzeros(operator.stiffness)
    fill!(stiffness_values, zero(T))
    for index in eachindex(operator.stiffness_pointers)
        pointer = @inbounds operator.stiffness_pointers[index]
        stress = @inbounds operator.stiffness_stresses[index]
        factor = @inbounds operator.stiffness_factors[index]
        @inbounds stiffness_values[pointer] +=
            operator.stress_weights[stress] * factor
    end
    return operator
end

function StaggeredDiffusionOperator(field::InterfaceField; viscosity::NTuple{2}, boundary = NoSlip())
    mu = dynamic_viscosity(field; viscosity)
    return StaggeredDiffusionOperator(field.mesh; viscosity = mu, boundary)
end
function diffusion!(
        destination::FaceVelocity, velocity::FaceVelocity,
        operator::VariableStaggeredDiffusionOperator; time = 0
    )
    destination.mesh === velocity.mesh === operator.mesh || throw(ArgumentError("diffusion arguments must share a mesh"))
    destination === velocity && throw(ArgumentError("diffusion output must not alias input"))
    mul!(destination.interior, operator.stiffness, velocity.interior)
    for index in eachindex(destination.interior)
        @inbounds destination.interior[index] =
            -destination.interior[index] / operator.volumes[index]
    end
    fill!(destination.boundary, zero(eltype(destination)))
    return destination
end

"""Block geometric preconditioner for the coupled variable-stress system.

The full Krylov matrix retains the cross-component shear terms. Each block
V-cycle is built from the corresponding principal submatrix, including the
actual staggered density mass and current viscosity coefficient.
"""
struct VariableStaggeredGeometricPreconditioner{T, X, Y}
    x::X
    y::Y
    x_indices::Vector{Int}
    y_indices::Vector{Int}
    x_rhs::Vector{T}
    y_rhs::Vector{T}
    x_solution::Vector{T}
    y_solution::Vector{T}
end

function ldiv!(
        destination::AbstractVector,
        preconditioner::VariableStaggeredGeometricPreconditioner,
        right_hand_side::AbstractVector
    )
    for (local_index, global_index) in pairs(preconditioner.x_indices)
        @inbounds preconditioner.x_rhs[local_index] =
            right_hand_side[global_index]
    end
    for (local_index, global_index) in pairs(preconditioner.y_indices)
        @inbounds preconditioner.y_rhs[local_index] =
            right_hand_side[global_index]
    end
    ldiv!(
        preconditioner.x_solution, preconditioner.x,
        preconditioner.x_rhs
    )
    ldiv!(
        preconditioner.y_solution, preconditioner.y,
        preconditioner.y_rhs
    )
    for (local_index, global_index) in pairs(preconditioner.x_indices)
        @inbounds destination[global_index] =
            preconditioner.x_solution[local_index]
    end
    for (local_index, global_index) in pairs(preconditioner.y_indices)
        @inbounds destination[global_index] =
            preconditioner.y_solution[local_index]
    end
    return destination
end

Base.size(preconditioner::VariableStaggeredGeometricPreconditioner) = begin
    n = length(preconditioner.x_indices) +
        length(preconditioner.y_indices)
    (n, n)
end
Base.eltype(::VariableStaggeredGeometricPreconditioner{T}) where {T} = T
_preconditioner_name(::VariableStaggeredGeometricPreconditioner) =
    :variable_staggered_geometric_multigrid

function _variable_geometric_component(
        mesh, matrix, stiffness, volumes,
        local_to_global::Vector{Int}, ::Val{Axis};
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        options
    ) where {Axis}
    T = eltype(volumes)
    n = length(local_to_global)
    component = StaggeredDiffusionComponent{Axis, T}(
        sparse(stiffness), spzeros(T, n, n), volumes,
        local_to_global, WallDiffusionConnection{T}[],
        zeros(T, n), zeros(T, n), zero(T)
    )
    elliptic = StaggeredEllipticComponent{Axis, T}(
        sparse(matrix), volumes, local_to_global
    )
    settings = merge(
        (; coarse_size, pre_sweeps, post_sweeps, max_levels), options
    )
    return _staggered_multigrid_component(
        mesh, component, elliptic;
        reaction = one(T), diffusion = one(T),
        # A principal block of div(2mu*D(u)) is positive definite, but its
        # shear contribution is not an M-matrix. Bound and normalize the
        # smoothed owning-aggregate transfer so it preserves constants and
        # positive dual volume. The Galerkin coarse matrix still contains the
        # actual inertia and variable-viscosity stiffness.
        positive_normalized_transfer = true, settings...
    )
end

function _variable_geometric_preconditioner(
        operator::VariableStaggeredDiffusionOperator{T}, matrix,
        inertia, dt;
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        options = (;)
    ) where {T}
    _, x_indices = _component_local_to_global(operator.mesh, 1)
    _, y_indices = _component_local_to_global(operator.mesh, 2)
    half_step = T(dt) / 2
    x_matrix = matrix[x_indices, x_indices]
    y_matrix = matrix[y_indices, y_indices]
    x_stiffness = half_step .* operator.stiffness[x_indices, x_indices]
    y_stiffness = half_step .* operator.stiffness[y_indices, y_indices]
    x_volumes = inertia[x_indices]
    y_volumes = inertia[y_indices]
    x = _variable_geometric_component(
        operator.mesh, x_matrix, x_stiffness, x_volumes,
        x_indices, Val(1); coarse_size, pre_sweeps, post_sweeps,
        max_levels, options
    )
    y = _variable_geometric_component(
        operator.mesh, y_matrix, y_stiffness, y_volumes,
        y_indices, Val(2); coarse_size, pre_sweeps, post_sweeps,
        max_levels, options
    )
    return VariableStaggeredGeometricPreconditioner(
        x, y, x_indices, y_indices,
        zeros(T, length(x_indices)), zeros(T, length(y_indices)),
        zeros(T, length(x_indices)), zeros(T, length(y_indices))
    )
end

function _refresh_variable_geometric_preconditioner!(
        preconditioner::VariableStaggeredGeometricPreconditioner{T},
        operator::VariableStaggeredDiffusionOperator{T}, matrix,
        inertia, dt
    ) where {T}
    x_indices = preconditioner.x_indices
    y_indices = preconditioner.y_indices
    half_step = T(dt) / 2
    _refresh_staggered_multigrid!(
        preconditioner.x,
        matrix[x_indices, x_indices],
        half_step .* operator.stiffness[x_indices, x_indices],
        inertia[x_indices],
    )
    _refresh_staggered_multigrid!(
        preconditioner.y,
        matrix[y_indices, y_indices],
        half_step .* operator.stiffness[y_indices, y_indices],
        inertia[y_indices],
    )
    return preconditioner
end

function _variable_diffusion_preconditioner(
        specification, operator, matrix, inertia, dt;
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        options
    )
    if specification === :geometric
        return _variable_geometric_preconditioner(
            operator, matrix, inertia, dt;
            coarse_size, pre_sweeps, post_sweeps, max_levels, options
        )
    end
    return _construct_matrix_preconditioner(
        specification, matrix;
        symmetric = true, singular = false, options
    )
end

mutable struct VariableStaggeredDiffusionSolver{
        T, K, KO, P, PO, KW, PC,
    }
    krylov_factory::K
    krylov_options::KO
    preconditioner_specification::P
    preconditioner_options::PO
    coarse_size::Int
    pre_sweeps::Int
    post_sweeps::Int
    max_levels::Int
    matrix::SparseMatrixCSC{T, Int}
    stiffness_pointers::Vector{Int}
    krylov::KW
    preconditioner::PC
    inertia::Vector{T}
    solution::Vector{T}
    right_hand_side::Vector{T}
    residual::Vector{T}
    product::Vector{T}
    stats::LinearSolverStats{T}
end


function _matrix_stiffness_pointers(matrix, stiffness)
    stiffness_rows = rowvals(stiffness)
    matrix_rows = rowvals(matrix)
    pointers = zeros(Int, nnz(matrix))
    for column in axes(matrix, 2), matrix_pointer in nzrange(matrix, column)
        row = matrix_rows[matrix_pointer]
        for stiffness_pointer in nzrange(stiffness, column)
            if stiffness_rows[stiffness_pointer] == row
                pointers[matrix_pointer] = stiffness_pointer
                break
            end
        end
    end
    return pointers
end

function _variable_diffusion_matrix(operator, density, dt)
    T = eltype(operator.volumes)
    length(density) == length(operator.volumes) ||
        throw(DimensionMismatch("one density per staggered volume is required"))
    all(rho -> isfinite(rho) && rho > zero(T), density) ||
        throw(ArgumentError("density must be finite and positive"))
    inertia = T.(density) .* operator.volumes
    matrix = spdiagm(0 => inertia) + (T(dt) / 2) * operator.stiffness
    dropzeros!(matrix)
    return sparse(matrix), inertia
end

function _refresh_variable_diffusion_matrix!(
        solver::VariableStaggeredDiffusionSolver{T}, operator,
        density, dt
    ) where {T}
    length(density) == length(operator.volumes) || throw(
        DimensionMismatch("one density per staggered volume is required"),
    )
    all(rho -> isfinite(rho) && rho > zero(T), density) || throw(
        ArgumentError("density must be finite and positive"),
    )
    matrix = solver.matrix
    stiffness = operator.stiffness
    half_step = T(dt) / 2
    matrix_values = nonzeros(matrix)
    stiffness_values = nonzeros(stiffness)
    for index in eachindex(matrix_values)
        pointer = @inbounds solver.stiffness_pointers[index]
        @inbounds matrix_values[index] = iszero(pointer) ? zero(T) :
            half_step * stiffness_values[pointer]
    end
    rows = rowvals(matrix)
    for column in axes(matrix, 2)
        @inbounds solver.inertia[column] =
            T(density[column]) * operator.volumes[column]
        for pointer in nzrange(matrix, column)
            if rows[pointer] == column
                @inbounds matrix_values[pointer] += solver.inertia[column]
                break
            end
        end
    end
    return matrix
end

function VariableStaggeredDiffusionSolver(
        operator::VariableStaggeredDiffusionOperator{T}, density, dt;
        krylov = Krylov.BicgstabWorkspace, krylov_options = (;),
        preconditioner = :geometric, preconditioner_options = (;),
        coarse_size = 64, pre_sweeps = 2, post_sweeps = 2,
        max_levels = 32
    ) where {T}
    matrix, inertia = _variable_diffusion_matrix(
        operator, density, dt
    )
    right_hand_side = zeros(T, length(inertia))
    krylov_workspace = _construct_krylov(
        krylov, matrix, right_hand_side
    )
    built_preconditioner = _variable_diffusion_preconditioner(
        preconditioner, operator, matrix, inertia, dt;
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        options = preconditioner_options
    )
    return VariableStaggeredDiffusionSolver{
        T, typeof(krylov), typeof(krylov_options),
        typeof(preconditioner), typeof(preconditioner_options),
        typeof(krylov_workspace), typeof(built_preconditioner),
    }(
        krylov, krylov_options, preconditioner,
        preconditioner_options, coarse_size, pre_sweeps,
        post_sweeps, max_levels, matrix,
        _matrix_stiffness_pointers(matrix, operator.stiffness),
        krylov_workspace,
        built_preconditioner, inertia, zeros(T, length(inertia)),
        right_hand_side, zeros(T, length(inertia)),
        zeros(T, length(inertia)),
        LinearSolverStats(
            T; method = _krylov_method(krylov_workspace),
            preconditioner = _preconditioner_name(built_preconditioner)
        )
    )
end

function _update_variable_diffusion_solver!(
        solver::VariableStaggeredDiffusionSolver,
        operator, density, dt
    )
    matrix = _refresh_variable_diffusion_matrix!(
        solver, operator, density, dt,
    )
    if solver.preconditioner_specification === :geometric &&
            solver.preconditioner isa VariableStaggeredGeometricPreconditioner
        _refresh_variable_geometric_preconditioner!(
            solver.preconditioner, operator, matrix, solver.inertia, dt,
        )
    else
        solver.preconditioner = _variable_diffusion_preconditioner(
            solver.preconditioner_specification, operator, matrix,
            solver.inertia, dt;
            coarse_size = solver.coarse_size,
            pre_sweeps = solver.pre_sweeps,
            post_sweeps = solver.post_sweeps,
            max_levels = solver.max_levels,
            options = solver.preconditioner_options
        )
    end
    return solver
end

function _solve_variable_diffusion!(
        destination, solver::VariableStaggeredDiffusionSolver,
        operator; rtol, atol, max_cycles
    )
    T = eltype(solver.solution)
    zero(T) <= rtol < one(T) ||
        throw(ArgumentError("rtol must lie in [0, 1)"))
    atol >= zero(T) || throw(ArgumentError("atol must be nonnegative"))
    max_cycles >= 0 ||
        throw(ArgumentError("max_cycles must be nonnegative"))
    copyto!(solver.solution, destination.interior)
    mul!(solver.product, solver.matrix, solver.solution)
    for index in eachindex(solver.residual)
        @inbounds solver.residual[index] =
            solver.right_hand_side[index] - solver.product[index]
    end
    volumes = operator.volumes
    initial_l2, initial_linf = _integrated_residual_norms(
        solver.residual, volumes
    )
    roundoff_floor = _roundoff_residual_floor!(
        solver.matrix, solver.solution, solver.right_hand_side,
        volumes, solver.product
    )
    target = max(
        T(atol), T(rtol) * max(initial_linf, eps(T)), roundoff_floor
    )
    if initial_linf > target
        _run_krylov!(
            solver.krylov, solver.matrix, solver.right_hand_side,
            solver.solution, solver.preconditioner, target,
            volumes, max_cycles, solver.krylov_options
        )
        copyto!(solver.solution, Krylov.solution(solver.krylov))
    end
    mul!(solver.product, solver.matrix, solver.solution)
    for index in eachindex(solver.residual)
        @inbounds solver.residual[index] =
            solver.right_hand_side[index] - solver.product[index]
    end
    final_l2, final_linf = _integrated_residual_norms(
        solver.residual, volumes
    )
    final_target = max(
        target,
        _roundoff_residual_floor!(
            solver.matrix, solver.solution,
            solver.right_hand_side, volumes, solver.product
        )
    )
    copyto!(destination.interior, solver.solution)
    fill!(destination.boundary, zero(T))
    stats = solver.stats
    stats.cycles = initial_linf <= target ? 0 :
        Krylov.iteration_count(solver.krylov)
    stats.converged = final_linf <= final_target
    stats.initial_l2 = initial_l2
    stats.final_l2 = final_l2
    stats.final_linf = final_linf
    stats.removed_rhs_mean = zero(T)
    stats.method = _krylov_method(solver.krylov)
    stats.preconditioner = _preconditioner_name(solver.preconditioner)
    return stats
end

"""Implicit midpoint of the coupled stress operator, with an optional positive
staggered density. `old_velocity` supplies the explicit half of the stress;
it defaults to `source` for a standalone diffusion substep. The sparse solve
is rebuilt here when coefficients change."""
function implicit_midpoint_diffusion!(
        destination::FaceVelocity, source::FaceVelocity,
        operator::VariableStaggeredDiffusionOperator, dt;
        old_velocity = source,
        density = ones(eltype(source), length(source.interior))
    )
    destination.mesh === source.mesh === old_velocity.mesh === operator.mesh || throw(ArgumentError("diffusion arguments must share a mesh"))
    isfinite(dt)&&dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    length(density) == length(operator.volumes) || throw(DimensionMismatch("one density per staggered volume is required"))
    all(rho -> isfinite(rho)&&rho > 0, density) || throw(ArgumentError("density must be finite and positive"))
    inertia = density .* operator.volumes
    matrix = spdiagm(0 => inertia) + (dt / 2) * operator.stiffness
    rhs = inertia .* source.interior -
        (dt / 2) * (operator.stiffness * old_velocity.interior)
    destination.interior .= matrix \ rhs
    fill!(destination.boundary, zero(eltype(destination)))
    return destination
end

function implicit_midpoint_diffusion!(
        destination::FaceVelocity{T}, source::FaceVelocity{T},
        operator::VariableStaggeredDiffusionOperator{T}, dt,
        solver::VariableStaggeredDiffusionSolver{T};
        old_velocity = source,
        density = ones(T, length(source.interior)),
        rtol = 1.0e-10, atol = zero(T), max_cycles = 500
    ) where {T}
    destination.mesh === source.mesh === old_velocity.mesh ===
        operator.mesh ||
        throw(ArgumentError("diffusion arguments must share a mesh"))
    isfinite(dt) && dt > zero(T) ||
        throw(ArgumentError("dt must be finite and positive"))
    _update_variable_diffusion_solver!(
        solver, operator, density, T(dt)
    )
    mul!(solver.product, operator.stiffness, old_velocity.interior)
    for index in eachindex(solver.right_hand_side)
        @inbounds solver.right_hand_side[index] =
            solver.inertia[index] * source.interior[index] -
            T(dt) * solver.product[index] / 2
    end
    stats = _solve_variable_diffusion!(
        destination, solver, operator; rtol, atol, max_cycles
    )
    stats.converged ||
        throw(
        ErrorException(
            "variable staggered diffusion solve failed to converge after " *
                "$(stats.cycles) iterations " *
                "(initial L2=$(stats.initial_l2), " *
                "final L∞=$(stats.final_linf))"
        )
    )
    return destination
end
