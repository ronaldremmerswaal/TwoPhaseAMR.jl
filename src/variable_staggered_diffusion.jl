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

"""Grow-only scratch storage for topology-dependent stress construction."""
mutable struct VariableDiffusionConstructionCache{T}
    cell_references::Vector{CellRef}
    strain_rows::Vector{Int}
    strain_columns::Vector{Int}
    strain_values::Vector{T}
    interpolation_rows::Vector{Int}
    interpolation_columns::Vector{Int}
    interpolation_values::Vector{T}
    vertices::Dict{
        NTuple{2, Int}, Tuple{T, Vector{Tuple{CellRef, T}}},
    }
    vertex_reference_pool::Vector{Vector{Tuple{CellRef, T}}}
    vertex_keys::Vector{NTuple{2, Int}}
    interior_face_indices::NTuple{2, Vector{Int}}
    interior_face_x::NTuple{2, Vector{T}}
    boundary_face_indices::NTuple{2, Vector{Int}}
    boundary_face_x::NTuple{2, Vector{T}}
    derivative_points::Vector{Tuple{Int, T, T}}
    derivative_candidates::Vector{Int}
    derivative_indices::Vector{Int}
    derivative_coefficients::Vector{T}
    derivative_templates::Dict{Any, Vector{T}}
    derivative_template_hits::Int
    derivative_template_misses::Int
    stiffness_entries::Vector{Vector{Tuple{Int, T}}}
end

function VariableDiffusionConstructionCache(::Type{T}) where {T}
    return VariableDiffusionConstructionCache(
        CellRef[], Int[], Int[], T[], Int[], Int[], T[],
        Dict{NTuple{2, Int}, Tuple{T, Vector{Tuple{CellRef, T}}}}(),
        Vector{Tuple{CellRef, T}}[], NTuple{2, Int}[],
        (Int[], Int[]), (T[], T[]), (Int[], Int[]), (T[], T[]),
        Tuple{Int, T, T}[], Int[], Int[], T[],
        Dict{Any, Vector{T}}(), 0, 0,
        Vector{Tuple{Int, T}}[],
    )
end

function _cached_cell_references!(references, mesh)
    empty!(references)
    nx, ny = mesh.block_size
    sizehint!(references, ncells(mesh))
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        push!(references, CellRef(block, i, j))
    end
    return references
end

function _stiffness_refresh_map!(
        pointers, stresses, factors, entries, strain, stiffness
    )
    T = eltype(stiffness)
    empty!(pointers)
    empty!(stresses)
    empty!(factors)
    old_length = length(entries)
    number_of_stresses = size(strain, 1)
    number_of_stresses > old_length && resize!(entries, number_of_stresses)
    for index in 1:number_of_stresses
        if index > old_length
            entries[index] = Tuple{Int, T}[]
        else
            empty!(entries[index])
        end
    end
    strain_rows = rowvals(strain)
    strain_values = nonzeros(strain)
    for column in axes(strain, 2), pointer in nzrange(strain, column)
        row = strain_rows[pointer]
        push!(entries[row], (column, strain_values[pointer]))
    end
    stiffness_rows = rowvals(stiffness)
    for stress in 1:number_of_stresses
        row_entries = entries[stress]
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

function _stress_vertices!(cache, mesh, ::Type{T}) where {T}
    # Cell-corner quadrature weights partition the domain volume exactly.
    lattice = mesh.block_size .* (1 << mesh.max_level)
    vertices = cache.vertices
    for (_, references) in values(vertices)
        empty!(references)
        push!(cache.vertex_reference_pool, references)
    end
    empty!(vertices)
    nx, ny = mesh.block_size
    for ref in cache.cell_references
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
                references = isempty(cache.vertex_reference_pool) ?
                    Tuple{CellRef, T}[] : pop!(cache.vertex_reference_pool)
                (zero(T), references)
            end
            weight = T(block.volume / 4)
            push!(refs, (ref, weight))
            vertices[key] = (volume + weight, refs)
        end
    end
    return vertices, lattice
end

function _release_stress_vertices!(cache)
    for (_, references) in values(cache.vertices)
        empty!(references)
        push!(cache.vertex_reference_pool, references)
    end
    empty!(cache.vertices)
    empty!(cache.vertex_keys)
    return cache
end

struct DiffusionFaceSearch{T}
    interior_indices::NTuple{2, Vector{Int}}
    interior_x::NTuple{2, Vector{T}}
    boundary_indices::NTuple{2, Vector{Int}}
    boundary_x::NTuple{2, Vector{T}}
end

function _sorted_face_axis!(indices, coordinates, faces, axis, ::Type{T}) where {T}
    empty!(indices)
    empty!(coordinates)
    for (index, face) in pairs(faces)
        face.axis == axis && push!(indices, index)
    end
    sort!(indices; by = index -> (faces[index].center[1], index))
    sizehint!(coordinates, length(indices))
    for index in indices
        push!(coordinates, T(faces[index].center[1]))
    end
    return indices, coordinates
end

function DiffusionFaceSearch(cache, mesh, ::Type{T}) where {T}
    for axis in 1:2
        _sorted_face_axis!(
            cache.interior_face_indices[axis], cache.interior_face_x[axis],
            mesh.faces, axis, T,
        )
        _sorted_face_axis!(
            cache.boundary_face_indices[axis], cache.boundary_face_x[axis],
            mesh.boundary_faces, axis, T,
        )
    end
    return DiffusionFaceSearch(
        cache.interior_face_indices, cache.interior_face_x,
        cache.boundary_face_indices, cache.boundary_face_x,
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
        dx = T(
            _wrapped_delta(
                face.center[1] - point[1], mesh.extent[1], mesh.periodic[1]
            )
        )
        dy = T(
            _wrapped_delta(
                face.center[2] - point[2], mesh.extent[2], mesh.periodic[2]
            )
        )
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
        points, candidates, ids, stencil, cache
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
    sort!(points; by = p -> (p[2]^2 + p[3]^2, p[2], p[3], iszero(p[1])))
    length(points) > 16 && resize!(points, 16)
    template_key = (
        component, typeof(boundary),
        Tuple((iszero(p[1]), p[2] / scale, p[3] / scale) for p in points),
    )
    template = get(cache.derivative_templates, template_key, nothing)
    if !isnothing(template)
        cache.derivative_template_hits += 1
        for (i, p) in pairs(points)
            p[1] > 0 || continue
            push!(ids, p[1])
            push!(stencil, template[i] / scale)
        end
        return ids, stencil
    end
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
        template = Vector{T}(undef, length(points))
        for (i, p) in pairs(points)
            value = weights[i] * coefficients[i]
            template[i] = value * scale
            p[1] > 0 || continue
            push!(ids, p[1]);push!(stencil, value)
        end
        cache.derivative_templates[template_key] = template
        cache.derivative_template_misses += 1
        return ids, stencil
    end
    throw(ArgumentError("insufficient faces for staggered shear reconstruction"))
end

function _variable_staggered_diffusion(
        mesh, viscosity::CellField, boundary, ::Type{T};
        cache = VariableDiffusionConstructionCache(T), reuse = nothing,
        assemble_stiffness = true,
    ) where {T}
    viscosity.mesh === mesh || throw(ArgumentError("viscosity field must share the operator mesh"))
    boundary isa FreeSlip || (boundary isa NoSlip && boundary.velocity isa StationaryWall) ||
        throw(ArgumentError("variable stress diffusion currently supports stationary NoSlip or FreeSlip walls"))
    all(c -> isfinite(c)&&c > 0, (v for b in eachindex(mesh.blocks) for v in interior(viscosity, b))) ||
        throw(ArgumentError("dynamic viscosity must be finite and positive"))
    cache isa VariableDiffusionConstructionCache{T} || throw(
        ArgumentError("diffusion construction cache has the wrong scalar type"),
    )
    isnothing(reuse) || reuse isa VariableStaggeredDiffusionOperator{T} ||
        throw(ArgumentError("reused diffusion operator has the wrong type"))
    refs = _cached_cell_references!(cache.cell_references, mesh)
    nx, ny = mesh.block_size
    rows = cache.strain_rows
    cols = cache.strain_columns
    vals = cache.strain_values
    interpolation_rows = cache.interpolation_rows
    interpolation_columns = cache.interpolation_columns
    interpolation_values = cache.interpolation_values
    for values in (
            rows, cols, vals, interpolation_rows,
            interpolation_columns, interpolation_values,
        )
        empty!(values)
    end
    measures = isnothing(reuse) ? T[] : reuse.stress_measures
    empty!(measures)
    face_search = DiffusionFaceSearch(cache, mesh, T)
    derivative_points = cache.derivative_points
    derivative_candidates = cache.derivative_candidates
    derivative_indices = cache.derivative_indices
    derivative_coefficients = cache.derivative_coefficients
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
    vertices, lattice = _stress_vertices!(cache, mesh, T)
    empty!(cache.vertex_keys)
    append!(cache.vertex_keys, keys(vertices))
    sort!(cache.vertex_keys)
    for key in cache.vertex_keys
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
                derivative_indices, derivative_coefficients, cache,
            )
            for index in eachindex(indices)
                push!(rows, row)
                push!(cols, indices[index])
                push!(vals, coefficients[index])
            end
        end
    end
    strain = sparse(rows, cols, vals, length(measures), length(mesh.faces))
    interpolation = sparse(
        interpolation_rows, interpolation_columns, interpolation_values,
        length(measures), length(refs)
    )
    cell_work = isnothing(reuse) ? T[] : reuse.cell_work
    resize!(cell_work, length(refs))
    for (index, ref) in pairs(refs)
        @inbounds cell_work[index] = log(_get(viscosity, ref))
    end
    stress_work = isnothing(reuse) ? T[] : reuse.stress_work
    resize!(stress_work, length(measures))
    mul!(stress_work, interpolation, cell_work)
    weights = isnothing(reuse) ? T[] : reuse.stress_weights
    resize!(weights, length(measures))
    for index in eachindex(weights)
        @inbounds weights[index] = measures[index] * exp(stress_work[index])
    end
    stiffness_pointers = isnothing(reuse) ? Int[] : reuse.stiffness_pointers
    stiffness_stresses = isnothing(reuse) ? Int[] : reuse.stiffness_stresses
    stiffness_factors = isnothing(reuse) ? T[] : reuse.stiffness_factors
    for values in (stiffness_pointers, stiffness_stresses, stiffness_factors)
        empty!(values)
    end
    stiffness = if assemble_stiffness
        assembled = transpose(strain) * spdiagm(0 => weights) * strain
        stiffness_pointers, stiffness_stresses, stiffness_factors =
            _stiffness_refresh_map!(
            stiffness_pointers, stiffness_stresses, stiffness_factors,
            cache.stiffness_entries, strain, assembled,
        )
        assembled
    else
        spzeros(T, 0, 0)
    end
    volumes = isnothing(reuse) ? T[] : reuse.volumes
    resize!(volumes, length(mesh.faces))
    for (index, face) in pairs(mesh.faces)
        @inbounds volumes[index] = face.area * (
            mesh.blocks[face.minus.block].spacing[face.axis] +
                mesh.blocks[face.plus.block].spacing[face.axis]
        ) / 2
    end
    _release_stress_vertices!(cache)
    return VariableStaggeredDiffusionOperator(
        mesh, stiffness, volumes, boundary, strain, interpolation,
        measures, weights, cell_work, stress_work,
        stiffness_pointers, stiffness_stresses, stiffness_factors,
    )
end

"""Refresh the stress-location geometric mean without rebuilding geometric
strain and interpolation stencils. This is used after every VOF update."""
function _update_log_dynamic_viscosity!(
        operator::VariableStaggeredDiffusionOperator{T},
        cell_log_viscosity::AbstractVector,
    ) where {T}
    length(cell_log_viscosity) == length(operator.cell_work) || throw(
        DimensionMismatch("one log-viscosity value is required per cell"),
    )
    cell_log_viscosity === operator.cell_work ||
        copyto!(operator.cell_work, cell_log_viscosity)
    mul!(
        operator.stress_work, operator.stress_interpolation,
        operator.cell_work
    )
    for index in eachindex(operator.stress_weights)
        @inbounds operator.stress_weights[index] =
            operator.stress_measures[index] * exp(operator.stress_work[index])
    end
    isempty(operator.stiffness_pointers) && return operator
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
    return _update_log_dynamic_viscosity!(operator, operator.cell_work)
end

function _staggered_diffusion_operator(
        field::InterfaceField; viscosity::NTuple{2}, boundary = NoSlip(),
        construction_cache = nothing, reuse = nothing,
        assemble_stiffness = true,
    )
    mu = dynamic_viscosity(field; viscosity)
    return _staggered_diffusion_operator(
        field.mesh; viscosity = mu, boundary, construction_cache, reuse,
        assemble_stiffness,
    )
end


function StaggeredDiffusionOperator(
        field::InterfaceField; viscosity::NTuple{2}, boundary = NoSlip(),
    )
    return _staggered_diffusion_operator(field; viscosity, boundary)
end
function diffusion!(
        destination::FaceVelocity, velocity::FaceVelocity,
        operator::VariableStaggeredDiffusionOperator; time = 0
    )
    destination.mesh === velocity.mesh === operator.mesh || throw(ArgumentError("diffusion arguments must share a mesh"))
    destination === velocity && throw(ArgumentError("diffusion output must not alias input"))
    _factorized_variable_stiffness_mul!(
        destination.interior, operator, velocity.interior,
        operator.stress_work,
    )
    for index in eachindex(destination.interior)
        @inbounds destination.interior[index] =
            -destination.interior[index] / operator.volumes[index]
    end
    fill!(destination.boundary, zero(eltype(destination)))
    return destination
end

"""Apply `E' * W_mu * E` with caller-owned stress scratch."""
function _factorized_variable_stiffness_mul!(
        destination::AbstractVector{T},
        operator::VariableStaggeredDiffusionOperator{T},
        source::AbstractVector{T},
        stress_work::AbstractVector{T},
    ) where {T}
    number_of_unknowns = length(operator.volumes)
    length(destination) == length(source) == number_of_unknowns ||
        throw(DimensionMismatch("variable diffusion vector sizes differ"))
    length(stress_work) == length(operator.stress_weights) ||
        throw(DimensionMismatch("variable diffusion stress-work size differs"))
    destination === source && throw(
        ArgumentError("variable diffusion output must not alias its input"),
    )
    mul!(stress_work, operator.strain, source)
    for index in eachindex(stress_work)
        @inbounds stress_work[index] *= operator.stress_weights[index]
    end
    mul!(destination, transpose(operator.strain), stress_work)
    return destination
end

"""Allocation-free fine-grid action for implicit variable-stress diffusion.

This evaluates `M_rho*x + dt/2*E'*W_mu*E*x` without materializing a new
coefficient-dependent sparse matrix.  The assembled matrix remains available
as a reference and for the current geometric hierarchy during the staged
matrix-free conversion.
"""
mutable struct FactorizedVariableDiffusionOperator{
        T, O <: VariableStaggeredDiffusionOperator{T}, V <: Vector{T},
    }
    operator::O
    inertia::V
    half_step::T
    stress_work::V
end

function FactorizedVariableDiffusionOperator(
        operator::O, inertia::Vector{T}, dt,
    ) where {T, O <: VariableStaggeredDiffusionOperator{T}}
    return FactorizedVariableDiffusionOperator{
        T, O, Vector{T},
    }(
        operator, inertia, T(dt) / 2,
        zeros(T, length(operator.stress_weights)),
    )
end

Base.size(action::FactorizedVariableDiffusionOperator) =
    (length(action.inertia), length(action.inertia))
Base.size(action::FactorizedVariableDiffusionOperator, dimension::Integer) =
    size(action)[dimension]
Base.eltype(::FactorizedVariableDiffusionOperator{T}) where {T} = T
issymmetric(::FactorizedVariableDiffusionOperator) = true

function mul!(
        destination::AbstractVector{T},
        action::FactorizedVariableDiffusionOperator{T},
        source::AbstractVector{T},
    ) where {T}
    _factorized_variable_stiffness_mul!(
        destination, action.operator, source, action.stress_work,
    )
    half_step = action.half_step
    for index in eachindex(destination)
        @inbounds destination[index] = muladd(
            half_step, destination[index],
            action.inertia[index] * source[index],
        )
    end
    return destination
end

function _factorized_variable_level_data(
        operator::VariableStaggeredDiffusionOperator{T}, inertia, dt,
    ) where {T}
    number_of_unknowns = length(inertia)
    diagonal = copy(inertia)
    row_sums = copy(inertia)
    stress_row_sums = zeros(T, length(operator.stress_weights))
    rows = rowvals(operator.strain)
    values = nonzeros(operator.strain)
    for column in axes(operator.strain, 2), pointer in nzrange(operator.strain, column)
        @inbounds stress_row_sums[rows[pointer]] += abs(values[pointer])
    end
    half_step = T(dt) / 2
    for column in axes(operator.strain, 2), pointer in nzrange(operator.strain, column)
        stress = @inbounds rows[pointer]
        value = @inbounds values[pointer]
        weight = @inbounds operator.stress_weights[stress]
        @inbounds diagonal[column] += half_step * weight * value * value
        @inbounds row_sums[column] += half_step * abs(value) * weight *
            stress_row_sums[stress]
    end
    inverse_diagonal = similar(diagonal)
    spectral_bound = zero(T)
    for index in eachindex(diagonal)
        diagonal[index] > zero(T) || throw(
            ArgumentError("factorized diffusion operator has a non-positive diagonal"),
        )
        @inbounds spectral_bound = max(
            spectral_bound, row_sums[index] / diagonal[index],
        )
        @inbounds inverse_diagonal[index] = inv(diagonal[index])
    end
    relaxation_weight = iszero(spectral_bound) ? one(T) :
        min(one(T), T(4) / (T(3) * spectral_bound))
    return inverse_diagonal, relaxation_weight
end

function _factorized_variable_diffusion_level(
        operator::VariableStaggeredDiffusionOperator{T}, inertia, dt,
        prolongation,
    ) where {T}
    action = FactorizedVariableDiffusionOperator(operator, inertia, dt)
    inverse_diagonal, relaxation_weight =
        _factorized_variable_level_data(operator, inertia, dt)
    number_of_unknowns = length(inertia)
    return MultigridLevel(
        action, inertia, prolongation, inverse_diagonal, relaxation_weight,
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
        zeros(T, number_of_unknowns), zeros(T, number_of_unknowns),
    )
end

function _refresh_factorized_variable_diffusion_level!(
        level::MultigridLevel,
        operator::VariableStaggeredDiffusionOperator{T}, inertia, dt,
    ) where {T}
    action = level.matrix
    action isa FactorizedVariableDiffusionOperator || throw(
        ArgumentError("expected a factorized variable-diffusion level"),
    )
    action.operator = operator
    action.inertia = inertia
    action.half_step = T(dt) / 2
    resize!(action.stress_work, length(operator.stress_weights))
    inverse_diagonal, relaxation_weight =
        _factorized_variable_level_data(operator, inertia, dt)
    copyto!(level.inverse_diagonal, inverse_diagonal)
    level.relaxation_weight = relaxation_weight
    level.volumes = inertia
    return level
end

function _roundoff_residual_floor!(
        action::FactorizedVariableDiffusionOperator{T}, solution,
        right_hand_side, volumes, temporary,
    ) where {T}
    operator = action.operator
    strain = operator.strain
    rows = rowvals(strain)
    values = nonzeros(strain)
    stress_bound = action.stress_work
    fill!(stress_bound, zero(T))
    for column in axes(strain, 2), pointer in nzrange(strain, column)
        @inbounds stress_bound[rows[pointer]] +=
            abs(values[pointer]) * abs(solution[column])
    end
    for index in eachindex(temporary)
        @inbounds temporary[index] =
            action.inertia[index] * abs(solution[index])
    end
    for column in axes(strain, 2), pointer in nzrange(strain, column)
        stress = @inbounds rows[pointer]
        @inbounds temporary[column] += action.half_step *
            abs(values[pointer]) * operator.stress_weights[stress] *
            stress_bound[stress]
    end
    scale = one(T)
    for index in eachindex(temporary)
        @inbounds scale = max(
            scale, temporary[index] / volumes[index],
            abs(right_hand_side[index] / volumes[index]),
        )
    end
    return T(128) * eps(T) * scale
end

"""Rediscretized full-vector diffusion multigrid on pruned block trees.

Each coarse level has its own staggered strain/stress discretization. Cell
log-viscosity is volume averaged, staggered density is area averaged on
coincident faces, and the shifted mass matrix is then assembled directly on
the coarse mesh. Thus no level forms `P' * A * P`. The face transfer is a
topology-only staggered interpolation between the nested tree meshes. Once
the root block is reached, its sparse coupled operator is factorized directly.
"""
mutable struct TreeVariableDiffusionPreconditioner{T, F, L, C} <:
    AbstractGeometricMultigridPreconditioner{T}
    operators::Vector{VariableStaggeredDiffusionOperator{T}}
    levels::Vector{L}
    cell_restrictions::Vector{CellCoefficientRestriction{T}}
    density_restrictions::Vector{FaceCoefficientRestriction{T}}
    densities::Vector{Vector{T}}
    construction_cache::C
    coarse_factorization::F
    coarse_augmented::Bool
    coarse_rhs::Vector{T}
    coarse_solution::Vector{T}
    pre_sweeps::Int
    post_sweeps::Int
    pruning::Symbol
    reused_levels::Int
end

_preconditioner_name(::TreeVariableDiffusionPreconditioner) =
    :tree_variable_diffusion_multigrid

function _cell_viscosity_from_log!(field::CellField{T}, values) where {T}
    length(values) == ncells(field.mesh) || throw(
        DimensionMismatch("one log-viscosity value is required per cell"),
    )
    for (index, ref) in pairs(_cell_references(field.mesh))
        @inbounds _set!(field, exp(T(values[index])), ref)
    end
    return field
end

function _tree_variable_diffusion_preconditioner(
        operator::VariableStaggeredDiffusionOperator{T}, matrix,
        inertia, dt;
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        options = (;), reuse = nothing,
    ) where {T}
    all(key -> key === :pruning, keys(options)) || throw(
        ArgumentError(
            "tree-geometric diffusion options currently support only pruning",
        ),
    )
    pruning = get(options, :pruning, :all)
    pruning in (:deepest, :all) || throw(
        ArgumentError("tree pruning must be :deepest or :all"),
    )
    coarse_size >= 1 || throw(ArgumentError("coarse_size must be positive"))
    pre_sweeps >= 0 || throw(ArgumentError("pre_sweeps must be nonnegative"))
    post_sweeps >= 0 || throw(ArgumentError("post_sweeps must be nonnegative"))
    max_levels >= 1 || throw(ArgumentError("max_levels must be positive"))
    isnothing(reuse) ||
        reuse isa TreeVariableDiffusionPreconditioner{T} || throw(
        ArgumentError("reused diffusion hierarchy has the wrong type"),
    )
    can_reuse = !isnothing(reuse) && reuse.pruning === pruning &&
        typeof(reuse.operators[1].boundary) === typeof(operator.boundary)

    operators = VariableStaggeredDiffusionOperator{T}[operator]
    matrices = Any[matrix]
    inertias = Vector{T}[inertia]
    fine_density = isnothing(reuse) ? T[] : reuse.densities[1]
    resize!(fine_density, length(inertia))
    for index in eachindex(fine_density)
        @inbounds fine_density[index] = inertia[index] / operator.volumes[index]
    end
    densities = Vector{T}[fine_density]
    prolongations = SparseMatrixCSC{T, Int}[]
    cell_restrictions = CellCoefficientRestriction{T}[]
    density_restrictions = FaceCoefficientRestriction{T}[]
    current_mesh = operator.mesh
    active_level = maximum(block.key.level for block in current_mesh.blocks)
    construction_cache = isnothing(reuse) ?
        VariableDiffusionConstructionCache(T) : reuse.construction_cache
    target_size = 2coarse_size
    reused_new_start = 0
    reused_old_start = 0
    matrix_free_fine = isnothing(matrix)
    initial_matching_level = can_reuse && !matrix_free_fine ?
        _matching_tree_level(reuse.operators, current_mesh, 1) : 0
    initial_suffix_length = iszero(initial_matching_level) ? 0 :
        length(reuse.operators) - initial_matching_level + 1
    if !iszero(initial_matching_level) &&
            initial_suffix_length <= max_levels
        reused_new_start = 1
        reused_old_start = initial_matching_level
        fine_density = reuse.densities[initial_matching_level]
        resize!(fine_density, length(inertia))
        for index in eachindex(fine_density)
            @inbounds fine_density[index] =
                inertia[index] / operator.volumes[index]
        end
        densities[1] = fine_density
        for old_index in (initial_matching_level + 1):length(reuse.operators)
            push!(operators, reuse.operators[old_index])
            push!(matrices, reuse.levels[old_index].matrix)
            push!(inertias, reuse.levels[old_index].volumes)
            push!(densities, reuse.densities[old_index])
        end
        for old_index in initial_matching_level:length(
                reuse.cell_restrictions)
            push!(cell_restrictions, reuse.cell_restrictions[old_index])
            push!(
                density_restrictions,
                reuse.density_restrictions[old_index],
            )
            push!(prolongations, reuse.levels[old_index].prolongation)
        end
    end

    while iszero(reused_new_start) &&
            length(current_mesh.faces) > target_size && active_level > 0 &&
            length(operators) < max_levels
        coarse_mesh = if pruning === :deepest
            active_level -= 1
            _capped_tree_mesh(operator.mesh, active_level)
        else
            _all_leaf_pruned_mesh(current_mesh)
        end
        if length(coarse_mesh.faces) >= length(current_mesh.faces)
            pruning === :all && break
            continue
        end

        prolongation = _tree_face_prolongation(
            current_mesh, coarse_mesh, T,
        )
        cell_restriction = _cell_coefficient_restriction(
            coarse_mesh, current_mesh, T,
        )
        density_restriction = _face_coefficient_restriction(
            coarse_mesh, current_mesh, T,
        )
        matching_level = can_reuse ?
            _matching_tree_level(reuse.operators, coarse_mesh, 1) : 0
        suffix_length = iszero(matching_level) ? 0 :
            length(reuse.operators) - matching_level + 1
        can_append_suffix = !iszero(matching_level) &&
            length(operators) + suffix_length <= max_levels
        if can_append_suffix
            coarse_operator = reuse.operators[matching_level]
            coarse_density = reuse.densities[matching_level]
            coarse_matrix = reuse.levels[matching_level].matrix
            coarse_inertia = reuse.levels[matching_level].volumes
        else
            coarse_log_viscosity = zeros(T, ncells(coarse_mesh))
            _restrict_cell_coefficient!(
                coarse_log_viscosity, last(operators).cell_work,
                cell_restriction,
            )
            coarse_viscosity = CellField(coarse_mesh, T)
            _cell_viscosity_from_log!(
                coarse_viscosity, coarse_log_viscosity,
            )
            coarse_operator = _staggered_diffusion_operator(
                coarse_mesh; viscosity = coarse_viscosity,
                boundary = operator.boundary,
                construction_cache,
            )
            coarse_density = zeros(T, length(coarse_mesh.faces))
            _restrict_face_coefficients!(
                coarse_density, last(densities), density_restriction,
            )
            coarse_matrix, coarse_inertia = _variable_diffusion_matrix(
                coarse_operator, coarse_density, dt,
            )
        end
        push!(prolongations, prolongation)
        push!(cell_restrictions, cell_restriction)
        push!(density_restrictions, density_restriction)
        push!(operators, coarse_operator)
        push!(matrices, coarse_matrix)
        push!(inertias, coarse_inertia)
        push!(densities, coarse_density)
        current_mesh = coarse_mesh
        if can_append_suffix
            reused_new_start = length(operators)
            reused_old_start = matching_level
            for old_index in (matching_level + 1):length(reuse.operators)
                push!(operators, reuse.operators[old_index])
                push!(matrices, reuse.levels[old_index].matrix)
                push!(inertias, reuse.levels[old_index].volumes)
                push!(densities, reuse.densities[old_index])
            end
            for old_index in matching_level:length(reuse.cell_restrictions)
                push!(
                    cell_restrictions,
                    reuse.cell_restrictions[old_index],
                )
                push!(
                    density_restrictions,
                    reuse.density_restrictions[old_index],
                )
                push!(
                    prolongations,
                    reuse.levels[old_index].prolongation,
                )
            end
            break
        end
    end

    levels = MultigridLevel{T, SparseMatrixCSC{T, Int}}[]
    for index in eachindex(matrices)
        prolongation = index < length(matrices) ?
            prolongations[index] :
            spzeros(T, length(inertias[index]), 0)
        if matrix_free_fine && index == 1
            length(matrices) > 1 || throw(
                ArgumentError(
                    "a matrix-free finest diffusion level requires a coarser tree level",
                ),
            )
            push!(
                levels,
                _factorized_variable_diffusion_level(
                    operators[index], inertias[index], dt, prolongation,
                ),
            )
        elseif !iszero(reused_new_start) && index >= reused_new_start
            old_index = reused_old_start + index - reused_new_start
            level = reuse.levels[old_index]
            level.matrix = matrices[index]
            level.volumes = inertias[index]
            level.prolongation = prolongation
            push!(levels, level)
        else
            push!(
                levels,
                _multigrid_level(
                    matrices[index], inertias[index], prolongation,
                ),
            )
        end
    end
    coarse = last(levels)
    if !iszero(reused_new_start)
        factorization = reuse.coarse_factorization
        coarse_rhs = reuse.coarse_rhs
        coarse_solution = reuse.coarse_solution
    else
        factorization = lu(coarse.matrix)
        coarse_rhs = zeros(T, length(coarse.volumes))
        coarse_solution = similar(coarse_rhs)
    end
    preconditioner = TreeVariableDiffusionPreconditioner(
        operators, levels, cell_restrictions, density_restrictions,
        densities, construction_cache,
        factorization, false, coarse_rhs, coarse_solution,
        pre_sweeps, post_sweeps, pruning,
        iszero(reused_new_start) ? 0 :
            length(operators) - reused_new_start + 1,
    )
    !iszero(reused_new_start) &&
        _refresh_tree_variable_diffusion_preconditioner!(
            preconditioner, operator, matrix, inertia, dt,
        )
    return preconditioner
end

function _refresh_tree_variable_diffusion_preconditioner!(
        preconditioner::TreeVariableDiffusionPreconditioner{T},
        operator::VariableStaggeredDiffusionOperator{T}, matrix,
        inertia, dt,
    ) where {T}
    preconditioner.operators[1] = operator
    fine_density = preconditioner.densities[1]
    for index in eachindex(fine_density)
        @inbounds fine_density[index] =
            inertia[index] / operator.volumes[index]
    end
    if preconditioner.levels[1].matrix isa
            FactorizedVariableDiffusionOperator
        _refresh_factorized_variable_diffusion_level!(
            preconditioner.levels[1], operator, inertia, dt,
        )
    else
        preconditioner.levels[1].matrix = matrix
        preconditioner.levels[1].volumes = inertia
    end

    for index in 2:length(preconditioner.operators)
        fine_operator = preconditioner.operators[index - 1]
        coarse_operator = preconditioner.operators[index]
        _restrict_cell_coefficient!(
            coarse_operator.cell_work, fine_operator.cell_work,
            preconditioner.cell_restrictions[index - 1],
        )
        _update_log_dynamic_viscosity!(
            coarse_operator, coarse_operator.cell_work,
        )
        _restrict_face_coefficients!(
            preconditioner.densities[index],
            preconditioner.densities[index - 1],
            preconditioner.density_restrictions[index - 1],
        )
        coarse_inertia = preconditioner.levels[index].volumes
        coarse_density = preconditioner.densities[index]
        for face in eachindex(coarse_inertia)
            @inbounds coarse_inertia[face] =
                coarse_density[face] * coarse_operator.volumes[face]
        end
        _refresh_variable_diffusion_matrix_from_inertia!(
            preconditioner.levels[index].matrix,
            coarse_operator, coarse_inertia, dt,
        )
    end

    for index in eachindex(preconditioner.levels)
        level = preconditioner.levels[index]
        level.matrix isa FactorizedVariableDiffusionOperator && continue
        _refresh_multigrid_level!(
            level, level.matrix, level.volumes, level.prolongation,
        )
    end
    coarse = last(preconditioner.levels)
    preconditioner.coarse_factorization = lu(coarse.matrix)
    return preconditioner
end

function _variable_diffusion_preconditioner(
        specification, operator, matrix, inertia, dt;
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        options, reuse = nothing,
    )
    if specification in (:geometric, :tree_geometric)
        return _tree_variable_diffusion_preconditioner(
            operator, matrix, inertia, dt;
            coarse_size, pre_sweeps, post_sweeps, max_levels, options, reuse,
        )
    end
    isnothing(reuse) || throw(
        ArgumentError("hierarchy reuse requires a tree-geometric preconditioner"),
    )
    return _construct_matrix_preconditioner(
        specification, matrix;
        symmetric = true, singular = false, options
    )
end

mutable struct VariableStaggeredDiffusionSolver{
        T, K, KO, P, PO, A, KW, PC,
    }
    krylov_factory::K
    krylov_options::KO
    preconditioner_specification::P
    preconditioner_options::PO
    coarse_size::Int
    pre_sweeps::Int
    post_sweeps::Int
    max_levels::Int
    matrix::Union{Nothing, SparseMatrixCSC{T, Int}}
    action::A
    krylov::KW
    preconditioner::PC
    inertia::Vector{T}
    solution::Vector{T}
    right_hand_side::Vector{T}
    residual::Vector{T}
    product::Vector{T}
    stats::LinearSolverStats{T}
end

function _variable_diffusion_matrix(operator, density, dt)
    T = eltype(operator.volumes)
    length(density) == length(operator.volumes) ||
        throw(DimensionMismatch("one density per staggered volume is required"))
    all(rho -> isfinite(rho) && rho > zero(T), density) ||
        throw(ArgumentError("density must be finite and positive"))
    inertia = T.(density) .* operator.volumes
    matrix = copy(operator.stiffness)
    matrix.nzval .*= T(dt) / 2
    rows = rowvals(matrix)
    for column in axes(matrix, 2)
        diagonal = 0
        for pointer in nzrange(matrix, column)
            if rows[pointer] == column
                diagonal = pointer
                break
            end
        end
        iszero(diagonal) && error(
            "variable diffusion stiffness has no diagonal at index $column",
        )
        @inbounds matrix.nzval[diagonal] += inertia[column]
    end
    return matrix, inertia
end

function _refresh_variable_diffusion_matrix_from_inertia!(
        matrix::SparseMatrixCSC{T, Int},
        operator::VariableStaggeredDiffusionOperator{T}, inertia, dt,
    ) where {T}
    length(inertia) == length(operator.volumes) || throw(
        DimensionMismatch("one inertia value per staggered volume is required"),
    )
    matrix.colptr == operator.stiffness.colptr &&
        rowvals(matrix) == rowvals(operator.stiffness) || throw(
        ArgumentError("diffusion matrix and stiffness sparsity differ"),
    )
    matrix_values = nonzeros(matrix)
    stiffness_values = nonzeros(operator.stiffness)
    half_step = T(dt) / 2
    for index in eachindex(matrix_values)
        @inbounds matrix_values[index] = half_step * stiffness_values[index]
    end
    rows = rowvals(matrix)
    for column in axes(matrix, 2)
        diagonal = 0
        for pointer in nzrange(matrix, column)
            if rows[pointer] == column
                diagonal = pointer
                break
            end
        end
        iszero(diagonal) && error(
            "variable diffusion stiffness has no diagonal at index $column",
        )
        @inbounds matrix_values[diagonal] += inertia[column]
    end
    return matrix
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
    for column in eachindex(solver.inertia)
        @inbounds solver.inertia[column] =
            T(density[column]) * operator.volumes[column]
    end
    return _refresh_variable_diffusion_matrix_from_inertia!(
        solver.matrix, operator, solver.inertia, dt,
    )
end

function VariableStaggeredDiffusionSolver(
        operator::VariableStaggeredDiffusionOperator{T}, density, dt;
        krylov = Krylov.BicgstabWorkspace, krylov_options = (;),
        preconditioner = :tree_geometric, preconditioner_options = (;),
        coarse_size = 64, pre_sweeps = 2, post_sweeps = 2,
        max_levels = 32, reuse_preconditioner = nothing,
        reuse_solver = nothing,
    ) where {T}
    matrix_free_fine = size(operator.stiffness) !=
        (length(operator.volumes), length(operator.volumes))
    matrix, inertia = if matrix_free_fine
        nothing, T.(density) .* operator.volumes
    else
        _variable_diffusion_matrix(operator, density, dt)
    end
    right_hand_side = zeros(T, length(inertia))
    action = FactorizedVariableDiffusionOperator(operator, inertia, dt)
    krylov_workspace = _construct_krylov(
        krylov, action, right_hand_side
    )
    built_preconditioner = _variable_diffusion_preconditioner(
        preconditioner, operator, matrix, inertia, dt;
        coarse_size, pre_sweeps, post_sweeps, max_levels,
        options = preconditioner_options,
        reuse = reuse_preconditioner,
    )
    if !isnothing(reuse_solver)
        reconfigured = _reconfigure_variable_diffusion_solver!(
            reuse_solver, operator, matrix, inertia, dt,
            built_preconditioner,
        )
        isnothing(reconfigured) || return reconfigured
    end
    return VariableStaggeredDiffusionSolver{
        T, typeof(krylov), typeof(krylov_options),
        typeof(preconditioner), typeof(preconditioner_options),
        typeof(action),
        typeof(krylov_workspace), typeof(built_preconditioner),
    }(
        krylov, krylov_options, preconditioner,
        preconditioner_options, coarse_size, pre_sweeps,
        post_sweeps, max_levels, matrix, action, krylov_workspace,
        built_preconditioner, inertia, zeros(T, length(inertia)),
        right_hand_side, zeros(T, length(inertia)),
        zeros(T, length(inertia)),
        LinearSolverStats(
            T; method = _krylov_method(krylov_workspace),
            preconditioner = _preconditioner_name(built_preconditioner)
        )
    )
end

"""Retarget package-owned diffusion solver storage after an AMR regrid."""
function _reconfigure_variable_diffusion_solver!(
        solver::VariableStaggeredDiffusionSolver{T},
        operator::VariableStaggeredDiffusionOperator{T}, matrix, inertia, dt,
        preconditioner,
    ) where {T}
    typeof(operator) === typeof(solver.action.operator) || return nothing
    typeof(preconditioner) === typeof(solver.preconditioner) || return nothing
    solver.matrix = matrix
    solver.inertia = inertia
    action = solver.action
    action.operator = operator
    action.inertia = inertia
    action.half_step = T(dt) / 2
    resize!(action.stress_work, length(operator.stress_weights))
    fill!(action.stress_work, zero(T))
    number_of_unknowns = length(inertia)
    for values in (
            solver.solution, solver.right_hand_side,
            solver.residual, solver.product,
        )
        resize!(values, number_of_unknowns)
        fill!(values, zero(T))
    end
    krylov = _resize_krylov_workspace!(solver.krylov, number_of_unknowns)
    isnothing(krylov) && (krylov = _construct_krylov(
        solver.krylov_factory, action, solver.right_hand_side,
    ))
    typeof(krylov) === typeof(solver.krylov) || return nothing
    solver.krylov = krylov
    solver.preconditioner = preconditioner
    solver.stats = LinearSolverStats(
        T; method = _krylov_method(krylov),
        preconditioner = _preconditioner_name(preconditioner),
    )
    return solver
end

function _update_variable_diffusion_solver!(
        solver::VariableStaggeredDiffusionSolver,
        operator, density, dt
    )
    matrix = if isnothing(solver.matrix)
        length(density) == length(operator.volumes) || throw(
            DimensionMismatch("one density per staggered volume is required"),
        )
        for column in eachindex(solver.inertia)
            @inbounds solver.inertia[column] =
                eltype(solver.inertia)(density[column]) * operator.volumes[column]
        end
        nothing
    else
        _refresh_variable_diffusion_matrix!(solver, operator, density, dt)
    end
    solver.action.operator = operator
    solver.action.half_step = eltype(solver.inertia)(dt) / 2
    if solver.preconditioner_specification in (:geometric, :tree_geometric) &&
            solver.preconditioner isa TreeVariableDiffusionPreconditioner
        _refresh_tree_variable_diffusion_preconditioner!(
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
    mul!(solver.product, solver.action, solver.solution)
    for index in eachindex(solver.residual)
        @inbounds solver.residual[index] =
            solver.right_hand_side[index] - solver.product[index]
    end
    volumes = operator.volumes
    initial_l2, initial_linf = _integrated_residual_norms(
        solver.residual, volumes
    )
    roundoff_floor = _roundoff_residual_floor!(
        solver.action, solver.solution, solver.right_hand_side,
        volumes, solver.product
    )
    target = max(
        T(atol), T(rtol) * max(initial_linf, eps(T)), roundoff_floor
    )
    if initial_linf > target
        _run_krylov!(
            solver.krylov, solver.action, solver.right_hand_side,
            solver.solution, solver.preconditioner, target,
            volumes, max_cycles, solver.krylov_options
        )
        copyto!(solver.solution, Krylov.solution(solver.krylov))
    end
    mul!(solver.product, solver.action, solver.solution)
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
            solver.action, solver.solution,
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
    _factorized_variable_stiffness_mul!(
        solver.product, operator, old_velocity.interior,
        solver.action.stress_work,
    )
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
