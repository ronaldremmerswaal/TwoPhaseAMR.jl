function _variable_staggered_diffusion(
        mesh, viscosity::CellField, boundary, ::Type{T};
        cache = VariableDiffusionConstructionCache(T), reuse = nothing,
        cache_layer = 1, geometry = nothing,
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
    isnothing(geometry) || geometry.mesh === mesh || throw(
        ArgumentError("geometry and variable diffusion must share a mesh"),
    )
    !isnothing(geometry) &&
        !(boundary isa Union{NoSlip, FreeSlip}) && throw(
        ArgumentError(
            "embedded variable diffusion requires NoSlip() or FreeSlip()",
        ),
    )
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
    face_search = DiffusionFaceSearch(cache, mesh)
    patch_layer = _prepare_shear_patch_layer!(cache, mesh, cache_layer)
    derivative_points = cache.derivative_points
    derivative_candidates = cache.derivative_candidates
    derivative_indices = cache.derivative_indices
    derivative_coefficients = cache.derivative_coefficients
    for ref in refs,axis in 1:2
        row = length(measures) + 1
        open = isnothing(geometry) ? one(T) :
            _get(geometry.open_fraction, ref)
        push!(measures, 2 * open * mesh.blocks[ref.block].volume)
        push!(interpolation_rows, row)
        push!(interpolation_columns, _cell_index(ref, nx, ny))
        push!(interpolation_values, one(T))
    end
    for (i, face) in pairs(mesh.faces)
        for (ref, sign) in ((face.minus, 1), (face.plus, -1))
            open = isnothing(geometry) ? one(T) :
                _get(geometry.open_fraction, ref)
            open > zero(T) || continue
            row = 2 * (_cell_index(ref, nx, ny) - 1) + face.axis
            push!(rows, row);push!(cols, i)
            aperture = isnothing(geometry) ? one(T) : geometry.aperture[i]
            push!(
                vals,
                sign * aperture * face.area /
                    (open * mesh.blocks[ref.block].volume),
            )
        end
    end
    vertices, lattice = _stress_vertices!(cache, mesh, T)
    empty!(cache.vertex_keys)
    append!(cache.vertex_keys, keys(vertices))
    sort!(cache.vertex_keys)
    for key in cache.vertex_keys
        volume, cells = vertices[key]
        point = ntuple(a -> mesh.origin[a] + key[a] * mesh.extent[a] / lattice[a], 2)
        open_volume = isnothing(geometry) ? volume : sum(
                weight * _get(geometry.open_fraction, ref)
                for (ref, weight) in cells;
                init = zero(T),
            )
        open_volume > zero(T) || continue
        push!(measures, open_volume);row = length(measures)
        for (ref, raw_weight) in cells
            weight = isnothing(geometry) ? raw_weight :
                raw_weight * _get(geometry.open_fraction, ref)
            weight > zero(T) || continue
            push!(interpolation_rows, row)
            push!(interpolation_columns, _cell_index(ref, nx, ny))
            push!(interpolation_values, weight / open_volume)
        end
        scale = maximum(maximum(mesh.blocks[r.block].spacing) for (r, w) in cells)
        for component in 1:2
            indices, coefficients = _cached_vertex_derivative!(
                mesh, key, point, component, scale, boundary, T,
                face_search, derivative_points, derivative_candidates,
                derivative_indices, derivative_coefficients, cache,
                patch_layer,
            )
            for index in eachindex(indices)
                push!(rows, row)
                push!(cols, indices[index])
                push!(vals, coefficients[index])
            end
        end
    end
    if !isnothing(geometry)
        for (face_index, conductance, cell_index) in
            _embedded_variable_wall_stresses(
                geometry; include_cut_wall = boundary isa NoSlip,
            )
            row = length(measures) + 1
            push!(measures, conductance)
            push!(interpolation_rows, row)
            push!(interpolation_columns, cell_index)
            push!(interpolation_values, one(T))
            push!(rows, row)
            push!(cols, face_index)
            push!(vals, one(T))
        end
    end
    strain = sparse(rows, cols, vals, length(measures), length(mesh.faces))
    strain_transpose = copy(transpose(strain))
    interpolation = sparse(
        interpolation_rows, interpolation_columns, interpolation_values,
        length(measures), length(refs)
    )
    interpolation_transpose = copy(transpose(interpolation))
    cell_work = isnothing(reuse) ? T[] : reuse.cell_work
    resize!(cell_work, length(refs))
    for (index, ref) in pairs(refs)
        @inbounds cell_work[index] = log(_get(viscosity, ref))
    end
    stress_work = isnothing(reuse) ? T[] : reuse.stress_work
    resize!(stress_work, length(measures))
    _parallel_csr_row_mul!(
        stress_work, interpolation_transpose, cell_work,
    )
    weights = isnothing(reuse) ? T[] : reuse.stress_weights
    resize!(weights, length(measures))
    for index in eachindex(weights)
        @inbounds weights[index] = measures[index] * exp(stress_work[index])
    end
    stiffness_pointers = isnothing(reuse) ? Int[] : reuse.stiffness_pointers
    stiffness_stresses = isnothing(reuse) ? Int[] : reuse.stiffness_stresses
    stiffness_factors = isnothing(reuse) ? T[] : reuse.stiffness_factors
    stiffness_refresh_offsets = isnothing(reuse) ? Int[] :
        reuse.stiffness_refresh_offsets
    for values in (stiffness_pointers, stiffness_stresses, stiffness_factors)
        empty!(values)
    end
    stiffness = _complete_symmetric_stiffness(strain, weights)
    stiffness_pointers, stiffness_stresses, stiffness_factors,
        stiffness_refresh_offsets =
        _stiffness_refresh_map!(
        stiffness_pointers, stiffness_stresses, stiffness_factors,
        stiffness_refresh_offsets,
        cache.stiffness_entries, strain, stiffness,
    )
    volumes = isnothing(reuse) ? T[] : reuse.volumes
    resize!(volumes, length(mesh.faces))
    if isnothing(geometry)
        # The operator may resize/reuse its own vector, so copy the mesh's
        # canonical uncut volumes rather than aliasing that topology cache.
        copyto!(volumes, mesh.face_dual_volumes)
    else
        copyto!(volumes, _geometry_face_dual_volumes(geometry))
    end
    _release_stress_vertices!(cache)
    return VariableStaggeredDiffusionOperator(
        mesh, stiffness, volumes, boundary, strain, strain_transpose,
        interpolation, interpolation_transpose,
        measures, weights, cell_work, stress_work,
        stiffness_pointers, stiffness_stresses, stiffness_factors,
        stiffness_refresh_offsets,
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
    _parallel_csr_row_mul!(
        operator.stress_work, operator.stress_interpolation_transpose,
        operator.cell_work,
    )
    @_parallel for index in 1:length(operator.stress_weights)
        @inbounds operator.stress_weights[index] =
            operator.stress_measures[index] * exp(operator.stress_work[index])
    end
    isempty(operator.stiffness_pointers) && return operator
    stiffness_values = nonzeros(operator.stiffness)
    offsets = operator.stiffness_refresh_offsets
    @_parallel for pointer in 1:length(stiffness_values)
        value = zero(T)
        for index in offsets[pointer]:(offsets[pointer + 1] - 1)
            stress = @inbounds operator.stiffness_stresses[index]
            factor = @inbounds operator.stiffness_factors[index]
            @inbounds value += operator.stress_weights[stress] * factor
        end
        @inbounds stiffness_values[pointer] = value
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
    mesh = field.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do b, first_j, last_j
        fraction = @inbounds field.conditional_fraction.blocks[b]
        offset = (b - 1) * cells_per_block
        @inbounds for j in first_j:last_j, i in 1:nx
            index = offset + (j - 1) * nx + i
            operator.cell_work[index] = muladd(
                fraction[i + h, j + h], log_jump, log_gas,
            )
        end
    end
    return _update_log_dynamic_viscosity!(operator, operator.cell_work)
end

function _staggered_diffusion_operator(
        field::InterfaceField; viscosity::NTuple{2}, boundary = NoSlip(),
        construction_cache = nothing, reuse = nothing,
        cache_layer = 1,
    )
    mu = dynamic_viscosity(field; viscosity)
    return _staggered_diffusion_operator(
        field.mesh; viscosity = mu, boundary, construction_cache, reuse,
        cache_layer, geometry = field.geometry,
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
    result = destination.interior
    volumes = operator.volumes
    @_parallel for index in 1:length(result)
        @inbounds result[index] = -result[index] / volumes[index]
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
    _parallel_csr_row_mul!(
        stress_work, operator.strain_transpose, source,
    )
    stress_weights = operator.stress_weights
    @_parallel for index in 1:length(stress_work)
        @inbounds stress_work[index] *= stress_weights[index]
    end
    _parallel_csr_row_mul!(destination, operator.strain, stress_work)
    return destination
end

"""Allocation-free fine-grid action for implicit variable-stress diffusion.

This evaluates `M_rho*x + dt/2*E'*W_mu*E*x` without materializing a new
coefficient-dependent sparse matrix.  The assembled matrix remains available
for the geometric hierarchy; this factorized form provides the exact Krylov
action, an equivalence reference, and a low-level matrix-free building block.
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
    inertia = action.inertia
    @_parallel for index in 1:length(destination)
        @inbounds destination[index] = muladd(
            half_step, destination[index],
            inertia[index] * source[index],
        )
    end
    return destination
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
