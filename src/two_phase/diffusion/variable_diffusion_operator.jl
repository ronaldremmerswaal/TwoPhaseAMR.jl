function _staggered_diffusion_operator(
        mesh::BlockTreeMesh, viscosity::CellField, boundary, T;
        construction_cache = nothing, reuse = nothing, cache_layer = 1,
        geometry = nothing,
        averaging::AbstractViscosityAveraging = HarmonicViscosityAveraging(),
    )
    cache = isnothing(construction_cache) ?
        VariableDiffusionConstructionCache(T) : construction_cache
    return _variable_staggered_diffusion(
        mesh, viscosity, boundary, T;
        cache, reuse, cache_layer, geometry, averaging,
    )
end

function _variable_staggered_diffusion(
        mesh, viscosity::CellField, boundary, ::Type{T};
        cache = VariableDiffusionConstructionCache(T), reuse = nothing,
        cache_layer = 1, geometry = nothing,
        averaging::AbstractViscosityAveraging = HarmonicViscosityAveraging(),
    ) where {T}
    viscosity.mesh === mesh || throw(ArgumentError("viscosity field must share the operator mesh"))
    boundary isa FreeSlip || (boundary isa NoSlip && boundary.velocity isa StationaryWall) ||
        throw(ArgumentError("variable stress diffusion currently supports stationary NoSlip or FreeSlip walls"))
    all(c -> isfinite(c)&&c > 0, (v for b in eachindex(mesh.blocks) for v in interior(viscosity, b))) ||
        throw(ArgumentError("cell dynamic viscosity must be finite and strictly positive"))
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
    reaction_rows = cache.wall_reaction_rows
    reaction_cols = cache.wall_reaction_columns
    reaction_vals = cache.wall_reaction_values
    vals = cache.strain_values
    interpolation_rows = cache.interpolation_rows
    interpolation_columns = cache.interpolation_columns
    interpolation_values = cache.interpolation_values
    for values in (
            rows, cols, vals, reaction_rows, reaction_cols, reaction_vals,
            interpolation_rows, interpolation_columns, interpolation_values,
        )
        empty!(values)
    end
    dirichlet_faces = isnothing(geometry) ? Int[] :
        _embedded_variable_wall_constraints(geometry)
    reaction_column = cache.dirichlet_columns
    if !isempty(dirichlet_faces)
        resize!(reaction_column, length(mesh.faces))
        fill!(reaction_column, 0)
        for (column, face_index) in pairs(dirichlet_faces)
            reaction_column[face_index] = column
        end
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
        # An aligned solid face has zero open aperture, but the eliminated
        # normal-strain column represents motion of the wall itself. Use its
        # full area for the reaction; the column never enters active strain.
        aperture = if isnothing(geometry) ||
                (!isempty(dirichlet_faces) && !iszero(reaction_column[i]))
            one(T)
        else
            geometry.aperture[i]
        end
        for (ref, sign) in ((face.minus, 1), (face.plus, -1))
            open = isnothing(geometry) ? one(T) :
                _get(geometry.open_fraction, ref)
            open > zero(T) || continue
            row = 2 * (_cell_index(ref, nx, ny) - 1) + face.axis
            push!(rows, row);push!(cols, i)
            push!(
                vals,
                sign * aperture * face.area /
                    (open * mesh.blocks[ref.block].volume),
            )
        end
    end
    vertices, lattice = _stress_vertices!(cache, mesh, T)
    wall_vertices = isnothing(geometry) ? nothing :
        _embedded_variable_wall_stress_vertices(geometry, lattice)
    empty!(cache.vertex_keys)
    append!(cache.vertex_keys, keys(vertices))
    sort!(cache.vertex_keys)
    for key in cache.vertex_keys
        volume, cells = vertices[key]
        point = ntuple(a -> mesh.origin[a] + key[a] * mesh.extent[a] / lattice[a], 2)
        wall_cells = isnothing(wall_vertices) ? nothing : get(wall_vertices, key, nothing)
        open_volume = if isnothing(geometry)
            volume
        else
            sum(
                weight * _get(geometry.open_fraction, ref)
                    for (ref, weight) in cells
                    if isnothing(wall_cells) || !(_cell_index(ref, nx, ny) in wall_cells);
                init = zero(T),
            )
        end
        open_volume > zero(T) || continue
        push!(measures, open_volume);row = length(measures)
        for (ref, raw_weight) in cells
            !isnothing(wall_cells) && _cell_index(ref, nx, ny) in wall_cells &&
                continue
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
                dirichlet_faces,
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
    if !isempty(dirichlet_faces)
        # Separate the original prescribed columns before adding algebraic
        # rows. Keeping their stress-row indices makes the wall reaction
        # recoverable with the current, dynamically refreshed stress weights.
        retained = 0
        for index in eachindex(rows)
            column = reaction_column[cols[index]]
            if !iszero(column)
                push!(reaction_rows, rows[index])
                push!(reaction_cols, column)
                push!(reaction_vals, vals[index])
            else
                retained += 1
                rows[retained] = rows[index]
                cols[retained] = cols[index]
                vals[retained] = vals[index]
            end
        end
        resize!(rows, retained)
        resize!(cols, retained)
        resize!(vals, retained)
        for face_index in dirichlet_faces
            # An empty interpolation row gives exp(0) == 1 on every
            # viscosity refresh, so this stress is an isolated unit row.
            push!(measures, one(T))
            push!(rows, length(measures))
            push!(cols, face_index)
            push!(vals, one(T))
        end
    end
    strain = sparse(rows, cols, vals, length(measures), length(mesh.faces))
    wall_reaction_strain = sparse(
        reaction_rows, reaction_cols, reaction_vals,
        length(measures), length(dirichlet_faces),
    )
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
    cell_inverse_work = isnothing(reuse) ? T[] : reuse.cell_inverse_work
    resize!(cell_inverse_work, length(refs))
    weights = isnothing(reuse) ? T[] : reuse.stress_weights
    resize!(weights, length(measures))
    _fill_stress_weights!(
        weights, measures, stress_work, cell_inverse_work,
        interpolation_transpose, cell_work, averaging,
    )
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
        mesh, stiffness, volumes, boundary, averaging, strain, strain_transpose,
        wall_reaction_strain, dirichlet_faces, interpolation, interpolation_transpose,
        measures, weights, cell_work, cell_inverse_work, stress_work,
        stiffness_pointers, stiffness_stresses, stiffness_factors,
        stiffness_refresh_offsets,
    )
end

"""Fill the stress-row weights `measures .* row viscosity` from the per-cell
log viscosities, with the phase-viscosity averaging rule of the operator.

Every row of the strain map is a strain at a stress location whose viscosity
is mixed from the cell quarters that the row spans, with interpolation weights
`w_c` (the cell's share of the stress location, `sum_c w_c = 1`). A single-cell
row -- a normal-stress row or a wall row -- takes its cell's viscosity.

[`HarmonicViscosityAveraging`](@ref) uses the series conductance
`1 / sum_c (w_c / mu_c)`: a centred shear row differences velocities across
the cells of its vertex, so the tangential traction it transmits is continuous
only if those quarters act in series.
[`GeometricViscosityAveraging`](@ref) uses the thesis' `prod_c mu_c^w_c`,
i.e. the volume-fraction-weighted geometric mean evaluated with the
interpolated fraction.

A row without cells is the isolated prescribed-face algebraic row; it keeps
unit viscosity.
"""
function _fill_stress_weights!(
        weights, measures, stress_work, cell_inverse_work,
        interpolation_transpose, cell_log_viscosity,
        ::HarmonicViscosityAveraging,
    )
    @_parallel for index in eachindex(cell_inverse_work)
        @inbounds cell_inverse_work[index] = exp(-cell_log_viscosity[index])
    end
    _parallel_csr_row_mul!(
        stress_work, interpolation_transpose, cell_inverse_work,
    )
    @_parallel for index in eachindex(weights)
        @inbounds reciprocal = stress_work[index]
        @inbounds weights[index] =
            iszero(reciprocal) ? measures[index] :
            measures[index] / reciprocal
    end
    return weights
end

function _fill_stress_weights!(
        weights, measures, stress_work, cell_inverse_work,
        interpolation_transpose, cell_log_viscosity,
        ::GeometricViscosityAveraging,
    )
    _parallel_csr_row_mul!(
        stress_work, interpolation_transpose, cell_log_viscosity,
    )
    @_parallel for index in eachindex(weights)
        @inbounds weights[index] =
            measures[index] * exp(stress_work[index])
    end
    return weights
end

"""Refresh the stress-location mixture viscosity without rebuilding geometric
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
    _fill_stress_weights!(
        operator.stress_weights, operator.stress_measures,
        operator.stress_work, operator.cell_inverse_work,
        operator.stress_interpolation_transpose, operator.cell_work,
        operator.averaging,
    )
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
    # log of the cell mixture; the same rule mixes it at the stress rows.
    mesh = field.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    if operator.averaging isa HarmonicViscosityAveraging
        # log of the series mixture 1 / (c / liquid + (1 - c) / gas) is not
        # affine in the fraction, unlike the geometric one.
        inverse_liquid = inv(phases[1])
        inverse_jump = inverse_liquid - inv(phases[2])
        inverse_gas = inv(phases[2])
        _parallel_block_columns(mesh) do b, first_j, last_j
            fraction = @inbounds field.conditional_fraction.blocks[b]
            offset = (b - 1) * cells_per_block
            @inbounds for j in first_j:last_j, i in 1:nx
                index = offset + (j - 1) * nx + i
                operator.cell_work[index] = -log(
                    muladd(fraction[i + h, j + h], inverse_jump, inverse_gas),
                )
            end
        end
    else
        log_gas = log(phases[2])
        log_jump = log(phases[1]) - log_gas
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
    end
    return _update_log_dynamic_viscosity!(operator, operator.cell_work)
end

function _staggered_diffusion_operator(
        field::InterfaceField; viscosity::NTuple{2}, boundary = NoSlip(),
        construction_cache = nothing, reuse = nothing,
        cache_layer = 1,
        averaging::AbstractViscosityAveraging = HarmonicViscosityAveraging(),
    )
    mu = dynamic_viscosity(field; viscosity, averaging)
    return _staggered_diffusion_operator(
        field.mesh; viscosity = mu, boundary, construction_cache, reuse,
        cache_layer, geometry = field.geometry, averaging,
    )
end


"""
    StaggeredDiffusionOperator(field::InterfaceField;
                               viscosity, boundary=NoSlip(),
                               averaging=HarmonicViscosityAveraging())

Assemble the two-phase stress operator `div(2 mu D(u))` of an interface field.
`viscosity` is the `(liquid, gas)` dynamic-viscosity tuple and `averaging` the
phase-viscosity mixture rule, see [`HarmonicViscosityAveraging`](@ref).
"""
function StaggeredDiffusionOperator(
        field::InterfaceField; viscosity::NTuple{2}, boundary = NoSlip(),
        averaging::AbstractViscosityAveraging = HarmonicViscosityAveraging(),
    )
    return _staggered_diffusion_operator(
        field; viscosity, boundary, averaging,
    )
end

function _zero_variable_diffusion_constraints!(values, operator)
    for index in operator.dirichlet_faces
        @inbounds values[index] = zero(eltype(values))
    end
    return values
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
    _zero_variable_diffusion_constraints!(result, operator)
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
        right_hand_side, volumes, temporary, partials = nothing,
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
