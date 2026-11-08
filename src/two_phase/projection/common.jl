"""Face data shared by both two-velocity projection methods."""
const _ProjectionFaceData{T} = NamedTuple{
    (
        :liquid_fraction, :liquid_mass, :gas_mass, :total_mass,
        :liquid_aperture, :gas_aperture, :zeta_liquid, :zeta_gas,
        :coupling, :relative_mass, :aperture_jump, :aperture_zeta, :mixed,
    ),
    Tuple{
        Vector{T}, Vector{T}, Vector{T}, Vector{T},
        Vector{T}, Vector{T}, Vector{T}, Vector{T},
        Vector{T}, Vector{T}, Vector{T}, Vector{T}, Vector{Int},
    },
}

"""Build the oriented, face-area-weighted incidence matrix, reusing CSC storage."""
function _projection_incidence!(matrix::SparseMatrixCSC{T}, counts, mesh) where {T}
    # Face traversal is in increasing global row order, so each column is
    # already sorted. Periodic self-edges retain the explicit triplet zero.
    nx, ny = mesh.block_size
    resize!(counts, ncells(mesh))
    fill!(counts, 0)
    for face in mesh.faces
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        counts[minus] += 1
        minus == plus || (counts[plus] += 1)
    end
    columns = matrix.colptr
    _csr_offsets(counts, columns)
    copyto!(counts, 1, columns, 1, length(counts))
    rows, values = rowvals(matrix), nonzeros(matrix)
    resize!(rows, columns[end] - 1)
    resize!(values, length(rows))
    for (index, face) in pairs(mesh.faces)
        minus = _cell_index(face.minus, nx, ny)
        plus = _cell_index(face.plus, nx, ny)
        pointer = counts[minus]
        rows[pointer] = index
        values[pointer] = minus == plus ? zero(T) : -T(face.area)
        counts[minus] += 1
        if minus != plus
            pointer = counts[plus]
            rows[pointer] = index
            values[pointer] = T(face.area)
            counts[plus] += 1
        end
    end
    return size(matrix) == (length(mesh.faces), ncells(mesh)) ? matrix :
        SparseMatrixCSC(length(mesh.faces), ncells(mesh), columns, rows, values)
end

"""Partition eligible cells by open-face connectivity and remove one gauge per component.

The scratch labels become a cell-to-retained-index lookup. `nothing` for the
eligibility mask or face weights means all cells or all faces, respectively.
"""
function _projection_gauge_partition!(
        retained, anchors, labels, stack, incidence, mesh,
        eligible, face_weights,
    )
    resize!(labels, ncells(mesh))
    fill!(labels, 0)
    empty!(retained)
    empty!(anchors)
    nx, ny = mesh.block_size
    for root in eachindex(labels)
        (!isnothing(eligible) && !eligible[root]) && continue
        iszero(labels[root]) || continue
        empty!(stack)
        push!(stack, root)
        labels[root] = root
        while !isempty(stack)
            cell = pop!(stack)
            for pointer in nzrange(incidence, cell)
                face_index = rowvals(incidence)[pointer]
                (!isnothing(face_weights) && iszero(face_weights[face_index])) && continue
                face = mesh.faces[face_index]
                minus = _cell_index(face.minus, nx, ny)
                plus = _cell_index(face.plus, nx, ny)
                neighbour = cell == minus ? plus : minus
                iszero(labels[neighbour]) || continue
                (!isnothing(eligible) && !eligible[neighbour]) && continue
                labels[neighbour] = root
                push!(stack, neighbour)
            end
        end
    end
    # Roots are visited in increasing cell order, hence are component minima.
    # Iterating the labels also produces sorted retained indices directly.
    for cell in eachindex(labels)
        anchor = labels[cell]
        if anchor > 0 && anchor != cell
            push!(retained, cell)
            push!(anchors, anchor)
        end
    end
    fill!(labels, 0)
    for (local_index, cell) in pairs(retained)
        labels[cell] = local_index
    end
    return nothing
end

"""Sum two weighted phase values without evaluating an absent phase product."""
@inline function _projection_phase_sum(liquid_weight, liquid, gas_weight, gas)
    return _phase_product(liquid_weight, liquid) + _phase_product(gas_weight, gas)
end

function _projection_global_face_fractions!(
        result, workspace,
        interface::InterfaceField,
    )
    _phase_face_average!(
        workspace.liquid_fraction_x, interface.conditional_fraction,
        interface.geometry,
    )
    _phase_face_average!(
        workspace.liquid_fraction_y, interface.conditional_fraction,
        interface.geometry,
    )
    local_x = local_y = 0
    for (index, face) in pairs(interface.mesh.faces)
        if face.axis == 1
            local_x += 1
            @inbounds result[index] =
                clamp(workspace.liquid_fraction_x.interior[local_x], zero(eltype(result)), one(eltype(result)))
        else
            local_y += 1
            @inbounds result[index] =
                clamp(workspace.liquid_fraction_y.interior[local_y], zero(eltype(result)), one(eltype(result)))
        end
    end
    return result
end

function _projection_face_data(
        solver, workspace; storage = nothing,
    )
    T = eltype(solver.interface)
    interface = solver.interface
    reconstruct!(interface)
    number_of_faces = length(interface.mesh.faces)
    arrays = isnothing(storage) ? ntuple(_ -> Vector{T}(undef, number_of_faces), 10) :
        (
            storage.liquid_fraction, storage.liquid_mass, storage.gas_mass,
            storage.total_mass, storage.zeta_liquid, storage.zeta_gas,
            storage.coupling, storage.relative_mass, storage.aperture_jump,
            storage.aperture_zeta,
        )
    for values in arrays
        resize!(values, number_of_faces)
    end
    liquid_fraction, liquid_mass, gas_mass, total_mass, zeta_liquid,
        zeta_gas, coupling, relative_mass, aperture_jump, aperture_zeta = arrays
    _projection_global_face_fractions!(liquid_fraction, workspace, interface)
    # Transport has moved the interface since its CFL froze the openings. This
    # updates the physical apertures for the new state; the next CFL copies them.
    liquid_aperture, gas_aperture = _current_phase_face_apertures!(interface)
    for index in eachindex(liquid_fraction)
        @inbounds begin
            liquid_mass[index] = solver.density[1] * liquid_fraction[index]
            gas_mass[index] = solver.density[2] * (one(T) - liquid_fraction[index])
            total_mass[index] = liquid_mass[index] + gas_mass[index]
            zeta_liquid[index] = -gas_mass[index]
            zeta_gas[index] = liquid_mass[index]
            aperture_zeta[index] = liquid_aperture[index] * zeta_liquid[index] +
                gas_aperture[index] * zeta_gas[index]
            coupling[index] = aperture_zeta[index] / total_mass[index]
            relative_mass[index] = liquid_mass[index] * gas_mass[index] / total_mass[index]
            aperture_jump[index] = gas_aperture[index] - liquid_aperture[index]
        end
    end
    floor = interface.fraction_floor
    # Exactly half-filled open faces have a zero normal-constraint coefficient.
    # Remove clipping roundoff without using the physical fraction floor.
    for index in eachindex(aperture_jump)
        open_aperture = liquid_aperture[index] + gas_aperture[index]
        threshold = T(64) * eps(T) * max(open_aperture, one(T))
        abs(aperture_jump[index]) <= threshold &&
            (aperture_jump[index] = zero(T))
    end
    mixed = isnothing(storage) ? Int[] : empty!(storage.mixed)
    for index in eachindex(total_mass)
        if liquid_fraction[index] > floor &&
                liquid_fraction[index] < one(T) - floor &&
                liquid_aperture[index] + gas_aperture[index] > zero(T)
            push!(mixed, index)
        end
    end
    return (;
        liquid_fraction, liquid_mass, gas_mass, total_mass,
        liquid_aperture, gas_aperture, zeta_liquid, zeta_gas,
        coupling, relative_mass, aperture_jump, aperture_zeta, mixed,
    )
end

function _projection_pressure_operator(
        solver; template = nothing,
        construction_cache = GradientConstructionCache(eltype(solver.interface)),
    )
    interface = solver.interface
    isnothing(interface.geometry) &&
        return PoissonOperator(
        interface; density = solver.density,
        template, construction_cache
    )

    # The two-velocity projection minimizes the correction in the phase
    # kinetic norm of the open staggered dual volume. Its common pressure
    # coefficient is A/rho; the one-velocity cut-cell coefficient A^2/rho
    # is intentionally a different discretization.
    geometry = interface.geometry
    density = mixture_density(interface, solver.density, geometry)
    coefficient = geometry.aperture ./ density
    isnothing(template) || return _replace_face_coefficient(template, coefficient)
    T = eltype(interface)
    return EllipticOperator(
        interface.mesh; reaction = 0, diffusion = 1,
        face_coefficient = coefficient,
        gradient_operator = GradientOperator(geometry; construction_cache),
        volumes = _geometry_cell_volumes(geometry),
        active_cells = geometry.active_cells, T,
    )
end

function _copy_projection_cell_solution!(field::CellField{T}, values) where {T}
    for block in eachindex(field.blocks)
        fill!(interior(field, block), zero(T))
    end
    nx, ny = field.mesh.block_size
    for (index, value) in values
        block = div(index - 1, nx * ny) + 1
        local_index = rem(index - 1, nx * ny)
        reference = CellRef(block, rem(local_index, nx) + 1, div(local_index, nx) + 1)
        _set!(field, T(value), reference)
    end
    fill_halo!(field)
    return field
end

function _projection_weighted_gradient!(
        destination::FaceVelocity, pressure::CellField,
        operator::EllipticOperator, work::EllipticWorkspace,
        geometry,
    )
    if isnothing(geometry)
        weighted_gradient!(destination, pressure, operator, work)
    else
        _cut_cell_weighted_gradient!(
            destination, pressure, operator, work, geometry,
        )
    end
    return destination
end

"""Reconstruct phase gradients after the caller validates fields and sizes scratch."""
function _projection_gradient_core!(
        destination, pressure, solver, workspace,
        pressure_operator, data, elliptic
    )
    T = eltype(solver.interface)
    _projection_weighted_gradient!(
        workspace.pressure_gradient, pressure, pressure_operator, elliptic,
        solver.interface.geometry,
    )
    gradient!(
        workspace.base_gradient, pressure,
        pressure_operator.gradient_operator,
    )
    for index in eachindex(solver.interface.mesh.faces)
        mass = @inbounds data.total_mass[index]
        jump = @inbounds workspace.jump[index]
        common = @inbounds workspace.pressure_gradient.interior[index]
        @inbounds begin
            destination[1].interior[index] = common +
                data.zeta_liquid[index] / mass * jump
            destination[2].interior[index] = common +
                data.zeta_gas[index] / mass * jump
        end
    end
    fill!(destination[1].boundary, zero(T))
    fill!(destination[2].boundary, zero(T))
    return destination
end

function _projection_correct_phase_velocities!(velocity, gradients, dt, ::Type{T}) where {T}
    for phase in 1:2
        for index in eachindex(velocity[phase].interior)
            isnan(velocity[phase].interior[index]) && continue
            @inbounds velocity[phase].interior[index] -=
                T(dt) * gradients[phase].interior[index]
        end
    end
    return velocity
end

"""Maximum pressure-row residual divided by the corresponding cell volume.

CMDGFM skips nonpositive volumes; the mimetic KKT residual divides every
retained row. The compile-time policy keeps those distinct semantics.
"""
function _projection_pressure_residual_maximum(
        residual, indices, volumes,
        ::Val{skip_nonpositive}
    ) where {skip_nonpositive}
    T = eltype(residual)
    pressure = zero(T)
    for local_index in 1:length(indices)
        cell = @inbounds indices[local_index]
        volume = @inbounds volumes[cell]
        skip_nonpositive && !(volume > zero(T)) && continue
        @inbounds pressure = max(
            pressure, abs(residual[local_index]) / volume,
        )
    end
    return pressure
end

@inline function _projection_momentum_identity_error(
        liquid_mass, gas_mass, liquid_gradient, gas_gradient, pressure_gradient,
    )
    weighted = liquid_mass * liquid_gradient + gas_mass * gas_gradient
    expected = (liquid_mass + gas_mass) * pressure_gradient
    return abs(weighted - expected)
end

function _projection_strong_maximum(values, volumes)
    result = zero(eltype(values))
    for index in eachindex(values)
        volumes[index] > 0 || continue
        @inbounds result = max(result, abs(values[index]) / volumes[index])
    end
    return result
end
