"""Shared staggered-mass and incidence utilities for two-velocity projections."""
function _mimetic_incidence(mesh::BlockTreeMesh, ::Type{T}) where {T}
    nx, ny = mesh.block_size
    rows = Int[]
    columns = Int[]
    values = T[]
    for (face_index, face) in pairs(mesh.faces)
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
        push!(rows, face_index, face_index)
        push!(columns, minus, plus)
        push!(values, -T(face.area), T(face.area))
    end
    return sparse(
        rows, columns, values, length(mesh.faces), ncells(mesh),
    )
end

function _mimetic_global_face_fractions!(
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
                workspace.liquid_fraction_x.interior[local_x]
        else
            local_y += 1
            @inbounds result[index] =
                workspace.liquid_fraction_y.interior[local_y]
        end
    end
    return result
end

function _mimetic_active_cells(interface::InterfaceField)
    geometry = interface.geometry
    return isnothing(geometry) ? trues(ncells(interface.mesh)) :
        copy(geometry.active_cells)
end

function _gauge_free_component_partition(
        mesh::BlockTreeMesh, eligible::AbstractVector{Bool}, face_weights,
    )
    nx, ny = mesh.block_size
    adjacency = [Int[] for _ in 1:ncells(mesh)]
    for (index, face) in pairs(mesh.faces)
        iszero(@inbounds face_weights[index]) && continue
        minus = _elliptic_cell_index(face.minus, nx, ny)
        plus = _elliptic_cell_index(face.plus, nx, ny)
        eligible[minus] && eligible[plus] || continue
        push!(adjacency[minus], plus)
        push!(adjacency[plus], minus)
    end
    retained = Int[]
    anchor_for_retained = Int[]
    visited = falses(length(eligible))
    stack = Int[]
    for root in eachindex(eligible)
        eligible[root] && !visited[root] || continue
        empty!(stack)
        push!(stack, root)
        visited[root] = true
        component = Int[]
        while !isempty(stack)
            cell = pop!(stack)
            push!(component, cell)
            for neighbour in adjacency[cell]
                visited[neighbour] && continue
                visited[neighbour] = true
                push!(stack, neighbour)
            end
        end
        length(component) > 1 || continue
        anchor = minimum(component)
        for cell in component
            cell == anchor && continue
            push!(retained, cell)
            push!(anchor_for_retained, anchor)
        end
    end
    permutation = sortperm(retained)
    return retained[permutation], anchor_for_retained[permutation]
end

function _gauge_free_component_rows(
        mesh::BlockTreeMesh, eligible::AbstractVector{Bool}, face_weights,
    )
    retained, = _gauge_free_component_partition(
        mesh, eligible, face_weights,
    )
    return retained
end

function _mimetic_face_data(
        solver, workspace,
    )
    T = eltype(solver.interface)
    interface = solver.interface
    reconstruct!(interface)
    number_of_faces = length(interface.mesh.faces)
    liquid_fraction = zeros(T, number_of_faces)
    _mimetic_global_face_fractions!(liquid_fraction, workspace, interface)
    liquid_mass = solver.density[1] .* liquid_fraction
    gas_mass = solver.density[2] .* (one(T) .- liquid_fraction)
    total_mass = liquid_mass .+ gas_mass
    liquid_aperture = zeros(T, number_of_faces)
    gas_aperture = zeros(T, number_of_faces)
    for index in 1:number_of_faces
        liquid, gas = _phase_face_apertures(
            interface, interface.geometry, index,
        )
        @inbounds begin
            liquid_aperture[index] = liquid
            gas_aperture[index] = gas
        end
    end
    zeta_liquid = .-gas_mass
    zeta_gas = liquid_mass
    aperture_zeta = liquid_aperture .* zeta_liquid .+
        gas_aperture .* zeta_gas
    coupling = aperture_zeta ./ total_mass
    relative_mass = liquid_mass .* gas_mass ./ total_mass
    aperture_jump = gas_aperture .- liquid_aperture
    floor = interface.fraction_floor
    # For an exactly half-filled open face the normal-constraint coefficient
    # vanishes.  PLIC clipping can leave roundoff-sized remnants here.  If
    # retained, those remnants create false graph connections when choosing
    # the multiplier gauge and make the KKT system numerically singular.
    # Remove roundoff only; this threshold is deliberately independent of the
    # physical volume-fraction floor.
    for index in eachindex(aperture_jump)
        open_aperture = liquid_aperture[index] + gas_aperture[index]
        threshold = T(64) * eps(T) * max(open_aperture, one(T))
        abs(aperture_jump[index]) <= threshold &&
            (aperture_jump[index] = zero(T))
    end
    mixed = findall(eachindex(total_mass)) do index
        liquid_fraction[index] > floor &&
            liquid_fraction[index] < one(T) - floor &&
            liquid_aperture[index] + gas_aperture[index] > zero(T)
    end
    return (;
        liquid_fraction, liquid_mass, gas_mass, total_mass,
        liquid_aperture, gas_aperture, zeta_liquid, zeta_gas,
        coupling, relative_mass, aperture_jump, mixed,
    )
end

function _mimetic_pressure_operator(
        solver,
    )
    interface = solver.interface
    isnothing(interface.geometry) &&
        return PoissonOperator(interface; density = solver.density)

    # The two-velocity projection minimizes the correction in the phase
    # kinetic norm of the *open staggered dual volume*.  Its common pressure
    # correction therefore has the coefficient A/rho, where A is the total
    # open aperture.  The one-velocity symmetry-preserving cut-cell operator
    # uses A^2/rho and is intentionally a different discretization.
    geometry = interface.geometry
    density = mixture_density(interface, solver.density, geometry)
    coefficient = geometry.aperture ./ density
    T = eltype(interface)
    return EllipticOperator(
        interface.mesh; reaction = 0, diffusion = 1,
        face_coefficient = coefficient,
        gradient_operator = GradientOperator(geometry),
        volumes = _geometry_cell_volumes(geometry),
        active_cells = geometry.active_cells, T,
    )
end

function _mimetic_boundary_constraints(
        interface::InterfaceField{T}, velocity::TwoPhaseVelocity,
    ) where {T}
    mesh = interface.mesh
    nx, ny = mesh.block_size
    mixture = zeros(T, ncells(mesh))
    normal_jump = zeros(T, ncells(mesh))
    for (index, face) in pairs(mesh.boundary_faces)
        liquid_aperture, gas_aperture =
            _phase_boundary_face_apertures(interface, index)
        cell = _elliptic_cell_index(face.cell, nx, ny)
        factor = T(face.side) * T(face.area)
        liquid = velocity[1].boundary[index]
        gas = velocity[2].boundary[index]
        @inbounds begin
            mixture[cell] += factor * (
                liquid_aperture * liquid + gas_aperture * gas
            )
            normal_jump[cell] += factor *
                (gas_aperture - liquid_aperture) * (gas - liquid)
        end
    end
    return mixture, normal_jump
end

function _copy_mimetic_cell_solution!(field::CellField{T}, values) where {T}
    for block in eachindex(field.blocks)
        fill!(interior(field, block), zero(T))
    end
    references = _cell_references(field.mesh)
    for (index, value) in values
        _set!(field, T(value), references[index])
    end
    fill_halo!(field)
    return field
end

function _mimetic_weighted_gradient!(
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

"""Reconstruct both phase pressure gradients from the latest solve."""
function _mimetic_integrated_constraints(
        incidence, data, interface, velocity, mixed_faces,
    )
    liquid = velocity[1].interior
    gas = velocity[2].interior
    mixture = data.liquid_aperture .* liquid .+
        data.gas_aperture .* gas
    # The face jump is a degree of freedom only where both staggered phase
    # masses are present.  As in the derivation, extend it by zero elsewhere;
    # inactive-phase storage must never enter the physical constraint.
    jump_flux = zeros(eltype(liquid), length(liquid))
    jump_flux[mixed_faces] .= data.aperture_jump[mixed_faces] .*
        (gas[mixed_faces] .- liquid[mixed_faces])
    boundary_mixture, boundary_normal =
        _mimetic_boundary_constraints(interface, velocity)
    return -transpose(incidence) * mixture .+ boundary_mixture,
        -transpose(incidence) * jump_flux .+ boundary_normal
end

function _mimetic_strong_maximum(values, volumes)
    result = zero(eltype(values))
    for index in eachindex(values)
        volumes[index] > 0 || continue
        @inbounds result = max(result, abs(values[index]) / volumes[index])
    end
    return result
end
