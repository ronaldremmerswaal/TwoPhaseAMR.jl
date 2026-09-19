"""Set every face whose open aperture vanishes to the stationary-wall value."""

function enforce_geometry_no_slip!(
        velocity::FaceVelocity{T}, geometry::GeometryField{T},
    ) where {T}
    velocity.mesh === geometry.mesh || throw(
        ArgumentError("velocity and geometry must share a mesh"),
    )
    @_parallel for index in eachindex(velocity.interior)
        @inbounds iszero(geometry.aperture[index]) &&
            (velocity.interior[index] = zero(T))
    end
    @_parallel for index in eachindex(velocity.boundary)
        @inbounds iszero(geometry.boundary_aperture[index]) &&
            (velocity.boundary[index] = zero(T))
    end
    return velocity
end

"""Finite-volume divergence over the open portion of every cut cell."""
function divergence!(
        destination::CellField{T}, velocity::FaceVelocity,
        geometry::GeometryField{T},
    ) where {T}
    destination.mesh === velocity.mesh === geometry.mesh || throw(
        ArgumentError("divergence arguments must share a mesh"),
    )
    _zero_interior!(destination)
    mesh = destination.mesh
    for (index, face) in pairs(mesh.faces)
        @inbounds flux = velocity.interior[index] * face.area *
            geometry.aperture[index]
        minus_open = _get(geometry.open_fraction, face.minus)
        plus_open = _get(geometry.open_fraction, face.plus)
        minus_open > zero(T) && _add!(
            destination,
            flux / (mesh.blocks[face.minus.block].volume * minus_open),
            face.minus,
        )
        plus_open > zero(T) && _add!(
            destination,
            -flux / (mesh.blocks[face.plus.block].volume * plus_open),
            face.plus,
        )
    end
    for (index, face) in pairs(mesh.boundary_faces)
        open = _get(geometry.open_fraction, face.cell)
        open > zero(T) || continue
        @inbounds flux = face.side * velocity.boundary[index] * face.area *
            geometry.boundary_aperture[index]
        _add!(
            destination,
            flux / (mesh.blocks[face.cell.block].volume * open), face.cell,
        )
    end
    return destination
end

"""Advect an intensive cell scalar through the open cut-cell faces."""
function advect!(
        scalar::CellField{T}, velocity::FaceVelocity, dt,
        workspace::AdvectionWorkspace{T}, geometry::GeometryField{T};
        boundary::AbstractBoundaryCondition = ZeroGradient(), time = 0,
    ) where {T}
    scalar.mesh === velocity.mesh === workspace.increment.mesh ===
        geometry.mesh || throw(
        ArgumentError("cut-cell advection arguments must share a mesh"),
    )
    mesh = scalar.mesh
    @_parallel for index in eachindex(mesh.faces)
        face = mesh.faces[index]
        @inbounds speed = velocity.interior[index]
        state = speed >= 0 ? _get(scalar, face.minus) :
            _get(scalar, face.plus)
        @inbounds workspace.interior_flux[index] =
            dt * speed * state * face.area * geometry.aperture[index]
    end
    @_parallel for index in eachindex(mesh.boundary_faces)
        face = mesh.boundary_faces[index]
        @inbounds coordinate_speed = velocity.boundary[index]
        outward_speed = face.side * coordinate_speed
        state = outward_speed >= 0 ? _get(scalar, face.cell) :
            _inflow_value(boundary, scalar, face, time)
        @inbounds workspace.boundary_flux[index] =
            dt * outward_speed * state * face.area *
            geometry.boundary_aperture[index]
    end
    nx, ny = mesh.block_size
    h = mesh.halo
    cells_per_block = nx * ny
    _parallel_block_columns(mesh) do block, first_j, last_j
        increment = @inbounds workspace.increment.blocks[block]
        values = @inbounds scalar.blocks[block]
        open = @inbounds geometry.open_fraction.blocks[block]
        offset = (block - 1) * cells_per_block
        cell_volume = mesh.blocks[block].volume
        @inbounds for j in first_j:last_j, i in 1:nx
            cell = offset + (j - 1) * nx + i
            integrated = zero(T)
            for pointer in workspace.cell_offsets[cell]:(
                    workspace.cell_offsets[cell + 1] - 1
                )
                face_index = workspace.face_indices[pointer]
                flux = workspace.boundary_entries[pointer] ?
                    workspace.boundary_flux[face_index] :
                    workspace.interior_flux[face_index]
                integrated += workspace.face_signs[pointer] * flux
            end
            fraction = open[i + h, j + h]
            change = fraction > zero(T) ?
                integrated / (cell_volume * fraction) : zero(T)
            increment[i + h, j + h] = change
            values[i + h, j + h] += change
        end
    end
    return scalar
end

function _cut_cell_dual_schedule(
        geometry::GeometryField{T}, ::Val{Axis},
        build_three_point_stencils::Bool,
    ) where {T, Axis}
    base = _amr_dual_schedule(
        geometry.mesh, Val(Axis), T; build_three_point_stencils,
    )
    weights = copy(base.flux_weights)
    @_parallel for index in eachindex(weights)
        @inbounds weights[index] *=
            geometry.aperture[base.flux_indices[index]]
    end
    volumes = copy(geometry.dual_volumes[Axis])
    # Completely covered Cartesian faces are retained in the background-grid
    # storage.  Their flux graph is identically zero; a unit algebraic volume
    # keeps this harmless extension finite without introducing a fraction
    # floor into any physical cut control volume.
    for index in eachindex(volumes)
        @inbounds volumes[index] > zero(T) || (volumes[index] = one(T))
    end
    return AMRDualAdvectionSchedule(
        base.faces, base.flux_indices, weights, volumes,
    )
end

function _geometry_phase_open_volumes(
        geometry::GeometryField{T}, ::Val{Axis},
    ) where {T, Axis}
    interior_count = length(geometry.dual_volumes[Axis])
    boundary = geometry.boundary_dual_volumes[Axis]
    volumes = zeros(T, interior_count + length(boundary))
    copyto!(volumes, 1, geometry.dual_volumes[Axis], 1, interior_count)
    copyto!(volumes, interior_count + 1, boundary, 1, length(boundary))
    return volumes
end

function _geometry_phase_schedule(
        geometry::GeometryField{T}, ::Val{Axis},
    ) where {T, Axis}
    mesh = geometry.mesh
    base = _bounded_phase_schedule(mesh, Val(Axis), T)
    volumes = _geometry_phase_open_volumes(geometry, Val(Axis))
    length(volumes) == length(base.volumes) || throw(
        DimensionMismatch("geometry and phase face schedules have different sizes")
    )
    for index in eachindex(volumes)
        @inbounds volumes[index] > zero(T) || (volumes[index] = one(T))
    end
    return AMRDualAdvectionSchedule(
        base.faces, base.flux_indices, base.flux_weights, volumes,
    )
end

function StaggeredAdvectionWorkspace(
        field::FaceField{Axis, T, M}, geometry::GeometryField{T, M},
    ) where {Axis, T, M}
    schedule = _cut_cell_dual_schedule(geometry, Val(Axis), true)
    return StaggeredAdvectionWorkspace(
        FaceField{Axis}(field.mesh, T), FaceField{Axis}(field.mesh, T),
        schedule, Ref(true),
    )
end

function StaggeredVelocityAdvectionWorkspace(
        velocity::FaceVelocity{T, M}, geometry::GeometryField{T, M},
    ) where {T, M}
    x = FaceField{1}(velocity)
    y = FaceField{2}(velocity)
    return StaggeredVelocityAdvectionWorkspace(
        x, y,
        StaggeredAdvectionWorkspace(x, geometry),
        StaggeredAdvectionWorkspace(y, geometry), Ref(true),
    )
end

StaggeredAdvectionWorkspace(
    velocity::FaceVelocity, geometry::GeometryField,
) = StaggeredVelocityAdvectionWorkspace(velocity, geometry)

function _component_volumes_by_global_face(
        mesh::BlockTreeMesh, component_volumes, ::Type{T};
        covered_value = one(T),
    ) where {T}
    result = Vector{T}(undef, length(mesh.faces))
    for axis in 1:2
        global_to_local, _ = _component_face_indices(mesh, axis)
        local_volumes = component_volumes[axis]
        for global_index in eachindex(mesh.faces)
            mesh.faces[global_index].axis == axis || continue
            local_index = global_to_local[global_index]
            value = local_volumes[local_index]
            @inbounds result[global_index] = value > zero(T) ?
                value : covered_value
        end
    end
    return result
end

function _geometry_face_dual_volumes(geometry::GeometryField{T}) where {T}
    return _component_volumes_by_global_face(
        geometry.mesh, geometry.dual_volumes, T,
    )
end

function _uncut_face_dual_volumes(mesh::BlockTreeMesh, ::Type{T}) where {T}
    volumes = ntuple(2) do axis
        _amr_dual_schedule(
            mesh, Val(axis), T; build_three_point_stencils = false,
        ).volumes
    end
    return _component_volumes_by_global_face(mesh, volumes, T)
end

function _scaled_inverse_metric(metric::DiagonalInverseFaceMetric{T}, scale) where {T}
    return DiagonalInverseFaceMetric(metric.diagonal .* scale .^ 2)
end

function _scaled_inverse_metric(metric::SparseInverseFaceMetric{T}, scale) where {T}
    matrix = copy(metric.matrix)
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    for column in axes(matrix, 2), pointer in nzrange(matrix, column)
        @inbounds values[pointer] *= scale[rows[pointer]] * scale[column]
    end
    return SparseInverseFaceMetric(matrix)
end

"""Compatible pressure-gradient Riesz map for fixed cut-cell geometry."""
function GradientOperator(
        geometry::GeometryField{T, M};
        construction_cache = GradientConstructionCache(T),
    ) where {T, M}
    mesh = geometry.mesh
    cut_volumes = _geometry_face_dual_volumes(geometry)
    uncut_volumes = _uncut_face_dual_volumes(mesh, T)
    scale = sqrt.(uncut_volumes ./ cut_volumes)
    base = GradientOperator(mesh, T; construction_cache)
    metric = _scaled_inverse_metric(base.inverse_metric, scale)
    return GradientOperator{T, M, typeof(metric), typeof(construction_cache)}(
        mesh, metric, zeros(T, length(mesh.faces)),
        construction_cache,
    )
end

function _geometry_cell_volumes(geometry::GeometryField{T}) where {T}
    mesh = geometry.mesh
    nx, ny = mesh.block_size
    result = Vector{T}(undef, ncells(mesh))
    for (block, leaf) in pairs(mesh.blocks)
        values = interior(geometry.open_fraction, block)
        offset = (block - 1) * nx * ny
        for j in 1:ny, i in 1:nx
            @inbounds result[offset + (j - 1) * nx + i] =
                values[i, j] * leaf.volume
        end
    end
    return result
end

"""Construct `-div(open_aperture*grad(p))` with cut-cell adjoint metrics."""
function PoissonOperator(
        geometry::GeometryField{T}; face_coefficient = 1,
        construction_cache = GradientConstructionCache(T),
        assemble_matrix = true,
    ) where {T}
    material = _face_coefficients(geometry.mesh, face_coefficient, T)
    coefficients = material .* geometry.aperture .^ 2
    return EllipticOperator(
        geometry.mesh; reaction = 0, diffusion = 1,
        face_coefficient = coefficients,
        gradient_operator = GradientOperator(
            geometry; construction_cache,
        ),
        volumes = _geometry_cell_volumes(geometry),
        active_cells = geometry.active_cells,
        construction_cache, assemble_matrix, T,
    )
end


function _cut_cell_weighted_gradient!(
        destination::FaceVelocity{T}, pressure::CellField{T},
        operator::EllipticOperator{T}, workspace::EllipticWorkspace{T},
        geometry::GeometryField{T},
    ) where {T}
    _pressure_jump!(operator.gradient_operator.pressure_jump, pressure)
    jumps = operator.gradient_operator.pressure_jump
    square_root = operator.square_root_face_coefficient
    @_parallel for index in eachindex(jumps)
        @inbounds workspace.face_input[index] = square_root[index] * jumps[index]
    end
    _mul_inverse_metric!(
        workspace.face_output,
        operator.gradient_operator.inverse_metric,
        workspace.face_input,
    )
    @_parallel for index in eachindex(destination.interior)
        aperture = @inbounds geometry.aperture[index]
        # The integrated operator is B*sqrt(k)*K*sqrt(k)*B', with
        # sqrt(k)=aperture*sqrt(material). Dividing the output factor by the
        # aperture yields the face acceleration whose aperture-weighted
        # divergence is exactly that operator, also when K is non-diagonal at
        # an AMR interface.
        @inbounds destination.interior[index] = aperture > zero(T) ?
            square_root[index] / aperture *
            workspace.face_output[index] : zero(T)
    end
    fill!(destination.boundary, zero(T))
    return destination
end

"""Pressure projection using cut-cell divergence and its adjoint gradient."""
function project!(
        velocity::FaceVelocity{T}, pressure::CellField{T},
        solver::EllipticSolver{T}, dt,
        workspace::ProjectionWorkspace{T}, geometry::GeometryField{T};
        velocity_boundary = nothing, boundary_time = zero(T),
        tolerance = sqrt(eps(T)), max_cycles = 100,
        compatibility = :project
    ) where {T}
    velocity.mesh === pressure.mesh === solver.operator.mesh ===
        workspace.divergence.mesh === geometry.mesh || throw(
        ArgumentError("cut-cell projection arguments must share a mesh"),
    )
    iszero(solver.operator.reaction) || throw(
        ArgumentError("pressure projection requires a zero-reaction operator"),
    )
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    isnothing(velocity_boundary) ? enforce_no_permeation!(velocity) :
        enforce_velocity_boundary!(
            velocity, velocity_boundary;
            time = boundary_time, geometry,
        )
    enforce_geometry_no_slip!(velocity, geometry)
    divergence!(workspace.divergence, velocity, geometry)
    nx, ny = velocity.mesh.block_size
    halo = velocity.mesh.halo
    _parallel_block_columns(velocity.mesh) do block, first_j, last_j
        values = @inbounds workspace.divergence.blocks[block]
        @inbounds for j in first_j:last_j, i in 1:nx
            values[i + halo, j + halo] = -values[i + halo, j + halo] / dt
        end
    end
    stats = solve!(
        pressure, solver, workspace.divergence;
        rtol = zero(T), atol = dimensionless_tolerance / T(dt)^2,
        max_cycles, compatibility,
    )
    stats.residual_scale = inv(T(dt)^2)
    _cut_cell_weighted_gradient!(
        workspace.weighted_gradient, pressure,
        solver.operator, workspace.elliptic, geometry,
    )
    scale = T(dt) * solver.operator.diffusion
    for index in eachindex(velocity.interior)
        @inbounds velocity.interior[index] -=
            scale * workspace.weighted_gradient.interior[index]
    end
    for index in eachindex(velocity.boundary)
        @inbounds velocity.boundary[index] -=
            scale * workspace.weighted_gradient.boundary[index]
    end
    isnothing(velocity_boundary) ? enforce_no_permeation!(velocity) :
        enforce_velocity_boundary!(
            velocity, velocity_boundary;
            time = boundary_time, geometry,
        )
    enforce_geometry_no_slip!(velocity, geometry)
    return stats
end

function _local_plic_segment(field::InterfaceField{T}, ref::CellRef) where {T}
    fraction = _conditional_fraction(field, ref)
    zero(T) < fraction < one(T) || return nothing
    nx = _get(field.normal_x, ref)
    ny = _get(field.normal_y, ref)
    intercept = _get(field.intercept, ref)
    polygon = isnothing(field.geometry) ? NTuple{2, T}[
            (-T(0.5), -T(0.5)), (T(0.5), -T(0.5)),
            (T(0.5), T(0.5)), (-T(0.5), T(0.5)),
        ] : _open_cell_polygon(field.geometry, ref)
    length(polygon) >= 3 || return nothing
    tolerance = T(64) * eps(T)
    points = NTuple{2, T}[]
    previous = last(polygon)
    previous_value = nx * previous[1] + ny * previous[2] - intercept
    for current in polygon
        current_value = nx * current[1] + ny * current[2] - intercept
        abs(previous_value) <= tolerance && push!(points, previous)
        if (previous_value < -tolerance && current_value > tolerance) ||
                (previous_value > tolerance && current_value < -tolerance)
            push!(
                points,
                _polygon_crossing(
                    previous, current, previous_value, current_value,
                ),
            )
        end
        previous = current
        previous_value = current_value
    end
    unique_points = NTuple{2, T}[]
    for point in points
        any(
            other -> hypot(point[1] - other[1], point[2] - other[2]) <=
                tolerance, unique_points,
        ) || push!(unique_points, point)
    end
    length(unique_points) >= 2 || return nothing
    best = (unique_points[1], unique_points[2])
    best_distance = -one(T)
    for first in 1:(length(unique_points) - 1),
            second in (first + 1):length(unique_points)
        distance = (unique_points[first][1] - unique_points[second][1])^2 +
            (unique_points[first][2] - unique_points[second][2])^2
        if distance > best_distance
            best = (unique_points[first], unique_points[second])
            best_distance = distance
        end
    end
    return best_distance > tolerance^2 ? best : nothing
end

function _embedded_wall_connections(
        geometry::GeometryField{T}, ::Val{Axis},
        incidences, global_to_local,
    ) where {T, Axis}
    mesh = geometry.mesh
    nx, ny = mesh.block_size
    connections = WallDiffusionConnection{T}[]
    for cell_index in eachindex(incidences)
        block = (cell_index - 1) ÷ (nx * ny) + 1
        cell_local = (cell_index - 1) % (nx * ny)
        ref = CellRef(block, cell_local % nx + 1, cell_local ÷ nx + 1)
        segment = _local_plic_segment(geometry.solid, ref)
        isnothing(segment) && continue
        leaf = mesh.blocks[block]
        center = cellcenter(mesh, ref.block, ref.i, ref.j)
        physical(point) = (
            T(center[1] + point[1] * leaf.spacing[1]),
            T(center[2] + point[2] * leaf.spacing[2]),
        )
        p0, p1 = physical(segment[1]), physical(segment[2])
        midpoint = ((p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2)
        length = hypot(p1[1] - p0[1], p1[2] - p0[2])
        length > zero(T) || continue
        for incidence in incidences[cell_index]
            face = mesh.faces[incidence.global_index]
            distance = hypot(
                _wrapped_delta(
                    face.center[1] - midpoint[1], mesh.extent[1],
                    mesh.periodic[1],
                ),
                _wrapped_delta(
                    face.center[2] - midpoint[2], mesh.extent[2],
                    mesh.periodic[2],
                ),
            )
            distance = max(
                T(distance), T(0.25) * min(leaf.spacing...),
            )
            push!(
                connections,
                WallDiffusionConnection(
                    global_to_local[incidence.global_index],
                    T(0.5) * incidence.fraction * length / distance,
                    midpoint, true,
                ),
            )
        end
    end
    # Fully covered background-grid face unknowns do not represent a physical
    # velocity.  Keep this algebraic extension constrained for both no-slip
    # and free-slip walls.  `tangential = false` distinguishes these rows from
    # the viscous tangential-wall rows above, which FreeSlip deliberately
    # omits.
    for (global_index, face) in pairs(mesh.faces)
        face.axis == Axis || continue
        local_index = global_to_local[global_index]
        geometry.dual_volumes[Axis][local_index] > zero(T) && continue
        push!(
            connections,
            WallDiffusionConnection(
                local_index, one(T), T.(face.center), false,
            ),
        )
    end
    return connections
end

function _embedded_variable_wall_stresses(
        geometry::GeometryField{T}; include_cut_wall = true,
    ) where {T}
    mesh = geometry.mesh
    nx, ny = mesh.block_size
    stresses = Tuple{Int, T, Int}[]
    if include_cut_wall
        for axis in 1:2
            incidences, _, _ = _component_incidences(mesh, Val(axis), T)
            for cell_index in eachindex(incidences)
                block = (cell_index - 1) ÷ (nx * ny) + 1
                cell_local = (cell_index - 1) % (nx * ny)
                ref = CellRef(
                    block, cell_local % nx + 1, cell_local ÷ nx + 1,
                )
                segment = _local_plic_segment(geometry.solid, ref)
                isnothing(segment) && continue
                leaf = mesh.blocks[block]
                center = cellcenter(mesh, ref.block, ref.i, ref.j)
                physical(point) = (
                    T(center[1] + point[1] * leaf.spacing[1]),
                    T(center[2] + point[2] * leaf.spacing[2]),
                )
                p0, p1 = physical(segment[1]), physical(segment[2])
                midpoint = (
                    (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2,
                )
                length = hypot(p1[1] - p0[1], p1[2] - p0[2])
                for incidence in incidences[cell_index]
                    face = mesh.faces[incidence.global_index]
                    distance = hypot(
                        _wrapped_delta(
                            face.center[1] - midpoint[1], mesh.extent[1],
                            mesh.periodic[1],
                        ),
                        _wrapped_delta(
                            face.center[2] - midpoint[2], mesh.extent[2],
                            mesh.periodic[2],
                        ),
                    )
                    distance = max(
                        T(distance), T(0.25) * min(leaf.spacing...),
                    )
                    push!(
                        stresses,
                        (
                            incidence.global_index,
                            T(0.5) * incidence.fraction * length / distance,
                            cell_index,
                        ),
                    )
                end
            end
        end
    end
    # Background-grid degrees of freedom whose complete staggered control
    # volume is covered have no physical stress row.  Constrain this harmless
    # algebraic extension to the stationary solid velocity.
    for axis in 1:2
        global_to_local, _ = _component_face_indices(mesh, axis)
        for (global_index, face) in pairs(mesh.faces)
            face.axis == axis || continue
            local_index = global_to_local[global_index]
            geometry.dual_volumes[axis][local_index] > zero(T) && continue
            cell_index = _cell_index(face.minus, nx, ny)
            push!(stresses, (global_index, one(T), cell_index))
        end
    end
    return stresses
end

function _cut_cell_diffusion_component(
        geometry::GeometryField{T}, ::Val{Axis}, boundary,
    ) where {T, Axis}
    mesh = geometry.mesh
    field = FaceField{Axis}(mesh, T)
    advection_workspace = StaggeredAdvectionWorkspace(field, geometry)
    schedule = advection_workspace.schedule
    incidences, global_to_local, _ =
        _component_incidences(mesh, Val(Axis), T)
    _, local_to_global = _component_local_to_global(mesh, Axis)
    volumes = copy(schedule.volumes)
    connections = _dual_diffusion_connections(
        mesh, schedule, local_to_global,
    )
    wall_connections = _wall_diffusion_connections(
        mesh, Val(Axis), T, incidences, global_to_local,
    )
    append!(
        wall_connections,
        _embedded_wall_connections(
            geometry, Val(Axis), incidences, global_to_local,
        ),
    )
    stiffness = _diffusion_stiffness(
        mesh, local_to_global, connections,
        wall_connections, boundary, T,
    )
    laplacian = sparse(-spdiagm(0 => inv.(volumes)) * stiffness)
    return StaggeredDiffusionComponent{Axis, T}(
        stiffness, laplacian, volumes, local_to_global, wall_connections,
        zeros(T, length(local_to_global)), zeros(T, length(local_to_global)),
        _maximum_absolute_row_sum(laplacian),
    )
end

"""Symmetric positive cut-cell diffusion with embedded no- or free-slip walls."""
function StaggeredDiffusionOperator(
        geometry::GeometryField{T, M}; viscosity = 1,
        boundary = NoSlip(),
    ) where {T, M}
    viscosity_value = T(viscosity)
    isfinite(viscosity_value) && viscosity_value > zero(T) || throw(
        ArgumentError("viscosity must be finite and strictly positive"),
    )
    boundary isa Union{NoSlip, FreeSlip, VelocityBoundaryConditions} || throw(
        ArgumentError(
            "embedded geometry requires NoSlip(), FreeSlip(), or " *
                "VelocityBoundaryConditions"
        ),
    )
    x = _cut_cell_diffusion_component(geometry, Val(1), boundary)
    y = _cut_cell_diffusion_component(geometry, Val(2), boundary)
    return StaggeredDiffusionOperator{
        T, M, typeof(x), typeof(y), typeof(boundary),
    }(geometry.mesh, x, y, viscosity_value, boundary)
end
