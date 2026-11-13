"""Set every face whose open aperture vanishes to the stationary-wall value."""

function enforce_geometry_no_slip!(
        velocity::FaceVelocity{T}, geometry::GeometryField{T}; preserve_absent = false,
    ) where {T}
    velocity.mesh === geometry.mesh || throw(
        ArgumentError("velocity and geometry must share a mesh"),
    )
    @_parallel for index in eachindex(velocity.interior)
        preserve_absent && isnan(velocity.interior[index]) && continue
        @inbounds iszero(geometry.aperture[index]) &&
            (velocity.interior[index] = zero(T))
    end
    @_parallel for index in eachindex(velocity.boundary)
        preserve_absent && isnan(velocity.boundary[index]) && continue
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
    # Serial: a `Dirichlet` inflow value is a user callback.
    for index in eachindex(mesh.boundary_faces)
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
        schedule, Ref(true);
        apertures = (geometry.aperture, geometry.boundary_aperture),
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

function _geometry_face_dual_volumes(geometry::GeometryField{T}) where {T}
    # Algebraic operators retain a unit extension on covered faces, while
    # diagnostics use the physical zero-volume cache itself.
    result = copy(geometry.face_dual_volumes)
    for index in eachindex(result)
        @inbounds result[index] > zero(T) || (result[index] = one(T))
    end
    return result
end

function _uncut_face_dual_volumes(mesh::BlockTreeMesh, ::Type{T}) where {T}
    # The uncut measure depends only on the face topology, so the mesh owns
    # it. In particular, a diagnostic must not build a dual *advection*
    # schedule merely to obtain these volumes.
    return T === eltype(mesh.face_dual_volumes) ?
        mesh.face_dual_volumes : T.(mesh.face_dual_volumes)
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
        velocity_boundary = nothing, boundary_face_coefficient = nothing,
        construction_cache = GradientConstructionCache(T),
        assemble_matrix = true,
    ) where {T}
    material = _face_coefficients(geometry.mesh, face_coefficient, T)
    coefficients = material .* geometry.aperture .^ 2
    boundary_material = isnothing(boundary_face_coefficient) ?
        _boundary_material(
            geometry.mesh, face_coefficient,
            _pressure_boundary_sides(velocity_boundary), T,
        ) : boundary_face_coefficient
    return EllipticOperator(
        geometry.mesh; reaction = 0, diffusion = 1,
        face_coefficient = coefficients,
        velocity_boundary, boundary_face_coefficient = boundary_material,
        boundary_aperture = geometry.boundary_aperture,
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
    _elliptic_boundary_gradient!(destination, pressure, operator)
    return destination
end

"""Incremental cut-cell projection.

The predictor must already include the old pressure's weighted gradient. This
method adds the solved increment to `pressure`; use zero scratch pressure for
an absolute projection of an unpredicted velocity.
"""
function project!(
        velocity::FaceVelocity{T}, pressure::CellField{T},
        solver::EllipticSolver{T}, dt,
        workspace::ProjectionWorkspace{T}, geometry::GeometryField{T};
        velocity_boundary = nothing, boundary_time = zero(T),
        tolerance = sqrt(eps(T)), max_cycles = 100,
        compatibility = :project, _inherited_divergence = nothing,
        _reference_dt = dt,
    ) where {T}
    _validate_projection_compatibility(compatibility)
    velocity.mesh === pressure.mesh === solver.operator.mesh ===
        workspace.divergence.mesh === geometry.mesh || throw(
        ArgumentError("cut-cell projection arguments must share a mesh"),
    )
    iszero(solver.operator.reaction) || throw(
        ArgumentError("pressure projection requires a zero-reaction operator"),
    )
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))
    _validate_projection_reference_dt(dt, _reference_dt)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    _validate_projection_boundary(solver.operator, velocity_boundary)
    isnothing(velocity_boundary) ? enforce_no_permeation!(velocity) :
        enforce_velocity_boundary!(
            velocity, velocity_boundary;
            time = boundary_time, geometry, preserve_open = true,
        )
    enforce_geometry_no_slip!(velocity, geometry)
    return _project_prepared!(
        velocity, pressure, solver, dt, workspace, geometry;
        velocity_boundary, boundary_time, tolerance = dimensionless_tolerance,
        max_cycles, compatibility, _inherited_divergence, _reference_dt,
    )
end

"""Cut-cell projection with physical boundary values prepared before the predictor."""
function _project_prepared!(
        velocity::FaceVelocity{T}, pressure::CellField{T},
        solver::EllipticSolver{T}, dt,
        workspace::ProjectionWorkspace{T}, geometry::GeometryField{T};
        velocity_boundary = nothing, boundary_time = zero(T),
        tolerance = sqrt(eps(T)), max_cycles = 100,
        compatibility = :project, _inherited_divergence = nothing,
        _reference_dt = dt,
    ) where {T}
    _validate_projection_boundary(solver.operator, velocity_boundary)
    dimensionless_tolerance = T(tolerance)
    divergence!(workspace.divergence, velocity, geometry)
    _projection_rhs!(
        workspace.divergence, T(dt), _inherited_divergence, T(_reference_dt),
    )
    _zero_interior!(workspace.increment)
    stats = solve!(
        workspace.increment, solver, workspace.divergence;
        rtol = zero(T), atol = dimensionless_tolerance / T(dt)^2,
        max_cycles, compatibility,
    )
    stats.residual_scale = inv(T(dt)^2)
    _projection_compatibility!(stats, workspace, dt, dimensionless_tolerance)
    _cut_cell_weighted_gradient!(
        workspace.weighted_gradient, workspace.increment,
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
    _accumulate_pressure_increment!(pressure, workspace.increment, solver)
    # Fixed faces have zero pressure gradient; open faces retain the
    # correction rather than being re-extrapolated from the interior.
    enforce_geometry_no_slip!(velocity, geometry)
    return stats
end

@inline function _embedded_pure_wall_side(minus::T, plus::T) where {T}
    tolerance = T(64) * eps(T)
    if abs(minus - one(T)) <= tolerance && abs(plus) <= tolerance
        return 1
    elseif abs(plus - one(T)) <= tolerance && abs(minus) <= tolerance
        return -1
    end
    return 0
end

@inline function _embedded_cartesian_wall_axis(segment, ::Type{T}) where {T}
    tolerance = T(64) * eps(T)
    return abs(segment[1][1] - segment[2][1]) <= tolerance ? 1 :
        abs(segment[1][2] - segment[2][2]) <= tolerance ? 2 : 0
end

# Pure-fluid and covered cells have no PLIC segment, but their shared face is
# still a wall. Join adjacent fine faces on a fluid cell's side to recover the
# segment geometry of a parallel PLIC plane approaching that grid face.
function _embedded_aligned_wall_segments(geometry::GeometryField{T}) where {T}
    mesh = geometry.mesh
    nx, ny = mesh.block_size
    lattice = _finest_lattice_size(mesh)
    sides = Dict{NTuple{3, Int}, Vector{Tuple{T, T}}}()
    for face in mesh.faces
        minus = _get(geometry.open_fraction, face.minus)
        plus = _get(geometry.open_fraction, face.plus)
        side = _embedded_pure_wall_side(minus, plus)
        iszero(side) && continue
        ref = side > 0 ? face.minus : face.plus
        # Use integer topology so adjacent fine spans meet exactly even when
        # their physical coordinates are not representable in floating point.
        _, face_lo, face_hi = _face_lattice_geometry(mesh, face, lattice)
        cell_lo, scale = _cell_lattice_bounds(mesh, mesh.blocks, ref, lattice)
        tangent = 3 - face.axis
        lo = T(face_lo - cell_lo[tangent]) / T(scale[tangent]) - T(0.5)
        hi = T(face_hi - cell_lo[tangent]) / T(scale[tangent]) - T(0.5)
        key = (_cell_index(ref, nx, ny), face.axis, side)
        spans = get!(sides, key) do
            Tuple{T, T}[]
        end
        push!(spans, (lo, hi))
    end
    segments = Dict{Int, Vector{Tuple{NTuple{2, T}, NTuple{2, T}}}}()
    for key in sort!(collect(keys(sides)))
        cell_index, axis, side = key
        spans = sort!(sides[key])
        lo, hi = first(spans)
        for index in 2:length(spans)
            next_lo, next_hi = spans[index]
            if next_lo <= hi
                hi = max(hi, next_hi)
            else
                _push_embedded_aligned_segment!(
                    segments, cell_index, axis, side, lo, hi,
                )
                lo, hi = next_lo, next_hi
            end
        end
        _push_embedded_aligned_segment!(segments, cell_index, axis, side, lo, hi)
    end
    return segments
end

function _push_embedded_aligned_segment!(
        segments::Dict{Int, Vector{Tuple{NTuple{2, T}, NTuple{2, T}}}},
        cell_index, axis, side, lo, hi,
    ) where {T}
    coordinate = T(side) / 2
    first_point = axis == 1 ? (coordinate, lo) : (lo, coordinate)
    last_point = axis == 1 ? (coordinate, hi) : (hi, coordinate)
    cell_segments = get!(segments, cell_index) do
        Tuple{NTuple{2, T}, NTuple{2, T}}[]
    end
    push!(cell_segments, (first_point, last_point))
    return nothing
end

# Conductance of the part of a Cartesian embedded segment owned by one
# tangential staggered face. Shared by scalar and coupled-stress diffusion.
function _embedded_cartesian_tangent_conductance(
        geometry::GeometryField{T}, ::Val{Axis}, ref, incidence, segment, midpoint,
    ) where {T, Axis}
    mesh = geometry.mesh
    geometry.dual_volumes[Axis][incidence.local_index] > zero(T) || return zero(T)
    face = mesh.faces[incidence.global_index]
    wall_axis = 3 - Axis
    strip_lo, strip_hi = _face_local_tangent_interval(mesh, ref, face)
    wall_coordinate = segment[1][wall_axis]
    strip_lo - 256eps(T) <= wall_coordinate <= strip_hi + 256eps(T) ||
        return zero(T)
    lo, hi = minmax(segment[1][Axis], segment[2][Axis])
    if incidence.side < 0
        hi = min(hi, zero(T))
    else
        lo = max(lo, zero(T))
    end
    spacing = mesh.blocks[ref.block].spacing
    measure = max(zero(T), hi - lo) * T(spacing[Axis])
    distance = abs(
        _wrapped_delta(
            face.center[wall_axis] - midpoint[wall_axis],
            mesh.extent[wall_axis], mesh.periodic[wall_axis],
        ),
    )
    distance = max(T(distance), T(0.25) * min(spacing...))
    return measure / distance
end

# Signed distance from `point` to the PLIC plane of cell `ref`, measured along
# the physical normal. The plane is stored in normalized cell coordinates, so
# the metric divides by the anisotropic spacing and then by the norm.
function _embedded_plic_distance(
        geometry::GeometryField{T}, ref, point,
    ) where {T}
    mesh = geometry.mesh
    spacing = mesh.blocks[ref.block].spacing
    center = cellcenter(mesh, ref.block, ref.i, ref.j)
    normal_x = _get(geometry.solid.normal_x, ref)
    normal_y = _get(geometry.solid.normal_y, ref)
    dx = _wrapped_delta(
        point[1] - center[1], mesh.extent[1], mesh.periodic[1],
    ) / T(spacing[1])
    dy = _wrapped_delta(
        point[2] - center[2], mesh.extent[2], mesh.periodic[2],
    ) / T(spacing[2])
    return T(
        (normal_x * dx + normal_y * dy - _get(geometry.solid.intercept, ref)) /
            hypot(normal_x / T(spacing[1]), normal_y / T(spacing[2])),
    )
end

# Length of the part of a cut-cell PLIC segment inside the tangential strip
# [tangent_lo, tangent_hi] and the component-direction half-strip of one dual
# node. A convex strip cut by the PLIC line has exactly that clipped piece as
# the wall boundary of its open part.
function _embedded_cut_wall_measure(
        geometry::GeometryField{T}, ::Val{Axis}, ref,
        tangent_lo, tangent_hi, side, segment,
    ) where {T, Axis}
    mesh = geometry.mesh
    half = T(0.5)
    axis_lo, axis_hi = side < 0 ? (-half, zero(T)) : (zero(T), half)
    lower = Axis == 1 ? (axis_lo, T(tangent_lo)) : (T(tangent_lo), axis_lo)
    upper = Axis == 1 ? (axis_hi, T(tangent_hi)) : (T(tangent_hi), axis_hi)
    first, last = segment
    delta = (last[1] - first[1], last[2] - first[2])
    parameter_lo, parameter_hi = zero(T), one(T)
    for dimension in 1:2
        if iszero(delta[dimension])
            # The component-direction partition is half open: a segment that
            # lies exactly on the cell mid-plane belongs to the positive side
            # only, so the pieces of one cell always sum to its segment.
            closed = dimension != Axis || side > 0
            inside = if closed
                lower[dimension] <= first[dimension] <= upper[dimension]
            else
                lower[dimension] <= first[dimension] < upper[dimension]
            end
            inside || return zero(T)
        else
            lo = (lower[dimension] - first[dimension]) / delta[dimension]
            hi = (upper[dimension] - first[dimension]) / delta[dimension]
            lo, hi = minmax(lo, hi)
            parameter_lo = max(parameter_lo, lo)
            parameter_hi = min(parameter_hi, hi)
            parameter_hi > parameter_lo || return zero(T)
        end
    end
    spacing = mesh.blocks[ref.block].spacing
    portion = parameter_hi - parameter_lo
    return max(
        zero(T), hypot(
            portion * delta[1] * T(spacing[1]),
            portion * delta[2] * T(spacing[2]),
        ),
    )
end

# No-slip closure of a cut-cell PLIC segment. The segment is split among the
# dual nodes by the tangential strip of each component face and by the
# half-open half-strip on that face's side of the cell mid-plane, so every
# piece is owned once, also by an unpaired face at a non-periodic boundary.
# A fluid-side node $f$ satisfies $u_w = u_f - d_f\,\partial_n u$ with
# $u_w = 0$, so its wall flux is $(A_f/d_f)u_f$: that flux is exact for linear
# data. The conductance is positive for every node, so a node whose centre
# lies inside the solid gets the same magnitude with the sign of $\abs{d_f}$,
# i.e. the opposite row contribution to the one its open dual volume needs;
# its rows are therefore first order. They stay diffusion unknowns because
# the cut-cell projection updates the same faces.
function _push_embedded_cut_wall!(
        connections, geometry::GeometryField{T}, ::Val{Axis}, ref, cell_faces,
        segment, midpoint, global_to_local,
    ) where {T, Axis}
    mesh = geometry.mesh
    spacing = mesh.blocks[ref.block].spacing
    # A dual node on the reconstructed plane has the exact no-slip value and
    # its conductance would diverge. The floor is one sixty-fourth of the
    # smallest cell width, and it is active: a cut plane passes through or
    # next to the centre of many wall-piece nodes, so a fifth to a third of
    # the pieces in the manufactured fixtures sit closer than h/64. It bounds
    # the largest wall conductance at 64*measure/min(h), which keeps the
    # assembled stiffness scaling as 1/h^2; the pinned errors and orders of
    # the convergence tests are measured with this floor.
    floor = min(spacing...) / T(64)
    minus_present, plus_present = false, false
    # A segment that lies exactly on the cell mid-plane satisfies the
    # half-open test on one side only, which must be the side whose dual node
    # is in the fluid; the positive side keeps it when the plane passes
    # through both face centres.
    midplane_owner = zero(Int8)
    if iszero(segment[1][Axis]) && iszero(segment[2][Axis])
        distances = ntuple(2) do position
            side = position == 1 ? -1 : 1
            incidence = findfirst(face -> face.side == side, cell_faces)
            isnothing(incidence) && return zero(T)
            face = mesh.faces[cell_faces[incidence].global_index]
            return _embedded_plic_distance(geometry, ref, face.center)
        end
        minus_distance, plus_distance = distances
        if minus_distance > zero(T) && plus_distance <= zero(T)
            midplane_owner = Int8(-1)
        elseif plus_distance > zero(T)
            midplane_owner = Int8(1)
        end
    end
    for incidence in cell_faces
        face = mesh.faces[incidence.global_index]
        incidence.side < 0 ? (minus_present = true) : (plus_present = true)
        if !iszero(midplane_owner) && incidence.side != midplane_owner
            continue
        end
        lo, hi = _face_local_tangent_interval(mesh, ref, face)
        # The owned piece is independent of the side for a segment on the
        # mid-plane, so the closed positive half measures it there.
        wall_side = iszero(midplane_owner) ? incidence.side : Int8(1)
        measure = _embedded_cut_wall_measure(
            geometry, Val(Axis), ref, lo, hi, wall_side, segment,
        )
        measure > zero(T) || continue
        distance = abs(_embedded_plic_distance(geometry, ref, face.center))
        push!(
            connections, WallDiffusionConnection(
                global_to_local[incidence.global_index],
                measure / max(distance, floor), midpoint, true,
            ),
        )
    end
    # A cell that touches a non-periodic boundary in the component direction
    # has no dual face on that side. Its existing node is then the only owner
    # of the strip, so it carries the missing piece over the whole cell at its
    # own wall distance: the pieces still sum to the segment length and the
    # wall flux is not dropped.
    # A segment on the mid-plane is already owned whole by the node found
    # above, so no missing side may add it again.
    isempty(cell_faces) && return connections
    iszero(midplane_owner) || return connections
    for (is_present, side) in ((minus_present, -1), (plus_present, 1))
        is_present && continue
        # Each present face takes the missing side's piece over its own
        # tangential strip, at its own wall distance, so a split side keeps
        # its strip ownership.
        for incidence in cell_faces
            face = mesh.faces[incidence.global_index]
            lo, hi = _face_local_tangent_interval(mesh, ref, face)
            measure = _embedded_cut_wall_measure(
                geometry, Val(Axis), ref, lo, hi, side, segment,
            )
            measure > zero(T) || continue
            distance = abs(_embedded_plic_distance(geometry, ref, face.center))
            push!(
                connections, WallDiffusionConnection(
                    global_to_local[incidence.global_index],
                    measure / max(distance, floor), midpoint, true,
                ),
            )
        end
    end
    return connections
end

function _embedded_wall_connections(
        geometry::GeometryField{T}, ::Val{Axis},
        incidences, global_to_local, aligned,
    ) where {T, Axis}
    mesh = geometry.mesh
    nx, ny = mesh.block_size
    connections = WallDiffusionConnection{T}[]
    for cell_index in eachindex(incidences)
        block = (cell_index - 1) ÷ (nx * ny) + 1
        cell_local = (cell_index - 1) % (nx * ny)
        ref = CellRef(block, cell_local % nx + 1, cell_local ÷ nx + 1)
        cut = _local_plic_segment(geometry.solid, ref)
        segments = isnothing(cut) ? get(aligned, cell_index, ()) : (cut,)
        for segment in segments
            leaf = mesh.blocks[block]
            center = cellcenter(mesh, ref.block, ref.i, ref.j)
            p0 = _local_to_physical(T, center, leaf.spacing, segment[1])
            p1 = _local_to_physical(T, center, leaf.spacing, segment[2])
            midpoint = ((p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2)
            length = hypot(p1[1] - p0[1], p1[2] - p0[2])
            length > zero(T) || continue
            # Cartesian segments recovered from the face aperture use their
            # wall-normal distance and owned portion. A reconstructed PLIC
            # segment of a cut cell, Cartesian or tilted, uses the cut-wall
            # closure below.
            wall_axis = _embedded_cartesian_wall_axis(segment, T)
            if !isnothing(cut)
                _push_embedded_cut_wall!(
                    connections, geometry, Val(Axis), ref, incidences[cell_index],
                    segment, midpoint, global_to_local,
                )
                continue
            end
            for incidence in incidences[cell_index]
                # A normal grid-aligned wall coincides with a component face,
                # which is prescribed zero and removed from the unknowns. The
                # mid-plane measure of its normal contact already carries the
                # full Dirichlet flux to that face, so no wall connection is
                # added here; a tangential face owns its part of the segment.
                wall_axis == Axis && continue
                conductance = _embedded_cartesian_tangent_conductance(
                    geometry, Val(Axis), ref, incidence, segment, midpoint,
                )
                conductance > zero(T) || continue
                push!(
                    connections,
                    WallDiffusionConnection(
                        incidence.local_index, conductance, midpoint, true,
                    ),
                )
            end
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

# For a Cartesian cut the cell-centred PLIC normal points out of the solid.
@inline function _embedded_cartesian_solid_side(solid, ref, axis)
    normal = axis == 1 ? _get(solid.normal_x, ref) : _get(solid.normal_y, ref)
    return normal > zero(normal) ? -1 : 1
end

# The coupled stress operator keeps global face ordering.  These faces are
# prescribed values, not evolving velocities: besides the normal wall face,
# eliminate covered neighbours that participate in its shear stencils.
function _embedded_variable_wall_constraints(geometry::GeometryField{T}) where {T}
    mesh = geometry.mesh
    nx, ny = mesh.block_size
    incidences = ntuple(axis -> first(_component_incidences(mesh, Val(axis), T)), 2)
    constrained = falses(length(mesh.faces))
    covered_cells = Set{CellRef}()
    for (index, face) in pairs(mesh.faces)
        minus = _get(geometry.open_fraction, face.minus)
        plus = _get(geometry.open_fraction, face.plus)
        side = _embedded_pure_wall_side(minus, plus)
        iszero(side) && continue
        constrained[index] = true
        push!(covered_cells, side > 0 ? face.plus : face.minus)
    end
    # Include the zero-aperture, solid-side face of a parallel PLIC segment.
    # This is the same boundary prescription in the grid-face limit.
    for cell_index in eachindex(incidences[1])
        block = (cell_index - 1) ÷ (nx * ny) + 1
        cell_local = (cell_index - 1) % (nx * ny)
        ref = CellRef(block, cell_local % nx + 1, cell_local ÷ nx + 1)
        cut = _local_plic_segment(geometry.solid, ref)
        isnothing(cut) && continue
        axis = _embedded_cartesian_wall_axis(cut, T)
        iszero(axis) && continue
        side = _embedded_cartesian_solid_side(geometry.solid, ref, axis)
        for incidence in incidences[axis][cell_index]
            incidence.side == side || continue
            index = incidence.global_index
            geometry.aperture[index] <= T(64) * eps(T) || continue
            constrained[index] = true
            face = mesh.faces[index]
            neighbour = side > 0 ? face.plus : face.minus
            _get(geometry.open_fraction, neighbour) <= T(64) * eps(T) &&
                push!(covered_cells, neighbour)
        end
    end
    for ref in covered_cells
        cell_index = _cell_index(ref, nx, ny)
        for axis in 1:2, incidence in incidences[axis][cell_index]
            index = incidence.global_index
            iszero(geometry.face_dual_volumes[index]) && (constrained[index] = true)
        end
    end
    return findall(constrained)
end

# A cell's shear quadrature at a Cartesian wall must not also approximate
# shear across the solid: that part is represented by the wall conductance.
# Record, per solid-side corner, the cell and the wall-normal axis; the shear
# rows omit the cell's quarter at that corner where their contact crosses the
# corner's line normal to that axis. Retain any other cells' contributions at
# a corner where wall segments meet. Use lattice corners and periodic
# identification to avoid coordinate roundoff.
function _embedded_variable_wall_stress_vertices(
        geometry::GeometryField{T}, lattice,
    ) where {T}
    mesh = geometry.mesh
    nx, ny = mesh.block_size
    aligned = _embedded_aligned_wall_segments(geometry)
    wall_vertices = Dict{NTuple{2, Int}, Vector{Tuple{Int, Int}}}()
    for block in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(block, i, j)
        cell_index = _cell_index(ref, nx, ny)
        cut = _local_plic_segment(geometry.solid, ref)
        segments = isnothing(cut) ? get(aligned, cell_index, ()) : (cut,)
        isempty(segments) && continue
        low, scale = _cell_lattice_bounds(mesh, mesh.blocks, ref, lattice)
        for segment in segments
            axis = _embedded_cartesian_wall_axis(segment, T)
            iszero(axis) && continue
            side = isnothing(cut) ?
                (segment[1][axis] < zero(T) ? -1 : 1) :
                _embedded_cartesian_solid_side(geometry.solid, ref, axis)
            tangent = 3 - axis
            lo, hi = minmax(segment[1][tangent], segment[2][tangent])
            for tangent_side in (-1, 1)
                coordinate = T(tangent_side) / 2
                lo - T(64) * eps(T) <= coordinate <= hi + T(64) * eps(T) ||
                    continue
                corner = ntuple(2) do dimension
                    high = dimension == axis ? side > 0 : tangent_side > 0
                    value = low[dimension] + (high ? scale[dimension] : 0)
                    mesh.periodic[dimension] ? mod(value, lattice[dimension]) : value
                end
                entries = get!(wall_vertices, corner) do
                    Tuple{Int, Int}[]
                end
                (cell_index, axis) in entries || push!(entries, (cell_index, axis))
            end
        end
    end
    return wall_vertices
end


function _embedded_variable_wall_stresses(
        geometry::GeometryField{T};
        include_cut_wall = true, dirichlet_faces = Int[],
    ) where {T}
    mesh = geometry.mesh
    nx, ny = mesh.block_size
    stresses = Tuple{Int, T, Int}[]
    if include_cut_wall
        aligned = _embedded_aligned_wall_segments(geometry)
        for axis in 1:2
            incidences, _, _ = _component_incidences(mesh, Val(axis), T)
            for cell_index in eachindex(incidences)
                block = (cell_index - 1) ÷ (nx * ny) + 1
                cell_local = (cell_index - 1) % (nx * ny)
                ref = CellRef(
                    block, cell_local % nx + 1, cell_local ÷ nx + 1,
                )
                cut = _local_plic_segment(geometry.solid, ref)
                segments = isnothing(cut) ? get(aligned, cell_index, ()) : (cut,)
                for segment in segments
                    leaf = mesh.blocks[block]
                    center = cellcenter(mesh, ref.block, ref.i, ref.j)
                    p0 = _local_to_physical(T, center, leaf.spacing, segment[1])
                    p1 = _local_to_physical(T, center, leaf.spacing, segment[2])
                    midpoint = (
                        (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2,
                    )
                    length = hypot(p1[1] - p0[1], p1[2] - p0[2])
                    length > zero(T) || continue
                    wall_axis = _embedded_cartesian_wall_axis(segment, T)
                    for incidence in incidences[cell_index]
                        if wall_axis != 0
                            # Normal strain already supplies the normal-wall
                            # flux. Tangential shear is supplied here rather
                            # than by a vertex stencil through the solid.
                            axis == wall_axis && continue
                            conductance = _embedded_cartesian_tangent_conductance(
                                geometry, Val(axis), ref, incidence, segment, midpoint,
                            )
                            conductance > zero(T) || continue
                            push!(stresses, (incidence.global_index, conductance, cell_index))
                            continue
                        end
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
    end
    # Background-grid degrees of freedom whose complete staggered control
    # volume is covered have no physical stress row.  Constrain this harmless
    # algebraic extension to the stationary solid velocity.
    prescribed = Set(dirichlet_faces)
    for axis in 1:2
        global_to_local, _ = _component_face_indices(mesh, axis)
        for (global_index, face) in pairs(mesh.faces)
            face.axis == axis || continue
            global_index in prescribed && continue
            local_index = global_to_local[global_index]
            geometry.dual_volumes[axis][local_index] > zero(T) && continue
            cell_index = _cell_index(face.minus, nx, ny)
            push!(stresses, (global_index, one(T), cell_index))
        end
    end
    return stresses
end

function _wall_connection_measure(
        geometry::GeometryField{T}, mesh, boundary_face, component_face,
        incidence, measure,
    ) where {T}
    ref = boundary_face.cell
    lo, hi = -T(0.5), T(0.5)
    lattice = _finest_lattice_size(mesh)
    cell_lo, scale = _cell_lattice_bounds(mesh, mesh.blocks, ref, lattice)
    _, component_lo, component_hi = _face_lattice_geometry(mesh, component_face, lattice)
    if component_face.axis != boundary_face.axis
        # A split component face owns only the strip it spans in the wall
        # normal direction. Only the strip touching this wall contributes;
        # it owns the whole adjacent half-wall, not its length fraction.
        wall_axis = boundary_face.axis
        wall_coordinate = cell_lo[wall_axis] +
            (boundary_face.side > 0 ? scale[wall_axis] : 0)
        component_lo <= wall_coordinate <= component_hi || return zero(T)
        measure /= incidence.fraction
        # A tangential component face owns the adjacent half of this wall.
        midpoint = (lo + hi) / 2
        if incidence.side < 0
            hi = midpoint
        else
            lo = midpoint
        end
    else
        # A split normal component face owns only its matching wall segment.
        tangent = 3 - boundary_face.axis
        face_lo = T(component_lo - cell_lo[tangent]) / T(scale[tangent]) - T(0.5)
        face_hi = T(component_hi - cell_lo[tangent]) / T(scale[tangent]) - T(0.5)
        lo, hi = max(lo, face_lo), min(hi, face_hi)
    end
    hi > lo || return zero(T)
    solid_fraction = _cell_solid_line_fraction(
        geometry.solid, ref, boundary_face.axis, T(boundary_face.side) / 2, lo, hi,
    )
    return measure * clamp(one(T) - T(solid_fraction), zero(T), one(T))
end
@inline _diffusion_geometry_mesh(mesh::BlockTreeMesh) = mesh
@inline _diffusion_geometry_mesh(geometry::GeometryField) = geometry.mesh
@inline _diffusion_open_degree(::BlockTreeMesh, ::Val, index) = true
@inline _diffusion_open_degree(geometry::GeometryField, ::Val{Axis}, index) where {Axis} =
    geometry.dual_volumes[Axis][index] > 0
@inline _diffusion_open_face(::BlockTreeMesh, index) = true
@inline _diffusion_open_face(geometry::GeometryField, index) = geometry.aperture[index] > 0
@inline _diffusion_solid_line_fraction(
    ::BlockTreeMesh, ref, axis, side, ::Type{T},
) where {T} = zero(T)
@inline _diffusion_solid_line_fraction(
    geometry::GeometryField, ref, axis, side, ::Type{T},
) where {T} = T(_cell_solid_line_fraction(geometry.solid, ref, axis, side))

# Shared by the unembedded and cut-cell paths: an unembedded mesh has no solid
# and no covered degrees of freedom, so its split-side contacts are fully open.
function _add_split_side_diffusion_connections!(
        connections::Vector{DualDiffusionConnection{T}},
        geometry, ::Val{Axis}, incidences,
    ) where {T, Axis}
    mesh = _diffusion_geometry_mesh(geometry)
    nx, ny = mesh.block_size
    tangent = 3 - Axis
    for (cell_index, cell_faces) in pairs(incidences)
        block_index = (cell_index - 1) ÷ (nx * ny) + 1
        cell_local = (cell_index - 1) % (nx * ny)
        ref = CellRef(block_index, cell_local % nx + 1, cell_local ÷ nx + 1)
        block = mesh.blocks[block_index]
        for first_index in eachindex(cell_faces)
            first = cell_faces[first_index]
            _diffusion_open_degree(geometry, Val(Axis), first.local_index) || continue
            for second_index in (first_index + 1):length(cell_faces)
                second = cell_faces[second_index]
                first.side == second.side || continue
                _diffusion_open_degree(geometry, Val(Axis), second.local_index) || continue
                first_face = mesh.faces[first.global_index]
                second_face = mesh.faces[second.global_index]
                first_lo, first_hi = _face_local_tangent_interval(mesh, ref, first_face)
                second_lo, second_hi = _face_local_tangent_interval(mesh, ref, second_face)
                split = if abs(first_hi - second_lo) <= 256eps(T)
                    (first_hi + second_lo) / 2
                elseif abs(second_hi - first_lo) <= 256eps(T)
                    (second_hi + first_lo) / 2
                else
                    continue
                end
                # Each segment owns half of its open coarse-cell strip. Their
                # shared transverse interface therefore has half the open
                # length of the clipped line through the split point. The
                # advection schedule only supplies the fine-cell contribution.
                solid_fraction = _diffusion_solid_line_fraction(
                    geometry, ref, tangent, T(split), T,
                )
                measure = T(0.5) * T(block.spacing[Axis]) *
                    clamp(one(T) - T(solid_fraction), zero(T), one(T))
                measure > zero(T) || continue
                distance = _component_face_distance(
                    mesh, first_face.center, second_face.center, tangent,
                )
                distance > 256eps(T) * mesh.extent[tangent] ||
                    throw(ArgumentError("degenerate split-side diffusion connection"))
                minus, plus = minmax(first.local_index, second.local_index)
                push!(
                    connections, DualDiffusionConnection(
                        minus, plus, tangent, measure, T(distance), true,
                    ),
                )
            end
        end
    end
    return connections
end

function _open_dual_trace(
        geometry::GeometryField{T}, ::Val{Axis}, ref, side, flux_face, ::Type{T},
    ) where {T, Axis}
    mesh = geometry.mesh
    block = mesh.blocks[ref.block]
    center = cellcenter(mesh, ref.block, ref.i, ref.j)
    offset = T(
        _wrapped_delta(
            center[Axis] - flux_face.center[Axis], mesh.extent[Axis], mesh.periodic[Axis],
        )
    )
    width = T(block.spacing[Axis])
    lo, hi = offset - width / 2, offset + width / 2
    solid_fraction = clamp(
        T(_cell_solid_line_fraction(geometry.solid, ref, 3 - Axis, T(side) / 2)),
        zero(T), one(T),
    )
    if solid_fraction >= one(T)
        return lo, lo, lo
    elseif solid_fraction > zero(T)
        normal = Axis == 1 ?
            _get(geometry.solid.normal_x, ref) : _get(geometry.solid.normal_y, ref)
        open_width = (one(T) - solid_fraction) * width
        if normal > zero(T)
            lo = hi - open_width
        else
            hi = lo + open_width
        end
    end
    # Each half of the cell's open trace belongs to the dual face on that side
    # of the cell centre, as on an uncut cell. Splitting at the midpoint of the
    # surviving open part would move the dual boundary whenever the solid clips
    # one side.
    return lo, clamp(offset, lo, hi), hi
end
function _open_dual_trace(
        mesh::BlockTreeMesh, ::Val{Axis}, ref, side, flux_face, ::Type{T},
    ) where {T, Axis}
    block = mesh.blocks[ref.block]
    center = cellcenter(mesh, ref.block, ref.i, ref.j)
    offset = T(
        _wrapped_delta(
            center[Axis] - flux_face.center[Axis], mesh.extent[Axis], mesh.periodic[Axis],
        )
    )
    width = T(block.spacing[Axis])
    return offset - width / 2, offset, offset + width / 2
end


# The advection flux star measures a normal contact by the apertures of the
# two primal flux faces flanking it. For an uncut row that average is the
# interface measure, but at a cut cell the flanking faces lie inside the solid
# and a closed face removes half of the interface. Replace the measure of every
# same-level normal contact by the open area of the interface itself: the cell
# mid-plane, clipped by the solid over the tangential overlap of the two dual
# faces. Callers on an unembedded mesh retain the advection measures.
@inline _cut_cell_normal_measures!(connections, ::BlockTreeMesh, ::Val, incidences) =
    connections

function _cut_cell_normal_measures!(
        connections::Vector{DualDiffusionConnection{T}},
        geometry::GeometryField{T}, ::Val{Axis}, incidences,
    ) where {T, Axis}
    mesh = geometry.mesh
    solid = geometry.solid
    nx, ny = mesh.block_size
    tangent = 3 - Axis
    positions = Dict{Tuple{Int, Int}, Int}()
    for (position, connection) in pairs(connections)
        connection.axis == Axis || continue
        positions[(connection.minus, connection.plus)] = position
    end
    extra = DualDiffusionConnection{T}[]
    for (cell_index, cell_faces) in pairs(incidences)
        length(cell_faces) >= 2 || continue
        block_index = (cell_index - 1) ÷ (nx * ny) + 1
        cell_local = (cell_index - 1) % (nx * ny)
        ref = CellRef(block_index, cell_local % nx + 1, cell_local ÷ nx + 1)
        spacing = T(mesh.blocks[block_index].spacing[Axis])
        tangent_spacing = T(mesh.blocks[block_index].spacing[tangent])
        for first in cell_faces
            first.side < 0 || continue
            first_face = mesh.faces[first.global_index]
            first_lo, first_hi = _face_local_tangent_interval(mesh, ref, first_face)
            for second in cell_faces
                second.side > 0 || continue
                first.local_index == second.local_index && continue
                second_face = mesh.faces[second.global_index]
                distance = _component_face_distance(
                    mesh, first_face.center, second_face.center, Axis,
                )
                # Only the two faces of one cell, one cell width apart, meet at
                # that cell's mid-plane. Level-transition contacts keep the
                # schedule measure.
                abs(distance - spacing) <= 256eps(T) * spacing || continue
                second_lo, second_hi = _face_local_tangent_interval(
                    mesh, ref, second_face,
                )
                lo = max(T(first_lo), T(second_lo))
                hi = min(T(first_hi), T(second_hi))
                hi > lo || continue
                solid_fraction = clamp(
                    T(_cell_solid_line_fraction(solid, ref, Axis, zero(T), lo, hi)),
                    zero(T), one(T),
                )
                measure = (hi - lo) * (one(T) - solid_fraction) * tangent_spacing
                # A mid-plane that is (almost) entirely solid leaves the two
                # faces with no diffusive path. Keep the schedule measure then,
                # so a cut control volume never loses its only normal contact
                # and no row keeps a round-off-sized diagonal: a boundary that
                # grazes the mid-plane produces relative measures of a few
                # hundred eps, well below this 1e-10 margin.
                measure > T(1.0e-10) * (hi - lo) * tangent_spacing || continue
                minus, plus = minmax(first.local_index, second.local_index)
                position = get(positions, (minus, plus), 0)
                if position > 0
                    existing = connections[position]
                    connections[position] = DualDiffusionConnection(
                        minus, plus, Axis, measure, existing.distance,
                        existing.refinement_interface,
                    )
                else
                    push!(
                        extra, DualDiffusionConnection(
                            minus, plus, Axis, measure, T(distance), true,
                        ),
                    )
                end
            end
        end
    end
    append!(connections, extra)
    return connections
end

function _geometric_diffusion_connections(
        geometry, ::Val{Axis}, schedule::AMRDualAdvectionSchedule{T}, incidences,
        local_to_global,
    ) where {T, Axis}
    mesh = _diffusion_geometry_mesh(geometry)
    nx, ny = mesh.block_size
    tangent = 3 - Axis
    excluded_fluxes = falses(length(mesh.faces))
    transverse_connections = DualDiffusionConnection{T}[]
    for (flux_index, flux_face) in pairs(mesh.faces)
        flux_face.axis == tangent || continue
        minus_faces = incidences[_cell_index(flux_face.minus, nx, ny)]
        plus_faces = incidences[_cell_index(flux_face.plus, nx, ny)]
        # The advection product plan is not a dual-face measure: it can pair
        # non-touching siblings and averages the two cut traces. Rebuild every
        # transverse contact from the same bisected-trace overlap, including
        # on uniform grids and away from refinement interfaces.
        split = any(f -> f.fraction < one(T) - 256eps(T), minus_faces) ||
            any(f -> f.fraction < one(T) - 256eps(T), plus_faces)
        level_change = mesh.blocks[flux_face.minus.block].key.level !=
            mesh.blocks[flux_face.plus.block].key.level
        excluded_fluxes[flux_index] = true
        _diffusion_open_face(geometry, flux_index) || continue
        minus_lo, minus_mid, minus_hi = _open_dual_trace(
            geometry, Val(Axis), flux_face.minus, 1, flux_face, T,
        )
        plus_lo, plus_mid, plus_hi = _open_dual_trace(
            geometry, Val(Axis), flux_face.plus, -1, flux_face, T,
        )
        for first in minus_faces, second in plus_faces
            first.local_index == second.local_index && continue
            _diffusion_open_degree(geometry, Val(Axis), first.local_index) || continue
            _diffusion_open_degree(geometry, Val(Axis), second.local_index) || continue
            first_face = mesh.faces[first.global_index]
            second_face = mesh.faces[second.global_index]
            _, first_hi = _face_local_tangent_interval(mesh, flux_face.minus, first_face)
            second_lo, _ = _face_local_tangent_interval(mesh, flux_face.plus, second_face)
            abs(first_hi - T(0.5)) <= 256eps(T) || continue
            abs(second_lo + T(0.5)) <= 256eps(T) || continue
            first_lo, first_end = first.side < 0 ?
                (minus_lo, minus_mid) : (minus_mid, minus_hi)
            second_start, second_hi = second.side < 0 ?
                (plus_lo, plus_mid) : (plus_mid, plus_hi)
            measure = min(first_end, second_hi, T(flux_face.area) / 2) -
                max(first_lo, second_start, -T(flux_face.area) / 2)
            # Retain arbitrarily narrow positive cut contacts; their areas
            # must not acquire a Float32-dependent aperture floor.
            measure > zero(T) || continue
            distance = _component_face_distance(
                mesh, first_face.center, second_face.center, tangent,
            )
            distance > 256eps(T) * mesh.extent[tangent] ||
                throw(ArgumentError("degenerate split-side diffusion connection"))
            minus, plus = minmax(first.local_index, second.local_index)
            refinement_interface = split || level_change ||
                mesh.face_dual_volumes[first.global_index] !=
                mesh.face_dual_volumes[second.global_index]
            push!(
                transverse_connections, DualDiffusionConnection(
                    minus, plus, tangent, measure, T(distance), refinement_interface,
                ),
            )
        end
    end
    connections = _dual_diffusion_connections(
        mesh, schedule, local_to_global; excluded_fluxes,
    )
    _cut_cell_normal_measures!(connections, geometry, Val(Axis), incidences)
    append!(connections, transverse_connections)
    _add_split_side_diffusion_connections!(
        connections, geometry, Val(Axis), incidences,
    )
    return connections
end

function _cut_cell_gradient_stencil(
        mesh, connection::DualDiffusionConnection{T}, local_to_global, adjacency,
    ) where {T}
    first = mesh.faces[local_to_global[connection.minus]]
    second = mesh.faces[local_to_global[connection.plus]]
    axis = connection.axis
    if axis == first.axis &&
            abs(first.area - second.area) > 256eps(T) * max(first.area, second.area)
        fine_index, coarse_index = first.area < second.area ?
            (connection.minus, connection.plus) : (connection.plus, connection.minus)
        fine = mesh.faces[local_to_global[fine_index]]
        coarse = mesh.faces[local_to_global[coarse_index]]
        tangent = 3 - axis
        tangent_offset = _wrapped_delta(
            coarse.center[tangent] - fine.center[tangent],
            mesh.extent[tangent], mesh.periodic[tangent],
        )
        normal_offset = _wrapped_delta(
            coarse.center[axis] - fine.center[axis],
            mesh.extent[axis], mesh.periodic[axis],
        )
        for sibling_index in adjacency[fine_index]
            sibling = mesh.faces[local_to_global[sibling_index]]
            same_side = fine.minus == sibling.minus || fine.plus == sibling.plus
            same_side || continue
            abs(sibling.area - fine.area) <= 256eps(T) * fine.area || continue
            _component_face_distance(
                mesh, fine.center, sibling.center, axis,
            ) <= 256eps(T) * mesh.extent[axis] || continue
            sibling_offset = _wrapped_delta(
                sibling.center[tangent] - fine.center[tangent],
                mesh.extent[tangent], mesh.periodic[tangent],
            )
            abs(sibling_offset) > 256eps(T) * mesh.extent[tangent] || continue
            fraction = tangent_offset / sibling_offset
            zero(T) < fraction < one(T) || continue
            # Both fine segments use the same interpolated normal gradient.
            # Its energy is symmetric and its linear normal flux balances
            # across the coarse side instead of scattering through a two-ring.
            inverse_distance = inv(T(normal_offset))
            return (
                [fine_index, sibling_index, coarse_index],
                T[
                    -(one(T) - fraction) * inverse_distance,
                    -fraction * inverse_distance, inverse_distance,
                ],
            )
        end
    end
    return _gradient_stencil(mesh, connection, local_to_global, adjacency)
end

# A normal velocity vanishes at a stationary wall. Interpolate the far contact
# to the near contact's wall-normal position using that zero boundary value.
# With r = d_near/d_far and k = measure/distance, the resulting wall-contact
# energy k*(u_near-r*u_far)^2 is exactly
# k*r*(u_near-u_far)^2 + k*(1-r)*u_near^2 - k*r*(1-r)*u_far^2.
# The conservative core is a two-point positive flux; the signed wall
# corrections complete the non-negative rank-one energy exactly.
function _wall_contact_stencil!(
        mesh, edge, measure, distance, local_to_global, wall_rows,
        wall_connections, ::Val{Axis}, ::Type{T},
    ) where {Axis, T}
    first_face = mesh.faces[local_to_global[edge[1]]]
    second_face = mesh.faces[local_to_global[edge[2]]]
    lattice = _finest_lattice_size(mesh)
    first_normal, _, _ = _face_lattice_geometry(mesh, first_face, lattice)
    second_normal, _, _ = _face_lattice_geometry(mesh, second_face, lattice)
    for side in (Int8(-1), Int8(1))
        ((edge[1], side) in wall_rows || (edge[2], side) in wall_rows) || continue
        first_distance = side < 0 ? first_normal : lattice[Axis] - first_normal
        second_distance = side < 0 ? second_normal : lattice[Axis] - second_normal
        near_distance, far_distance = minmax(first_distance, second_distance)
        zero(near_distance) < near_distance < far_distance || continue
        near, far = first_distance < second_distance ?
            (edge[1], edge[2]) : (edge[2], edge[1])
        near_face, far_face = first_distance < second_distance ?
            (first_face, second_face) : (second_face, first_face)
        ratio = T(near_distance) / T(far_distance)
        factor = measure / distance
        near_increment = factor * (one(T) - ratio)
        far_decrement = factor * ratio * (one(T) - ratio)
        coordinate = T(mesh.origin[Axis] + (side > 0 ? mesh.extent[Axis] : 0))
        near_point = Axis == 1 ? (coordinate, T(near_face.center[2])) :
            (T(near_face.center[1]), coordinate)
        far_point = Axis == 1 ? (coordinate, T(far_face.center[2])) :
            (T(far_face.center[1]), coordinate)
        push!(
            wall_connections,
            WallDiffusionConnection(near, near_increment, near_point, false, Int8(Axis), side),
            WallDiffusionConnection(far, -far_decrement, far_point, false, Int8(Axis), side),
        )
        inverse_distance = sqrt(ratio) / distance
        return Int[edge[1], edge[2]], T[-inverse_distance, inverse_distance]
    end
    return nothing
end

function _cut_cell_transverse_stencils(
        mesh, connections, local_to_global, wall_connections, boundary,
        ::Val{Axis}, ::Type{T},
    ) where {Axis, T}
    normal = 3 - Axis
    skew_edges = Tuple{Int, Int}[]
    for connection in connections
        connection.axis == normal || continue
        first = mesh.faces[local_to_global[connection.minus]].center
        second = mesh.faces[local_to_global[connection.plus]].center
        _component_face_distance(mesh, first, second, Axis) >
            256eps(T) * mesh.extent[Axis] &&
            push!(skew_edges, (connection.minus, connection.plus))
    end
    isempty(skew_edges) && return nothing
    unique!(skew_edges)
    measures = Dict{Tuple{Int, Int}, T}()
    for connection in connections
        connection.axis == normal || continue
        edge = (connection.minus, connection.plus)
        measures[edge] = get(measures, edge, zero(T)) + connection.measure
    end

    groups = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    # Only anchors touching a skew edge can form a nontrivial centred group.
    for (a, b) in skew_edges
        first = mesh.faces[local_to_global[a]].center
        second = mesh.faces[local_to_global[b]].center
        offset = _wrapped_delta(
            second[normal] - first[normal], mesh.extent[normal], mesh.periodic[normal],
        )
        side = offset > zero(T) ? 1 : -1
        get!(groups, (a, side)) do
            Tuple{Int, Int}[]
        end
        get!(groups, (b, -side)) do
            Tuple{Int, Int}[]
        end
    end
    for edge in keys(measures)
        a, b = edge
        first = mesh.faces[local_to_global[a]].center
        second = mesh.faces[local_to_global[b]].center
        offset = _wrapped_delta(
            second[normal] - first[normal], mesh.extent[normal], mesh.periodic[normal],
        )
        side = offset > zero(T) ? 1 : -1
        first_group = get(groups, (a, side), nothing)
        second_group = get(groups, (b, -side), nothing)
        isnothing(first_group) || push!(first_group, edge)
        isnothing(second_group) || push!(second_group, edge)
    end

    function centred_stencils(candidates)
        stencils = Dict{Tuple{Int, Int}, Tuple{Vector{Int}, Vector{T}}}()
        for (anchor, side) in candidates
            edges = groups[(anchor, side)]
            length(edges) > 1 || continue
            any(edge -> haskey(stencils, edge), edges) && continue
            sort!(edges)
            point = mesh.faces[local_to_global[anchor]].center
            total = zero(T)
            moment = zero(T)
            normal_moment = zero(T)
            scale = zero(T)
            for edge in edges
                neighbor = edge[1] == anchor ? edge[2] : edge[1]
                other = mesh.faces[local_to_global[neighbor]].center
                measure = measures[edge]
                tangent_offset = T(
                    _wrapped_delta(
                        other[Axis] - point[Axis], mesh.extent[Axis], mesh.periodic[Axis],
                    )
                )
                normal_offset = T(
                    _wrapped_delta(
                        other[normal] - point[normal],
                        mesh.extent[normal], mesh.periodic[normal],
                    )
                )
                total += measure
                moment += measure * tangent_offset
                normal_moment += measure * normal_offset
                scale = max(scale, abs(tangent_offset))
            end
            scale > 256eps(T) * mesh.extent[Axis] || continue
            abs(moment) <= 256eps(T) * total * scale || continue
            distance = normal_moment / total
            abs(distance) > 256eps(T) * mesh.extent[normal] || continue
            indices = Int[anchor]
            coefficients = T[-inv(distance)]
            for edge in edges
                neighbor = edge[1] == anchor ? edge[2] : edge[1]
                push!(indices, neighbor)
                push!(coefficients, measures[edge] / total / distance)
            end
            # Every centred group shares a linear-exact gradient; its energy
            # scatters flux according to the actual contact measures.
            stencil = (indices, coefficients)
            for edge in edges
                stencils[edge] = stencil
            end
        end
        return stencils
    end

    candidates = sort!(
        collect(keys(groups));
        by = key -> (mesh.faces[local_to_global[key[1]]].area, key[1], key[2]),
    )
    stencils = centred_stencils(candidates)
    any(edge -> !haskey(stencils, edge), skew_edges) || return stencils
    wall_rows = Set{Tuple{Int, Int8}}()
    for wall in wall_connections
        wall.boundary_axis == Axis && !wall.tangential || continue
        side_boundary = boundary isa VelocityBoundaryConditions ?
            _side_boundary(boundary, Axis, wall.boundary_side) : boundary
        side_boundary isa Union{NoSlip, FreeSlip} || continue
        push!(wall_rows, (wall.degree_of_freedom, wall.boundary_side))
    end
    # Unpaired contacts must not scatter flux into a widened two-ring.
    # Use a Dirichlet partner at stationary walls and an adjacent-pair flux
    # for remaining interior, periodic or non-Dirichlet contacts.
    for edge in skew_edges
        haskey(stencils, edge) && continue
        first = mesh.faces[local_to_global[edge[1]]].center
        second = mesh.faces[local_to_global[edge[2]]].center
        distance = T(_component_face_distance(mesh, first, second, normal))
        stencil = _wall_contact_stencil!(
            mesh, edge, measures[edge], distance, local_to_global, wall_rows,
            wall_connections, Val(Axis), T,
        )
        if isnothing(stencil)
            inverse_distance = inv(distance)
            stencil = (
                Int[edge[1], edge[2]], T[-inverse_distance, inverse_distance],
            )
        end
        stencils[edge] = stencil
    end
    return isempty(stencils) ? nothing : stencils
end

function _geometric_diffusion_stiffness(
        mesh, connections, local_to_global, wall_connections, boundary,
        ::Val{Axis}, ::Type{T},
    ) where {Axis, T}
    transverse_stencils = _cut_cell_transverse_stencils(
        mesh, connections, local_to_global, wall_connections, boundary, Val(Axis), T,
    )
    gradient_stencil = if isnothing(transverse_stencils)
        _cut_cell_gradient_stencil
    else
        function (mesh, connection, local_to_global, adjacency)
            if connection.axis != Axis
                stencil = get(
                    transverse_stencils, (connection.minus, connection.plus), nothing,
                )
                isnothing(stencil) || return stencil
            end
            return _cut_cell_gradient_stencil(
                mesh, connection, local_to_global, adjacency,
            )
        end
    end
    return _diffusion_stiffness(
        mesh, local_to_global, connections,
        wall_connections, boundary, T; gradient_stencil,
    )
end

function _cut_cell_diffusion_component(
        geometry::GeometryField{T}, ::Val{Axis}, boundary, aligned,
    ) where {T, Axis}
    mesh = geometry.mesh
    schedule = _cut_cell_dual_schedule(geometry, Val(Axis), false)
    incidences, global_to_local, _ =
        _component_incidences(mesh, Val(Axis), T)
    _, local_to_global = _component_local_to_global(mesh, Axis)
    volumes = copy(schedule.volumes)
    connections = _geometric_diffusion_connections(
        geometry, Val(Axis), schedule, incidences, local_to_global,
    )
    wall_connections = _wall_diffusion_connections(
        mesh, Val(Axis), T, incidences, global_to_local; geometry,
    )
    append!(
        wall_connections,
        _embedded_wall_connections(
            geometry, Val(Axis), incidences, global_to_local, aligned,
        ),
    )
    stiffness = _geometric_diffusion_stiffness(
        mesh, connections, local_to_global, wall_connections, boundary, Val(Axis), T,
    )
    dirichlet_faces = Int[]
    wall_reactions = WallDiffusionReaction{T}[]
    if !isempty(aligned)
        active = trues(length(local_to_global))
        for (local_index, global_index) in pairs(local_to_global)
            face = mesh.faces[global_index]
            minus = _get(geometry.open_fraction, face.minus)
            plus = _get(geometry.open_fraction, face.plus)
            if !iszero(_embedded_pure_wall_side(minus, plus))
                active[local_index] = false
                push!(dirichlet_faces, global_index)
            end
        end
        if !isempty(dirichlet_faces)
            # The normal velocity is prescribed on the wall itself. Retain
            # the principal energy submatrix so every remaining row sees a
            # zero Dirichlet value without coupling to an evolving wall DOF.
            retained = findall(active)
            renumber = zeros(Int, length(active))
            # Recover the force on the eliminated wall faces from the
            # original stiffness cross-block. For a retained velocity u,
            # the integrated reaction is -sum(S[wall, retained] * u).
            # Coefficients can be signed for widened AMR energy stencils.
            stiffness_rows = rowvals(stiffness)
            stiffness_values = nonzeros(stiffness)
            for (new_index, old_index) in pairs(retained)
                renumber[old_index] = new_index
                coefficient = zero(T)
                for pointer in nzrange(stiffness, old_index)
                    row = @inbounds stiffness_rows[pointer]
                    if !active[row]
                        coefficient -= @inbounds stiffness_values[pointer]
                    end
                end
                iszero(coefficient) ||
                    push!(wall_reactions, WallDiffusionReaction(new_index, coefficient))
            end
            retained_walls = WallDiffusionConnection{T}[]
            for wall in wall_connections
                index = renumber[wall.degree_of_freedom]
                iszero(index) && continue
                push!(
                    retained_walls, WallDiffusionConnection(
                        index, wall.conductance, wall.point, wall.tangential,
                        wall.boundary_axis, wall.boundary_side,
                    ),
                )
            end
            stiffness = stiffness[retained, retained]
            volumes = volumes[retained]
            local_to_global = local_to_global[retained]
            wall_connections = retained_walls
        end
    end
    laplacian = sparse(-spdiagm(0 => inv.(volumes)) * stiffness)
    return StaggeredDiffusionComponent{Axis, T}(
        stiffness, laplacian, volumes, local_to_global, dirichlet_faces, wall_connections,
        wall_reactions,
        zeros(T, length(local_to_global)), zeros(T, length(local_to_global)),
        _maximum_absolute_row_sum(laplacian),
    )
end

"""
    StaggeredDiffusionOperator(geometry; viscosity=1, boundary=NoSlip())

The conservative cut-cell stiffness is symmetric positive semidefinite.
Embedded walls include surfaces aligned with cell faces as well as PLIC cuts.
For Cartesian wall segments the tangential closure uses the wall-normal distance
and the portion of the segment owned by the dual face. A wall-aligned normal
velocity is prescribed zero and removed from the diffusion unknowns.
A cut-cell PLIC segment, Cartesian or tilted, is split among the dual faces by
their tangential strips and by the mid-plane of the component direction; the
conductance of a piece is its length divided by the perpendicular distance from
the node to the reconstructed plane, which reproduces the wall flux of a linear
velocity exactly. A node whose centre lies inside the solid keeps its piece, so
every positive-aperture face stays a projection degree of freedom.
Normal contacts take the open area of the cell mid-plane, so a closed flanking
primal face cannot halve them. Transverse contact measures use overlaps of the
bisected open traces, including across AMR interfaces, and a clipped open trace
is bisected at the cell centre. Centred contacts share a linear-exact gradient;
wall-adjacent orphans use a Dirichlet partner when available, while remaining
orphans use flux-local two-point stencils. Domain-wall measures belong only
to the dual strips that touch the wall.
Advection flux-star redistribution between sibling faces is not a diffusive
interface.
"""
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
    aligned = _embedded_aligned_wall_segments(geometry)
    x = _cut_cell_diffusion_component(geometry, Val(1), boundary, aligned)
    y = _cut_cell_diffusion_component(geometry, Val(2), boundary, aligned)
    return StaggeredDiffusionOperator{
        T, M, typeof(x), typeof(y), typeof(boundary),
    }(geometry.mesh, x, y, viscosity_value, boundary)
end
