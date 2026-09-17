"""
    GeometryField(mesh, levelset; kwargs...)
    GeometryField(solid::InterfaceField)

A fixed embedded solid represented by the same cell-averaged volume fractions
and cell-local PLIC planes as a phase [`InterfaceField`](@ref).  The convention
is `levelset(x,y) <= 0` inside the solid.  `open_fraction` is the complementary
fluid fraction.  Shared open-face apertures and the staggered open control
volumes of Verstappen and Dröge (2005) are cached by [`refresh_geometry!`](@ref).

The solid interface is fixed in physical space, but its discrete PLIC data are
geometrically restricted and prolonged when the background tree adapts.
"""
mutable struct GeometryField{
        T, M <: BlockTreeMesh, F <: InterfaceField{T, M},
    } <: AbstractMeshField{T}
    mesh::M
    solid::F
    open_fraction::CellField{T, M}
    aperture::Vector{T}
    boundary_aperture::Vector{T}
    dual_volumes::NTuple{2, Vector{T}}
    active_cells::Vector{Bool}
    levelset::Any
end

placement(::GeometryField) = Cell()
Base.eltype(::GeometryField{T}) where {T} = T
interior(field::GeometryField, block::Integer) =
    interior(field.solid.fraction, block)
blockdata(field::GeometryField, block::Integer) =
    blockdata(field.solid.fraction, block)
integrate(field::GeometryField) = integrate(field.solid.fraction)

function GeometryField(
        solid::InterfaceField{T, M}; levelset = nothing,
    ) where {T, M}
    isnothing(solid.geometry) || throw(ArgumentError(
        "a solid GeometryField cannot itself be conditional on geometry",
    ))
    geometry = GeometryField{T, M, typeof(solid)}(
        solid.mesh, solid, CellField(solid.mesh, T), T[], T[],
        (T[], T[]), Bool[], levelset,
    )
    return refresh_geometry!(geometry)
end

function GeometryField(
        mesh::BlockTreeMesh, levelset;
        T = eltype(mesh.origin), kwargs...
    )
    return GeometryField(
        InterfaceField(mesh, levelset; T, kwargs...); levelset,
    )
end

function Base.copy(geometry::GeometryField)
    return GeometryField(copy(geometry.solid); levelset = geometry.levelset)
end

"""
    GeometryRefinementIndicator(geometry; level, buffer_cells=1,
                                fraction_floor=geometry.solid.fraction_floor,
                                t_init=-Inf, t_end=Inf)

Refine blocks intersecting an embedded solid boundary, or its cell-halo
neighbourhood, to a fixed level. Blocks away from the boundary request
coarsening, including blocks wholly inside the fixed solid. A boundary aligned
with cell or block faces is detected from halo jumps in the solid fraction.
The criterion is active for `t_init <= time <= t_end`.
"""
struct GeometryRefinementIndicator{G <: GeometryField, T} <:
    AbstractRefinementIndicator
    field::G
    level::Int
    buffer_cells::Int
    fraction_floor::T
    t_init::T
    t_end::T
end

function GeometryRefinementIndicator(
        field::GeometryField{T}; level,
        buffer_cells = 1, fraction_floor = field.solid.fraction_floor,
        t_init = -Inf, t_end = Inf
    ) where {T}
    interface_indicator = InterfaceRefinementIndicator(
        field.solid; level, buffer_cells, fraction_floor, t_init, t_end
    )
    return GeometryRefinementIndicator(
        field, interface_indicator.level, interface_indicator.buffer_cells,
        interface_indicator.fraction_floor, interface_indicator.t_init,
        interface_indicator.t_end
    )
end

_indicator_field(indicator::GeometryRefinementIndicator) = indicator.field

function refinement_labels!(
        labels::AbstractVector{Symbol},
        indicator::GeometryRefinementIndicator;
        time = 0
    )
    length(labels) == nblocks(indicator.field.mesh) || throw(
        DimensionMismatch("one label is required per block")
    )
    if !_indicator_active(indicator, time)
        fill!(labels, :coarsen)
        return labels
    end
    return _fixed_interface_refinement_labels!(
        labels, indicator.field.solid, indicator.level,
        indicator.buffer_cells, indicator.fraction_floor
    )
end

_refinement_labels_at_time!(
    labels, indicator::GeometryRefinementIndicator, time
) = refinement_labels!(labels, indicator; time)

@inline function _polygon_crossing(first, second, value_first, value_second)
    fraction = value_first / (value_first - value_second)
    return (
        first[1] + fraction * (second[1] - first[1]),
        first[2] + fraction * (second[2] - first[2]),
    )
end


function _clip_polygon(poly, nx, ny, intercept; keep_less = true)
    isempty(poly) && return copy(poly)
    result = eltype(poly)[]
    previous = last(poly)
    previous_value = nx * previous[1] + ny * previous[2] - intercept
    previous_inside = keep_less ? previous_value <= 0 : previous_value >= 0
    for current in poly
        current_value = nx * current[1] + ny * current[2] - intercept
        current_inside = keep_less ? current_value <= 0 : current_value >= 0
        if current_inside != previous_inside
            push!(
                result,
                _polygon_crossing(
                    previous, current, previous_value, current_value,
                ),
            )
        end
        current_inside && push!(result, current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    end
    return result
end

function _polygon_area(poly::AbstractVector{<:NTuple{2, T}}) where {T}
    length(poly) >= 3 || return zero(T)
    area = zero(T)
    previous = last(poly)
    for point in poly
        area += previous[1] * point[2] - previous[2] * point[1]
        previous = point
    end
    return abs(area) / 2
end

function _open_cell_polygon(geometry::GeometryField{T}, ref::CellRef) where {T}
    square = NTuple{2, T}[
        (-T(0.5), -T(0.5)), (T(0.5), -T(0.5)),
        (T(0.5), T(0.5)), (-T(0.5), T(0.5)),
    ]
    solid_fraction = _get(geometry.solid.fraction, ref)
    solid_fraction <= zero(T) && return square
    solid_fraction >= one(T) && return NTuple{2, T}[]
    return _clip_polygon(
        square,
        _get(geometry.solid.normal_x, ref),
        _get(geometry.solid.normal_y, ref),
        _get(geometry.solid.intercept, ref); keep_less = false,
    )
end

function _open_plane_intercept(
        geometry::GeometryField{T}, ref::CellRef,
        conditional_fraction, nx, ny,
    ) where {T}
    poly = _open_cell_polygon(geometry, ref)
    isempty(poly) && return zero(T)
    open_area = _polygon_area(poly)
    target = clamp(T(conditional_fraction), zero(T), one(T)) * open_area
    projections = [nx * point[1] + ny * point[2] for point in poly]
    lo, hi = extrema(projections)
    target <= zero(T) && return lo
    target >= open_area && return hi
    for _ in 1:54
        middle = (lo + hi) / 2
        clipped = _clip_polygon(poly, nx, ny, middle; keep_less = true)
        if _polygon_area(clipped) < target
            lo = middle
        else
            hi = middle
        end
    end
    return (lo + hi) / 2
end

@inline function _solid_line_fraction(normal, intercept, axis, side, lo, hi)
    tangent = 3 - axis
    nnormal = normal[axis]
    ntangent = normal[tangent]
    threshold = intercept - nnormal * side
    width = hi - lo
    width > zero(width) || return zero(width)
    tolerance = 64eps(typeof(float(intercept)))
    if abs(ntangent) <= tolerance
        return threshold >= zero(threshold) ? one(width) : zero(width)
    end
    crossing = threshold / ntangent
    length = ntangent > 0 ? clamp(crossing - lo, zero(width), width) :
        clamp(hi - crossing, zero(width), width)
    return length / width
end

function _cell_solid_line_fraction(
        solid::InterfaceField, ref::CellRef, axis, side,
        tangent_lo = -0.5, tangent_hi = 0.5,
    )
    fraction = _get(solid.fraction, ref)
    fraction <= zero(fraction) && return zero(fraction)
    fraction >= one(fraction) && return one(fraction)
    normal = (_get(solid.normal_x, ref), _get(solid.normal_y, ref))
    return _solid_line_fraction(
        normal, _get(solid.intercept, ref), axis, side,
        tangent_lo, tangent_hi,
    )
end

function _face_local_tangent_interval(mesh, ref, face)
    tangent = 3 - face.axis
    center = cellcenter(mesh, ref.block, ref.i, ref.j)
    spacing = mesh.blocks[ref.block].spacing[tangent]
    lo = _wrapped_delta(
        face.center[tangent] - face.area / 2 - center[tangent],
        mesh.extent[tangent], mesh.periodic[tangent],
    ) / spacing
    hi = lo + face.area / spacing
    return lo, hi
end

@inline function _cell_face_open_fraction(solid, ref, face, side)
    lo, hi = _face_local_tangent_interval(solid.mesh, ref, face)
    return clamp(
        one(eltype(solid)) - _cell_solid_line_fraction(
            solid, ref, face.axis, side / 2, lo, hi,
        ), zero(eltype(solid)), one(eltype(solid)),
    )
end

function _interior_open_aperture(solid, face)
    minus = _cell_face_open_fraction(solid, face.minus, face, 1)
    plus = _cell_face_open_fraction(solid, face.plus, face, -1)
    # A single shared aperture is required for exact conservation and for the
    # pressure gradient to remain the adjoint of divergence.  Averaging the two
    # independently reconstructed traces is symmetric and second-order for a
    # resolved smooth boundary.
    return clamp((minus + plus) / 2, zero(minus), one(minus))
end

@inline _boundary_open_aperture(solid, face) =
    _cell_face_open_fraction(solid, face.cell, face, face.side)

function _open_subcell_area(
        solid::InterfaceField{T}, ref::CellRef,
        xlo, xhi, ylo, yhi,
    ) where {T}
    fraction = _get(solid.fraction, ref)
    normalized_area = (xhi - xlo) * (yhi - ylo)
    fraction <= zero(T) && return normalized_area
    fraction >= one(T) && return zero(T)
    solid_area = _plane_fraction(
        _get(solid.normal_x, ref), _get(solid.normal_y, ref),
        _get(solid.intercept, ref), xlo, xhi, ylo, yhi,
    ) * normalized_area
    return clamp(normalized_area - solid_area, zero(T), normalized_area)
end

function _geometry_dual_volumes(
        solid::InterfaceField{T}, ::Val{Axis}, aperture,
    ) where {T, Axis}
    mesh = solid.mesh
    incidences, global_to_local, _ = _component_incidences(
        mesh, Val(Axis), T,
    )
    volumes = zeros(T, maximum(global_to_local; init = 0))
    nx, ny = mesh.block_size
    tangent = 3 - Axis
    for cell_index in eachindex(incidences)
        block_index = (cell_index - 1) ÷ (nx * ny) + 1
        cell_local = (cell_index - 1) % (nx * ny)
        ref = CellRef(
            block_index, cell_local % nx + 1, cell_local ÷ nx + 1,
        )
        block = mesh.blocks[block_index]
        cell_center = cellcenter(mesh, ref.block, ref.i, ref.j)
        for incidence in incidences[cell_index]
            face = mesh.faces[incidence.global_index]
            face_center = face.center[tangent]
            tangent_spacing = block.spacing[tangent]
            lo = _wrapped_delta(
                face_center - face.area / 2 - cell_center[tangent],
                mesh.extent[tangent], mesh.periodic[tangent],
            ) / tangent_spacing
            hi = lo + face.area / tangent_spacing
            bounds = Axis == 1 ? (-0.5, 0.5, lo, hi) :
                (lo, hi, -0.5, 0.5)
            # The paper bisects every open line segment in the component
            # direction.  Therefore either neighboring staggered control
            # volume owns exactly half of the clipped strip measure.
            portion = _open_subcell_area(solid, ref, bounds...)
            local_index = global_to_local[incidence.global_index]
            @inbounds volumes[local_index] += T(0.5) * portion * block.volume
        end
    end
    return volumes
end

"""Reconstruct the fixed solid and refresh all open cut-cell metrics."""
function refresh_geometry!(geometry::GeometryField{T}) where {T}
    solid = reconstruct!(geometry.solid)
    mesh = geometry.mesh
    for field in (geometry.open_fraction,)
        _resize_interface_cache!(field)
    end
    nx, ny = mesh.block_size
    h = mesh.halo
    resize!(geometry.active_cells, ncells(mesh))
    _parallel_block_columns(mesh) do block, first_j, last_j
        source = @inbounds solid.fraction.blocks[block]
        destination = @inbounds geometry.open_fraction.blocks[block]
        offset = (block - 1) * nx * ny
        @inbounds for j in first_j:last_j, i in 1:nx
            open = clamp(one(T) - source[i + h, j + h], zero(T), one(T))
            destination[i + h, j + h] = open
            geometry.active_cells[offset + (j - 1) * nx + i] = open > zero(T)
        end
    end
    fill_halo!(geometry.open_fraction, ZeroGradient())
    resize!(geometry.aperture, length(mesh.faces))
    @_parallel for index in eachindex(mesh.faces)
        @inbounds geometry.aperture[index] =
            _interior_open_aperture(solid, mesh.faces[index])
    end
    resize!(geometry.boundary_aperture, length(mesh.boundary_faces))
    @_parallel for index in eachindex(mesh.boundary_faces)
        @inbounds geometry.boundary_aperture[index] =
            _boundary_open_aperture(solid, mesh.boundary_faces[index])
    end
    geometry.dual_volumes = (
        _geometry_dual_volumes(solid, Val(1), geometry.aperture),
        _geometry_dual_volumes(solid, Val(2), geometry.aperture),
    )
    return geometry
end

struct GeometryProlongation{P, G}
    interface::P
    geometry::G
end

function _prepare_prolongation!(
        strategy::GeometryProlongation, field, workspace::AMRWorkspace,
    )
    return _prepare_prolongation!(strategy.interface, field, workspace)
end
_prepare_parallel_prolongation!(strategy::GeometryProlongation, workspace) =
    _prepare_parallel_prolongation!(strategy.interface, workspace)
(strategy::GeometryProlongation)(child, parent, halo, key::BlockKey) =
    strategy.interface(child, parent, halo, key)
function _restrict_parent!(
        strategy::GeometryProlongation, parent, children, halo, key,
    )
    return _restrict_parent!(strategy.interface, parent, children, halo, key)
end
function _finish_prolongation!(strategy::GeometryProlongation)
    _finish_prolongation!(strategy.interface)
    refresh_geometry!(strategy.geometry)
    return nothing
end

function register_adaptive_field!(
        workspace::AMRWorkspace, geometry::GeometryField{T},
    ) where {T}
    fraction = geometry.solid.fraction
    index = _find_field_buffer(workspace, fraction)
    if !isnothing(index)
        workspace.field_buffers[index].prolongation isa GeometryProlongation &&
            return workspace
        throw(ArgumentError("register GeometryField before its solid fraction"))
    end
    interface = InterfaceProlongation(
        geometry.solid, Dict{BlockKey, NTuple{3, Matrix{T}}}(),
        NTuple{3, Matrix{T}}[], Any[],
        Dict{BlockKey, NTuple{3, Matrix{T}}}(),
    )
    strategy = GeometryProlongation(interface, geometry)
    register_adaptive_field!(workspace, fraction; prolongation = strategy)
    return workspace
end

function _restrict_geometry_to_mesh(
        fine::GeometryField{T}, coarse_mesh::BlockTreeMesh,
        restriction::CellCoefficientRestriction{T},
    ) where {T}
    fine_fraction = [
        _get(fine.solid.fraction, ref)
            for ref in _cell_references(fine.mesh)
    ]
    coarse_fraction = zeros(T, ncells(coarse_mesh))
    _restrict_cell_coefficient!(
        coarse_fraction, fine_fraction, restriction,
    )
    solid = InterfaceField(coarse_mesh; T)
    for (index, ref) in pairs(_cell_references(coarse_mesh))
        _set!(solid.fraction, coarse_fraction[index], ref)
    end
    reconstruct!(solid)
    return GeometryField(solid; levelset = fine.levelset)
end

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
    mesh = geometry.mesh
    interior_count = length(geometry.dual_volumes[Axis])
    volumes = zeros(T, interior_count + _nboundary_faces(mesh, Axis))
    copyto!(volumes, 1, geometry.dual_volumes[Axis], 1, interior_count)
    local_index = interior_count
    for face in mesh.boundary_faces
        face.axis == Axis || continue
        local_index += 1
        ref = face.cell
        tangent = 3 - Axis
        center = cellcenter(mesh, ref.block, ref.i, ref.j)
        spacing = mesh.blocks[ref.block].spacing[tangent]
        lo = (face.center[tangent] - face.area / 2 - center[tangent]) /
            spacing
        hi = lo + face.area / spacing
        bounds = Axis == 1 ? (-0.5, 0.5, lo, hi) :
            (lo, hi, -0.5, 0.5)
        @inbounds volumes[local_index] = T(0.5) *
            _open_subcell_area(geometry.solid, ref, bounds...) *
            mesh.blocks[ref.block].volume
    end
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
        velocity_boundary = nothing, boundary_time = zero(T), kwargs...
    ) where {T}
    velocity.mesh === pressure.mesh === solver.operator.mesh ===
        workspace.divergence.mesh === geometry.mesh || throw(
        ArgumentError("cut-cell projection arguments must share a mesh"),
    )
    iszero(solver.operator.reaction) || throw(
        ArgumentError("pressure projection requires a zero-reaction operator"),
    )
    dt > zero(dt) || throw(ArgumentError("dt must be strictly positive"))
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
    stats = solve!(pressure, solver, workspace.divergence; kwargs...)
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

"""Return physical PLIC line segments of a reconstructed field."""
function plic_segments(field::InterfaceField{T}) where {T}
    segments = NTuple{2, NTuple{2, T}}[]
    nx, ny = field.mesh.block_size
    for block in eachindex(field.mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(block, i, j)
        fraction = _conditional_fraction(field, ref)
        zero(T) < fraction < one(T) || continue
        segment = _local_plic_segment(field, ref)
        isnothing(segment) && continue
        points = segment
        leaf = field.mesh.blocks[block]
        center = cellcenter(field.mesh, block, i, j)
        physical(point) = (
            T(center[1] + point[1] * leaf.spacing[1]),
            T(center[2] + point[2] * leaf.spacing[2]),
        )
        push!(segments, (physical(points[1]), physical(points[2])))
    end
    return segments
end

plic_segments(geometry::GeometryField) = plic_segments(geometry.solid)

"""
    PLICSolidGeometry

Triangulated physical-space solid reconstructed from the cell-local PLIC
planes of a [`GeometryField`](@ref).  Vertices are private to each cut cell so
the object remains valid across later mesh adaptation.
"""
struct PLICSolidGeometry{T}
    points::Vector{NTuple{2, T}}
    triangles::Vector{NTuple{3, Int}}
end

"""Return a filled PLIC reconstruction of the solid for visualization."""
function plic_solid_geometry(geometry::GeometryField{T}) where {T}
    points = NTuple{2, T}[]
    triangles = NTuple{3, Int}[]
    square = NTuple{2, T}[
        (-T(0.5), -T(0.5)), (T(0.5), -T(0.5)),
        (T(0.5), T(0.5)), (-T(0.5), T(0.5)),
    ]
    nx, ny = geometry.mesh.block_size
    for block in eachindex(geometry.mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(block, i, j)
        fraction = _get(geometry.solid.fraction, ref)
        fraction <= zero(T) && continue
        polygon = if fraction >= one(T)
            square
        else
            _clip_polygon(
                square,
                _get(geometry.solid.normal_x, ref),
                _get(geometry.solid.normal_y, ref),
                _get(geometry.solid.intercept, ref); keep_less = true,
            )
        end
        length(polygon) >= 3 || continue
        leaf = geometry.mesh.blocks[block]
        center = cellcenter(geometry.mesh, block, i, j)
        first_vertex = length(points) + 1
        for point in polygon
            push!(points, (
                T(center[1] + point[1] * leaf.spacing[1]),
                T(center[2] + point[2] * leaf.spacing[2]),
            ))
        end
        for vertex in 2:(length(polygon) - 1)
            push!(triangles, (
                first_vertex, first_vertex + vertex - 1,
                first_vertex + vertex,
            ))
        end
    end
    return PLICSolidGeometry(points, triangles)
end
