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
    # Physical open dual volumes in global interior-face order. Unlike the
    # operator metric, completely covered faces retain zero volume here.
    face_dual_volumes::Vector{T}
    boundary_dual_volumes::NTuple{2, Vector{T}}
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
    isnothing(solid.geometry) || throw(
        ArgumentError(
            "a solid GeometryField cannot itself be conditional on geometry",
        )
    )
    geometry = GeometryField{T, M, typeof(solid)}(
        solid.mesh, solid, CellField(solid.mesh, T), T[], T[],
        (T[], T[]), T[], (T[], T[]), Bool[], levelset,
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

"""Return polygon area and its two first moments in the same coordinates."""
function _polygon_area_moments(poly::AbstractVector{<:NTuple{2, T}}) where {T}
    length(poly) >= 3 || return (zero(T), zero(T), zero(T))
    twice_area = zero(T)
    first_x = zero(T)
    first_y = zero(T)
    previous = last(poly)
    for point in poly
        cross = previous[1] * point[2] - point[1] * previous[2]
        twice_area += cross
        first_x += (previous[1] + point[1]) * cross
        first_y += (previous[2] + point[2]) * cross
        previous = point
    end
    orientation = ifelse(twice_area < zero(T), -one(T), one(T))
    return (
        orientation * twice_area / 2,
        orientation * first_x / 6,
        orientation * first_y / 6,
    )
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
    # If either adjacent Cartesian cell is covered, continuity of the solid
    # region implies that their shared face has zero open measure.  The two
    # pure cells contain no PLIC plane, so averaging their independently
    # inferred traces would otherwise create a spurious aperture of 1/2 at a
    # grid-aligned solid boundary (one trace is closed and the other open).
    # Besides leaking flux through the wall, that false aperture makes a
    # geometry-conditioned liquid field look like a liquid-gas jump to GFM.
    for ref in (face.minus, face.plus)
        _get(solid.fraction, ref) >= one(eltype(solid)) &&
            return zero(eltype(solid))
    end
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

function _geometry_boundary_dual_volumes(
        solid::InterfaceField{T}, ::Val{Axis},
    ) where {T, Axis}
    mesh = solid.mesh
    volumes = zeros(T, _nboundary_faces(mesh, Axis))
    local_index = 0
    tangent = 3 - Axis
    for face in mesh.boundary_faces
        face.axis == Axis || continue
        local_index += 1
        ref = face.cell
        center = cellcenter(mesh, ref.block, ref.i, ref.j)
        spacing = mesh.blocks[ref.block].spacing[tangent]
        lo = (face.center[tangent] - face.area / 2 - center[tangent]) /
            spacing
        hi = lo + face.area / spacing
        bounds = Axis == 1 ? (-0.5, 0.5, lo, hi) :
            (lo, hi, -0.5, 0.5)
        @inbounds volumes[local_index] = T(0.5) *
            _open_subcell_area(solid, ref, bounds...) *
            mesh.blocks[ref.block].volume
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
    resize!(geometry.face_dual_volumes, length(mesh.faces))
    local_x = local_y = 0
    for (index, face) in pairs(mesh.faces)
        if face.axis == 1
            local_x += 1
            value = geometry.dual_volumes[1][local_x]
        else
            local_y += 1
            value = geometry.dual_volumes[2][local_y]
        end
        @inbounds geometry.face_dual_volumes[index] =
            value > zero(T) ? value : zero(T)
    end
    geometry.boundary_dual_volumes = (
        _geometry_boundary_dual_volumes(solid, Val(1)),
        _geometry_boundary_dual_volumes(solid, Val(2)),
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
    solid = InterfaceField(
        coarse_mesh; T, fraction_floor = fine.solid.fraction_floor,
    )
    for (index, ref) in pairs(_cell_references(coarse_mesh))
        _set!(solid.fraction, coarse_fraction[index], ref)
    end
    reconstruct!(solid)
    return GeometryField(solid; levelset = fine.levelset)
end
