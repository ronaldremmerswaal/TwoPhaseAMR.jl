"""
    InterfaceField(mesh; init=0)
    InterfaceField(fraction::CellField)

Centred liquid volume fractions and derived, cell-local PLIC planes. The liquid
occupies `normal_x*ξ + normal_y*η <= intercept`, with local coordinates in
`[-1/2,1/2]^2`. `reconstruct!` refreshes derived data after modifying fractions.
The field constructor takes ownership of `fraction` (it does not copy it).

When an embedded `geometry` is present, `fraction` is the liquid volume divided
by the complete Cartesian-cell volume.  It therefore obeys
`0 <= fraction <= geometry.open_fraction`.  `conditional_fraction` is the
derived liquid fraction within the open part of the cell and is used by PLIC,
height functions, and the two-phase constitutive laws.
"""
struct InterfaceField{T, M <: BlockTreeMesh} <: AbstractMeshField{T}
    mesh::M
    fraction::CellField{T, M}
    conditional_fraction::CellField{T, M}
    normal_x::CellField{T, M}
    normal_y::CellField{T, M}
    intercept::CellField{T, M}
    curvature::CellField{T, M}
    reconstruction::CellField{UInt8, M}
    previous_normal_x::CellField{T, M}
    previous_normal_y::CellField{T, M}
    previous_intercept::CellField{T, M}
    height_cache::HeightFunctionCache{T}
    contact_angles::NTuple{4, Union{Nothing, T}}
    geometry::Any
    embedded_contact_angle::Union{Nothing, T}
end
placement(::InterfaceField) = Cell()
Base.eltype(::InterfaceField{T}) where {T} = T
interior(field::InterfaceField, b::Integer) = interior(field.fraction, b)
blockdata(field::InterfaceField, b::Integer) = blockdata(field.fraction, b)
integrate(field::InterfaceField) = integrate(field.fraction)
function InterfaceField(
        fraction::CellField{T, M}; contact_angle = nothing,
        geometry = nothing, embedded_contact_angle = nothing,
    ) where {T <: AbstractFloat, M}
    isnothing(geometry) || (
        hasproperty(geometry, :mesh) && geometry.mesh === fraction.mesh
    ) || throw(ArgumentError("geometry must share the interface mesh"))
    embedded = isnothing(embedded_contact_angle) && !isnothing(geometry) &&
        contact_angle isa Real ? contact_angle : embedded_contact_angle
    wall_contact = !isnothing(geometry) && contact_angle isa Real ? nothing :
        contact_angle
    angles = isnothing(wall_contact) ? (nothing, nothing, nothing, nothing) :
        wall_contact isa Real ? ntuple(_ -> T(wall_contact), 4) : wall_contact
    angles isa Tuple && length(angles) == 4 &&
        all(a -> isnothing(a)||(isfinite(a)&&0 < a < pi), angles) ||
        throw(ArgumentError("contact_angle must be in (0,π), or a (left,right,bottom,top) tuple with optional nothing entries"))
    angles = ntuple(i -> isnothing(angles[i]) ? nothing : T(angles[i]), 4)
    isnothing(embedded) || (
        isfinite(embedded) && 0 < embedded < pi
    ) || throw(ArgumentError("embedded_contact_angle must be in (0,π)"))
    field = InterfaceField{T, M}(
        fraction.mesh, fraction, CellField(fraction.mesh, T),
        CellField(fraction.mesh, T), CellField(fraction.mesh, T), CellField(fraction.mesh, T),
        CellField(fraction.mesh, T; init = NaN), CellField(fraction.mesh, UInt8),
        CellField(fraction.mesh, T), CellField(fraction.mesh, T),
        CellField(fraction.mesh, T), HeightFunctionCache(T), angles,
        geometry, isnothing(embedded) ? nothing : T(embedded),
    )
    return reconstruct!(field)
end
InterfaceField(
    mesh::BlockTreeMesh; init = 0, T = eltype(mesh.origin),
    contact_angle = nothing, geometry = nothing,
    embedded_contact_angle = nothing,
) = begin
    values = CellField(mesh, T; init = isnothing(geometry) ? init : zero(T))
    if !isnothing(geometry)
        init isa Number || throw(ArgumentError(
            "geometry-aware InterfaceField init must be a scalar",
        ))
        isfinite(init) && zero(init) <= init <= one(init) || throw(
            ArgumentError("InterfaceField init must lie in [0,1]"),
        )
        for block in eachindex(mesh.blocks)
            interior(values, block) .= T(init) .* interior(
                geometry.open_fraction, block,
            )
        end
    end
    InterfaceField(
        values; contact_angle, geometry, embedded_contact_angle,
    )
end
Base.copy(field::InterfaceField) = InterfaceField(
    copy(field.fraction); contact_angle = field.contact_angles,
    geometry = field.geometry,
    embedded_contact_angle = field.embedded_contact_angle,
)

@inline _conditional_fraction(field::InterfaceField, ref::CellRef) =
    _get(field.conditional_fraction, ref)

function _refresh_conditional_fraction!(field::InterfaceField{T}) where {T}
    mesh = field.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    _resize_interface_cache!(field.conditional_fraction)
    _parallel_block_columns(mesh) do block, first_j, last_j
        physical = @inbounds field.fraction.blocks[block]
        conditional = @inbounds field.conditional_fraction.blocks[block]
        open = isnothing(field.geometry) ? nothing :
            @inbounds field.geometry.open_fraction.blocks[block]
        @inbounds for j in first_j:last_j, i in 1:nx
            ii, jj = i + h, j + h
            available = isnothing(open) ? one(T) : open[ii, jj]
            value = physical[ii, jj]
            tolerance = T(128) * eps(T) * max(one(T), available)
            isfinite(value) && value >= -tolerance &&
                value <= available + tolerance || throw(ArgumentError(
                    "liquid volume fraction $value lies outside [0,$available] " *
                    "in cell $(CellRef(block, i, j))",
                ))
            value = clamp(value, zero(T), available)
            ratio = available > zero(T) ? value / available : zero(T)
            if ratio <= T(128) * eps(T)
                value = zero(T)
                ratio = zero(T)
            elseif ratio >= one(T) - T(128) * eps(T)
                value = available
                ratio = one(T)
            end
            physical[ii, jj] = value
            conditional[ii, jj] = ratio
        end
    end
    fill_halo!(field.fraction)
    fill_halo!(field.conditional_fraction)
    return field.conditional_fraction
end

# Exact area fraction of a rectangle clipped by one plane. Reflect the axes
# so both coefficients are positive, then integrate a triangular distribution.
@inline function _plane_fraction(nx, ny, a, xlo = -0.5, xhi = 0.5, ylo = -0.5, yhi = 0.5)
    ax, ay = abs(nx) * (xhi - xlo), abs(ny) * (yhi - ylo)
    shift = nx * (nx >= 0 ? xlo : xhi) + ny * (ny >= 0 ? ylo : yhi)
    t = a - shift
    t <= 0 && return zero(a)
    t >= ax + ay && return one(a)
    small, large = minmax(ax, ay)
    small <= eps(typeof(a)) * large && return clamp(t / large, zero(a), one(a))
    # Symmetry avoids cancellation near full cells.
    reflected = t > (ax + ay) / 2
    z = reflected ? ax + ay - t : t
    result = z < small ? z * z / (2small * large) : (z - small / 2) / large
    return reflected ? one(a) - result : result
end

function _plane_intercept(c, nx, ny)
    radius = (abs(nx) + abs(ny)) / 2
    c <= 0 && return -radius
    c >= 1 && return radius
    lo, hi = -radius, radius
    for _ in 1:54
        middle = (lo + hi) / 2
        if _plane_fraction(nx, ny, middle) < c
            lo = middle
        else
            hi = middle
        end
    end
    return (lo + hi) / 2
end

function _embedded_contact_normal(field, ref, wy_normal)
    geometry = field.geometry
    theta = field.embedded_contact_angle
    (isnothing(geometry) || isnothing(theta)) && return wy_normal
    solid_fraction = _get(geometry.solid.fraction, ref)
    zero(eltype(field)) < solid_fraction < one(eltype(field)) ||
        return wy_normal
    hx, hy = field.mesh.blocks[ref.block].spacing
    sx = _get(geometry.solid.normal_x, ref) / hx
    sy = _get(geometry.solid.normal_y, ref) / hy
    wall_scale = hypot(sx, sy)
    wall_scale > eps(eltype(field)) || return wy_normal
    wall = (sx / wall_scale, sy / wall_scale)
    tangent = (-wall[2], wall[1])
    cosine, sine = cos(theta), sin(theta)
    candidates = (
        (cosine * wall[1] + sine * tangent[1],
            cosine * wall[2] + sine * tangent[2]),
        (cosine * wall[1] - sine * tangent[1],
            cosine * wall[2] - sine * tangent[2]),
    )
    wx, wy = wy_normal[1] / hx, wy_normal[2] / hy
    wy_scale = hypot(wx, wy)
    wy_physical = wy_scale > eps(eltype(field)) ?
        (wx / wy_scale, wy / wy_scale) : candidates[1]
    selected = sum(candidates[1] .* wy_physical) >=
        sum(candidates[2] .* wy_physical) ? candidates[1] : candidates[2]
    mx, my = selected[1] * hx, selected[2] * hy
    scale = abs(mx) + abs(my)
    return mx / scale, my / scale
end

function _resize_interface_cache!(field::CellField)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    while length(field.blocks) < nblocks(field.mesh)
        push!(field.blocks, zeros(eltype(field), nx + 2h, ny + 2h))
    end
    return resize!(field.blocks, nblocks(field.mesh))
end

function _reconstruct_interface_cell!(field, values, b, i, j)
    block = field.mesh.blocks[b]
    ref = CellRef(b, i, j)
    c = _conditional_fraction(field, ref)
    if c == 0 || c == 1
        _set!(field.reconstruction, 0, ref)
        _set!(field.curvature, NaN, ref)
        return nothing
    end
    geometry = _contact_geometry(field, values, b, i, j)
    method = 4
    embedded_triple = !isnothing(field.geometry) &&
        !isnothing(field.embedded_contact_angle) &&
        zero(c) < _get(field.geometry.solid.fraction, ref) < one(c)
    if isnothing(geometry) && !embedded_triple
        geometry = _lhf_geometry(values, i + 4, j + 4, block.spacing)
        method = 1
    end
    if isnothing(geometry)
        geometry = _ghf_geometry(
            field, values, b, i, j,
            field.previous_normal_x, field.previous_normal_y,
            field.previous_intercept,
        )
        method = 2
    end
    if isnothing(geometry)
        _set!(field.reconstruction, 3, ref)
        _set!(field.curvature, NaN, ref)
        return nothing
    end
    gx, gy, kappa = geometry
    mx, my = gx * block.spacing[1], gy * block.spacing[2]
    scale = abs(mx) + abs(my)
    mx, my = mx / scale, my / scale
    _set!(field.normal_x, mx, ref)
    _set!(field.normal_y, my, ref)
    intercept = isnothing(field.geometry) ? _plane_intercept(c, mx, my) :
        _open_plane_intercept(field.geometry, ref, c, mx, my)
    _set!(field.intercept, intercept, ref)
    _set!(field.curvature, kappa, ref)
    _set!(field.reconstruction, method, ref)
    return nothing
end

"""Reconstruct PLIC planes with LHF normals, GHF fits and a final WY fallback.
`reconstruction` records 0 (pure), 1 (LHF), 2 (GHF), 3 (WY), or 4
(wall contact-angle reconstruction).
Curvature is NaN where neither a height stencil nor a GHF fit is available.
"""
function reconstruct!(field::InterfaceField)
    mesh = field.mesh
    mesh.halo >= 1 || throw(ArgumentError("interface reconstruction requires one halo cell"))
    for cache in (
            field.conditional_fraction, field.normal_x, field.normal_y,
            field.intercept, field.curvature,
            field.reconstruction, field.previous_normal_x,
            field.previous_normal_y, field.previous_intercept,
        )
        _resize_interface_cache!(cache)
    end
    _refresh_conditional_fraction!(field)
    nx, ny = mesh.block_size
    h = mesh.halo
    _parallel_block_columns(mesh) do b, first_j, last_j
        fraction = @inbounds field.conditional_fraction.blocks[b]
        normal_x = @inbounds field.normal_x.blocks[b]
        normal_y = @inbounds field.normal_y.blocks[b]
        intercept = @inbounds field.intercept.blocks[b]
        @inbounds for j in first_j:last_j, i in 1:nx
            ii, jj = i + h, j + h
            mx, my = _wy_normal(fraction, ii, jj)
            ref = CellRef(b, i, j)
            if !isnothing(field.geometry) &&
                    zero(eltype(field)) < fraction[ii, jj] < one(eltype(field))
                mx, my = _embedded_contact_normal(field, ref, (mx, my))
            end
            normal_x[ii, jj] = mx
            normal_y[ii, jj] = my
            intercept[ii, jj] = isnothing(field.geometry) ?
                _plane_intercept(fraction[ii, jj], mx, my) :
                _open_plane_intercept(
                    field.geometry, ref, fraction[ii, jj], mx, my,
                )
        end
    end
    # Refresh geometrically sampled coarse--fine height columns between passes.
    # Each pass reads a fixed snapshot, making results independent of block order.
    for pass in 1:3
        for (current, previous) in (
                (field.normal_x, field.previous_normal_x),
                (field.normal_y, field.previous_normal_y),
                (field.intercept, field.previous_intercept),
            )
            _parallel_for(nblocks(mesh); grain = 4) do block
                copyto!(previous.blocks[block], current.blocks[block])
            end
        end
        values = _fill_height_values!(field)
        _parallel_block_columns(mesh) do b, first_j, last_j
            height_values = @inbounds values[b]
            for j in first_j:last_j, i in 1:nx
                _reconstruct_interface_cell!(field, height_values, b, i, j)
            end
        end
    end
    return field
end

"""
    initialize!(interface, normal, intercept)

Initialize exact cell averages of the half-plane `dot(normal, (x,y)) <= intercept`.
"""
function initialize!(field::InterfaceField, normal::NTuple{2}, intercept::Real)
    all(isfinite, normal) && any(!iszero, normal) && isfinite(intercept) ||
        throw(ArgumentError("a finite, nonzero plane normal and finite intercept are required"))
    mesh = field.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    _parallel_block_columns(mesh) do b, first_j, last_j
        block = @inbounds mesh.blocks[b]
        values = @inbounds field.fraction.blocks[b]
        dx, dy = block.spacing
        mx, my = normal .* block.spacing
        @inbounds for j in first_j:last_j, i in 1:nx
            x = block.origin[1] + (i - 1 / 2) * dx
            y = block.origin[2] + (j - 1 / 2) * dy
            plane_intercept = float(
                intercept - normal[1] * x - normal[2] * y,
            )
            if isnothing(field.geometry)
                values[i + h, j + h] = _plane_fraction(
                    mx, my, plane_intercept,
                )
            else
                ref = CellRef(b, i, j)
                open_polygon = _open_cell_polygon(field.geometry, ref)
                liquid_polygon = _clip_polygon(
                    open_polygon, mx, my, plane_intercept; keep_less = true,
                )
                values[i + h, j + h] = _polygon_area(liquid_polygon)
            end
        end
    end
    return reconstruct!(field)
end

struct InterfaceProlongation{F, T}
    field::F
    planes::Dict{BlockKey, NTuple{3, Matrix{T}}}
    pool::Vector{NTuple{3, Matrix{T}}}
    staggered::Vector{Any}
    geometry_planes::Dict{BlockKey, NTuple{3, Matrix{T}}}
end
function _finish_prolongation!(p::InterfaceProlongation)
    # All registered fields have already been transferred when finish hooks
    # run. A phase interface must refresh its transferred solid first because
    # open-cell intercepts and GHF stencils read topology-sized geometry
    # caches. This also makes registration order irrelevant.
    isnothing(p.field.geometry) || refresh_geometry!(p.field.geometry)
    reconstruct!(p.field)
    for scalar in p.staggered
        _finish_phase_face_transfer!(scalar)
    end
    return
end
function _prepare_prolongation!(
        p::InterfaceProlongation, field, workspace::AMRWorkspace
    )
    reconstruct!(p.field)
    for q in p.staggered
        _phase_face_average!(
            q.fraction, p.field.conditional_fraction, p.field.geometry,
        )
    end
    for planes in values(p.planes)
        push!(p.pool, planes)
    end
    empty!(p.planes)
    empty!(p.geometry_planes)
    for (b, block) in pairs(field.mesh.blocks)
        if !isnothing(p.field.geometry)
            solid = p.field.geometry.solid
            p.geometry_planes[block.key] = (
                copy(solid.normal_x.blocks[b]),
                copy(solid.normal_y.blocks[b]),
                copy(solid.intercept.blocks[b]),
            )
        end
        haskey(workspace.nodes_by_key, block.key) && continue
        children = _child_keys(block.key)
        all(haskey(workspace.nodes_by_key, child) for child in children) ||
            continue
        planes = if isempty(p.pool)
            (
                similar(p.field.normal_x.blocks[b]),
                similar(p.field.normal_y.blocks[b]),
                similar(p.field.intercept.blocks[b]),
            )
        else
            pop!(p.pool)
        end
        copyto!(planes[1], p.field.normal_x.blocks[b])
        copyto!(planes[2], p.field.normal_y.blocks[b])
        copyto!(planes[3], p.field.intercept.blocks[b])
        p.planes[block.key] = planes
    end
    return
end
function (p::InterfaceProlongation)(child, parent, halo, key::BlockKey)
    normal_x, normal_y, intercept = p.planes[_parent_key(key)]
    nx, ny = size(parent) .- 2halo
    for j in 1:ny,i in 1:nx
        fi, fj = mod(key.i, 2) * nx + i, mod(key.j, 2) * ny + j
        pi, pj = (fi + 1) ÷ 2 + halo, (fj + 1) ÷ 2 + halo
        xlo, ylo = isodd(fi) ? -0.5 : 0.0, isodd(fj) ? -0.5 : 0.0
        if isnothing(p.field.geometry)
            child[i + halo, j + halo] = _plane_fraction(
                normal_x[pi, pj], normal_y[pi, pj],
                intercept[pi, pj], xlo, xlo + 0.5, ylo, ylo + 0.5,
            )
        else
            solid_x, solid_y, solid_intercept =
                p.geometry_planes[_parent_key(key)]
            T = eltype(child)
            rectangle = NTuple{2, T}[
                (xlo, ylo), (xlo + 0.5, ylo),
                (xlo + 0.5, ylo + 0.5), (xlo, ylo + 0.5),
            ]
            open_polygon = _clip_polygon(
                rectangle, solid_x[pi, pj], solid_y[pi, pj],
                solid_intercept[pi, pj]; keep_less = false,
            )
            liquid_polygon = _clip_polygon(
                open_polygon, normal_x[pi, pj], normal_y[pi, pj],
                intercept[pi, pj]; keep_less = true,
            )
            # `liquid_polygon` is expressed in the parent's local coordinates;
            # a child occupies area 1/4 in those coordinates.  Store liquid
            # volume over the complete child cell, bounded by its open volume.
            child[i + halo, j + halo] = clamp(
                4 * _polygon_area(liquid_polygon), zero(T),
                4 * _polygon_area(open_polygon),
            )
        end
    end
    return child
end

function _restrict_parent!(
        p::InterfaceProlongation, parent, children, halo, key,
    )
    isnothing(p.field.geometry) && return restrict!(
        parent, children, halo,
    )
    nx, ny = size(parent) .- 2halo
    for j in 1:ny, i in 1:nx
        liquid = zero(eltype(parent))
        for fj in (2j - 1):2j, fi in (2i - 1):2i
            cx, ii = _fine_location(fi, nx)
            cy, jj = _fine_location(fj, ny)
            child_index = cx + 2(cy - 1)
            liquid += children[child_index][ii + halo, jj + halo]
        end
        parent[i + halo, j + halo] = liquid / 4
    end
    return parent
end
function register_adaptive_field!(workspace::AMRWorkspace, field::InterfaceField{T}) where {T}
    index = _find_field_buffer(workspace, field.fraction)
    if !isnothing(index)
        strategy = workspace.field_buffers[index].prolongation
        strategy isa InterfaceProlongation && return workspace
        if strategy isa GeometryProlongation &&
                strategy.geometry.solid === field
            return workspace
        end
        throw(ArgumentError("register the InterfaceField before its raw fraction field"))
    end
    strategy = InterfaceProlongation(
        field, Dict{BlockKey, NTuple{3, Matrix{T}}}(),
        NTuple{3, Matrix{T}}[], Any[],
        Dict{BlockKey, NTuple{3, Matrix{T}}}(),
    )
    register_adaptive_field!(workspace, field.fraction; prolongation = strategy)
    return workspace
end

"""
    InterfaceRefinementIndicator(interface; level, buffer_cells=1,
                                 fraction_floor=sqrt(eps(T)),
                                 t_init=-Inf, t_end=Inf)

Refine blocks intersecting the interface or its cell-halo neighbourhood to a
fixed level. Pure blocks request coarsening. Aligned interfaces between pure
cells are detected through halo fraction jumps as well as mixed cells. The
criterion is active for `t_init <= time <= t_end`.
"""
struct InterfaceRefinementIndicator{F <: InterfaceField, T} <:
    AbstractRefinementIndicator
    field::F
    level::Int
    buffer_cells::Int
    fraction_floor::T
    t_init::T
    t_end::T
end
function InterfaceRefinementIndicator(
        field::InterfaceField{T}; level,
        buffer_cells = 1, fraction_floor = sqrt(eps(T)),
        t_init = -Inf, t_end = Inf
    ) where {T}
    level isa Integer && !(level isa Bool) &&
        0 <= level <= field.mesh.max_level ||
        throw(ArgumentError("level must be an integer within mesh limits"))
    0 <= buffer_cells <= field.mesh.halo || throw(ArgumentError("buffer_cells must fit the mesh halo"))
    floor = T(fraction_floor)
    isfinite(floor) && zero(T) < floor < T(0.5) ||
        throw(ArgumentError("fraction_floor must be finite and in (0, 0.5)"))
    start_time, end_time = _refinement_window(T, t_init, t_end)
    return InterfaceRefinementIndicator(
        field, Int(level), buffer_cells, floor, start_time, end_time
    )
end
_indicator_field(indicator::InterfaceRefinementIndicator) = indicator.field
function _fixed_interface_refinement_labels!(
        labels::AbstractVector{Symbol}, field::InterfaceField,
        level::Int, buffer_cells::Int, fraction_floor
    )
    mesh = field.mesh
    length(labels) == nblocks(mesh) || throw(DimensionMismatch("one label is required per block"))
    _refresh_conditional_fraction!(field)
    nx, ny = mesh.block_size
    h, r = mesh.halo, buffer_cells
    _parallel_for(nblocks(mesh); grain = 4) do b
        block = mesh.blocks[b]
        values = @view field.conditional_fraction.blocks[b][
            (h + 1 - r):(h + nx + r), (h + 1 - r):(h + ny + r)
        ]
        low, high = extrema(values)
        near = low <= 1 - fraction_floor && high >= fraction_floor
        labels[b] = near ? (
                block.key.level < level ? :refine :
                block.key.level > level ? :coarsen : :keep
            ) : :coarsen
    end
    return labels
end

function refinement_labels!(
        labels::AbstractVector{Symbol},
        indicator::InterfaceRefinementIndicator;
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
        labels, indicator.field, indicator.level,
        indicator.buffer_cells, indicator.fraction_floor
    )
end

_refinement_labels_at_time!(
    labels, indicator::InterfaceRefinementIndicator, time
) = refinement_labels!(labels, indicator; time)

"""
    CurvatureRefinementIndicator(interface; cells_per_radius=8,
                                  coarsen_ratio=1/3, minimum_level=0,
                                  maximum_level=interface.mesh.max_level,
                                  t_init=-Inf, t_end=Inf)

Request refinement where `max(dx,dy)*abs(curvature) > 1/cells_per_radius`.
Following the thesis, a mixed cell also requests refinement when no valid LHF
exists or when its height columns reveal a thin filament. Use hysteresis for
coarsening; combine with `InterfaceRefinementIndicator` for a protected band.
The criterion is active for `t_init <= time <= t_end` and enforces its level
bounds independently of the curvature estimate.
"""
struct CurvatureRefinementIndicator{F <: InterfaceField, T} <: AbstractRefinementIndicator
    field::F
    cells_per_radius::T
    coarsen_ratio::T
    minimum_level::Int
    maximum_level::Int
    t_init::T
    t_end::T
end
function CurvatureRefinementIndicator(
        field::InterfaceField;
        cells_per_radius = 8.0, coarsen_ratio = 1 / 3,
        minimum_level = 0, maximum_level = field.mesh.max_level,
        t_init = -Inf, t_end = Inf
    )
    isfinite(cells_per_radius) && cells_per_radius > 0 || throw(ArgumentError("cells_per_radius must be finite and positive"))
    isfinite(coarsen_ratio) && 0 < 2coarsen_ratio < 1 || throw(ArgumentError("coarsen_ratio must satisfy 0 < 2coarsen_ratio < 1"))
    lower_level, upper_level = _refinement_levels(
        field.mesh, minimum_level, maximum_level
    )
    a, b = promote(float(cells_per_radius), float(coarsen_ratio))
    start_time, end_time = _refinement_window(typeof(a), t_init, t_end)
    return CurvatureRefinementIndicator(
        field, a, b, lower_level, upper_level, start_time, end_time
    )
end

"""
    VorticityRefinementIndicator(velocity; refine_tolerance,
        coarsen_tolerance=refine_tolerance/3, minimum_level=0,
        maximum_level=velocity.mesh.max_level, t_init=-Inf, t_end=Inf)

Block AMR criterion from the thesis. Refine if any cell in a block has
`max(dx,dy)*abs(vorticity) > refine_tolerance`; coarsen when every cell lies
below `coarsen_tolerance`. The strict hysteresis condition
`2coarsen_tolerance < refine_tolerance` prevents immediate re-refinement after
coarsening. The criterion is active for `t_init <= time <= t_end` and enforces
its level bounds independently of the vorticity estimate.
"""
struct VorticityRefinementIndicator{V <: FaceVelocity, T} <:
    AbstractRefinementIndicator
    field::V
    refine_tolerance::T
    coarsen_tolerance::T
    minimum_level::Int
    maximum_level::Int
    t_init::T
    t_end::T
end

function VorticityRefinementIndicator(
        velocity::FaceVelocity{T}; refine_tolerance,
        coarsen_tolerance = refine_tolerance / 3,
        minimum_level = 0, maximum_level = velocity.mesh.max_level,
        t_init = -Inf, t_end = Inf
    ) where {T}
    refine = T(refine_tolerance)
    coarsen = T(coarsen_tolerance)
    isfinite(refine) && refine > zero(T) ||
        throw(ArgumentError("refine_tolerance must be finite and positive"))
    isfinite(coarsen) && zero(T) <= 2coarsen < refine ||
        throw(
        ArgumentError(
            "tolerances must satisfy 0 <= 2coarsen_tolerance < refine_tolerance"
        )
    )
    lower_level, upper_level = _refinement_levels(
        velocity.mesh, minimum_level, maximum_level
    )
    start_time, end_time = _refinement_window(T, t_init, t_end)
    return VorticityRefinementIndicator(
        velocity, refine, coarsen, lower_level, upper_level,
        start_time, end_time
    )
end

_indicator_field(indicator::VorticityRefinementIndicator) = indicator.field

function refinement_labels!(
        labels::AbstractVector{Symbol},
        indicator::VorticityRefinementIndicator;
        time = 0
    )
    mesh = indicator.field.mesh
    length(labels) == nblocks(mesh) ||
        throw(DimensionMismatch("one label is required per block"))
    if !_indicator_active(indicator, time)
        fill!(labels, :coarsen)
        return labels
    end
    vorticity = visualization_values(indicator.field, Vorticity())
    nx, ny = mesh.block_size
    block_cells = nx * ny
    _parallel_for(nblocks(mesh); grain = 4) do b
        block = mesh.blocks[b]
        offset = (b - 1) * block_cells
        error = zero(eltype(vorticity))
        h = maximum(block.spacing)
        for local_index in 1:block_cells
            error = max(
                error,
                h * abs(@inbounds vorticity[offset + local_index])
            )
        end
        level = block.key.level
        labels[b] = if level < indicator.minimum_level
            :refine
        elseif level > indicator.maximum_level
            :coarsen
        elseif error > indicator.refine_tolerance &&
                level < indicator.maximum_level
            :refine
        elseif error < indicator.coarsen_tolerance &&
                _indicator_can_coarsen(level, indicator.minimum_level)
            :coarsen
        else
            :keep
        end
    end
    return labels
end

_refinement_labels_at_time!(
    labels, indicator::VorticityRefinementIndicator, time
) = refinement_labels!(labels, indicator; time)
_indicator_field(indicator::CurvatureRefinementIndicator) = indicator.field
function refinement_labels!(
        labels::AbstractVector{Symbol},
        indicator::CurvatureRefinementIndicator;
        time = 0
    )
    field = indicator.field;mesh = field.mesh
    length(labels) == nblocks(mesh) || throw(DimensionMismatch("one label is required per block"))
    if !_indicator_active(indicator, time)
        fill!(labels, :coarsen)
        return labels
    end
    reconstruct!(field)
    height_values = _fill_height_values!(field)
    nx, ny = mesh.block_size
    _parallel_for(nblocks(mesh); grain = 4) do b
        block = mesh.blocks[b]
        error = 0.0
        for j in 1:ny, i in 1:nx
            c = interior(field, b)[i, j]
            0 < c < 1 || continue
            valid_lhf, filament = _lhf_refinement_status(
                height_values[b], i + 4, j + 4
            )
            kappa = interior(field.curvature, b)[i, j]
            unresolved = !valid_lhf || filament || !isfinite(kappa)
            error = max(
                error,
                unresolved ? Inf : maximum(block.spacing) *
                    abs(kappa) * indicator.cells_per_radius
            )
        end
        level = block.key.level
        labels[b] = if level < indicator.minimum_level
            :refine
        elseif level > indicator.maximum_level
            :coarsen
        elseif error > 1 && level < indicator.maximum_level
            :refine
        elseif error < indicator.coarsen_ratio &&
                _indicator_can_coarsen(level, indicator.minimum_level)
            :coarsen
        else
            :keep
        end
    end
    return labels
end


_refinement_labels_at_time!(
    labels, indicator::CurvatureRefinementIndicator, time
) = refinement_labels!(labels, indicator; time)
