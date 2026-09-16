"""
    InterfaceField(mesh; init=0)
    InterfaceField(fraction::CellField)

Centred liquid volume fractions and derived, cell-local PLIC planes. The liquid
occupies `normal_x*ξ + normal_y*η <= intercept`, with local coordinates in
`[-1/2,1/2]^2`. `reconstruct!` refreshes derived data after modifying fractions.
The field constructor takes ownership of `fraction` (it does not copy it).
"""
struct InterfaceField{T, M <: BlockTreeMesh} <: AbstractMeshField{T}
    mesh::M
    fraction::CellField{T, M}
    normal_x::CellField{T, M}
    normal_y::CellField{T, M}
    intercept::CellField{T, M}
    curvature::CellField{T, M}
    reconstruction::CellField{UInt8, M}
    height_cache::HeightFunctionCache{T}
    contact_angles::NTuple{4, Union{Nothing, T}}
end
placement(::InterfaceField) = Cell()
Base.eltype(::InterfaceField{T}) where {T} = T
interior(field::InterfaceField, b::Integer) = interior(field.fraction, b)
blockdata(field::InterfaceField, b::Integer) = blockdata(field.fraction, b)
integrate(field::InterfaceField) = integrate(field.fraction)
function InterfaceField(fraction::CellField{T, M}; contact_angle = nothing) where {T <: AbstractFloat, M}
    angles = isnothing(contact_angle) ? (nothing, nothing, nothing, nothing) :
        contact_angle isa Real ? ntuple(_ -> T(contact_angle), 4) : contact_angle
    angles isa Tuple && length(angles) == 4 &&
        all(a -> isnothing(a)||(isfinite(a)&&0 < a < pi), angles) ||
        throw(ArgumentError("contact_angle must be in (0,π), or a (left,right,bottom,top) tuple with optional nothing entries"))
    angles = ntuple(i -> isnothing(angles[i]) ? nothing : T(angles[i]), 4)
    field = InterfaceField{T, M}(
        fraction.mesh, fraction,
        CellField(fraction.mesh, T), CellField(fraction.mesh, T), CellField(fraction.mesh, T),
        CellField(fraction.mesh, T; init = NaN), CellField(fraction.mesh, UInt8), HeightFunctionCache(T), angles
    )
    return reconstruct!(field)
end
InterfaceField(mesh::BlockTreeMesh; init = 0, T = eltype(mesh.origin), contact_angle = nothing) =
    InterfaceField(CellField(mesh, T; init); contact_angle)
Base.copy(field::InterfaceField) = InterfaceField(copy(field.fraction); contact_angle = field.contact_angles)

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

function _resize_interface_cache!(field::CellField)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    while length(field.blocks) < nblocks(field.mesh)
        push!(field.blocks, zeros(eltype(field), nx + 2h, ny + 2h))
    end
    return resize!(field.blocks, nblocks(field.mesh))
end

"""Reconstruct PLIC planes with LHF normals, GHF fits and a final WY fallback.
`reconstruction` records 0 (pure), 1 (LHF), 2 (GHF), 3 (WY), or 4
(wall contact-angle reconstruction).
Curvature is NaN where neither a height stencil nor a GHF fit is available.
"""
function reconstruct!(field::InterfaceField)
    mesh = field.mesh
    mesh.halo >= 1 || throw(ArgumentError("interface reconstruction requires one halo cell"))
    for cache in (field.normal_x, field.normal_y, field.intercept, field.curvature, field.reconstruction)
        _resize_interface_cache!(cache)
    end
    for b in eachindex(mesh.blocks), c in interior(field, b)
        isfinite(c) && 0 <= c <= 1 || throw(ArgumentError("volume fractions must be finite and in [0,1]"))
    end
    fill_halo!(field.fraction)
    nx, ny = mesh.block_size
    h = mesh.halo
    for b in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        ref = CellRef(b, i, j)
        mx, my = _wy_normal(field.fraction.blocks[b], i + h, j + h)
        _set!(field.normal_x, mx, ref)
        _set!(field.normal_y, my, ref)
        _set!(field.intercept, _plane_intercept(_get(field.fraction, ref), mx, my), ref)
    end
    # Refresh geometrically sampled coarse--fine height columns between passes.
    # Each pass reads a fixed snapshot, making results independent of block order.
    for pass in 1:3
        values = _fill_height_values!(field)
        for (b, block) in pairs(mesh.blocks),j in 1:ny,i in 1:nx
            ref = CellRef(b, i, j)
            c = _get(field.fraction, ref)
            if c == 0 || c == 1
                _set!(field.reconstruction, 0, ref)
                _set!(field.curvature, NaN, ref)
                continue
            end
            geometry = _contact_geometry(field, values[b], b, i, j)
            method = 4
            if isnothing(geometry)
                geometry = _lhf_geometry(values[b], i + 4, j + 4, block.spacing)
                method = 1
            end
            if isnothing(geometry)
                geometry = _ghf_geometry(field, values[b], b, i, j)
                method = 2
            end
            if isnothing(geometry)
                _set!(field.reconstruction, 3, ref)
                _set!(field.curvature, NaN, ref)
                continue
            end
            gx, gy, kappa = geometry
            mx, my = gx * block.spacing[1], gy * block.spacing[2]
            scale = abs(mx) + abs(my)
            mx, my = mx / scale, my / scale
            _set!(field.normal_x, mx, ref); _set!(field.normal_y, my, ref)
            _set!(field.intercept, _plane_intercept(c, mx, my), ref)
            _set!(field.curvature, kappa, ref)
            _set!(field.reconstruction, method, ref)
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
    for (b, block) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        x, y = cellcenter(mesh, b, i, j)
        mx, my = normal .* block.spacing
        c = _plane_fraction(mx, my, float(intercept - normal[1] * x - normal[2] * y))
        _set!(field.fraction, c, CellRef(b, i, j))
    end
    return reconstruct!(field)
end

struct InterfaceProlongation{F, T}
    field::F
    planes::Dict{BlockKey, NTuple{3, Matrix{T}}}
    staggered::Vector{Any}
end
function _finish_prolongation!(p::InterfaceProlongation)
    reconstruct!(p.field)
    for scalar in p.staggered
        _finish_phase_face_transfer!(scalar)
    end
    return
end
function _prepare_prolongation!(p::InterfaceProlongation, field)
    reconstruct!(p.field)
    for q in p.staggered
        _phase_face_average!(q.fraction, field)
    end
    empty!(p.planes)
    for (b, block) in pairs(field.mesh.blocks)
        p.planes[block.key] = (
            copy(p.field.normal_x.blocks[b]),
            copy(p.field.normal_y.blocks[b]), copy(p.field.intercept.blocks[b]),
        )
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
        child[i + halo, j + halo] = _plane_fraction(
            normal_x[pi, pj], normal_y[pi, pj],
            intercept[pi, pj], xlo, xlo + 0.5, ylo, ylo + 0.5
        )
    end
    return child
end
function register_adaptive_field!(workspace::AMRWorkspace, field::InterfaceField{T}) where {T}
    index = _find_field_buffer(workspace, field.fraction)
    if !isnothing(index)
        workspace.field_buffers[index].prolongation isa InterfaceProlongation && return workspace
        throw(ArgumentError("register the InterfaceField before its raw fraction field"))
    end
    strategy = InterfaceProlongation(field, Dict{BlockKey, NTuple{3, Matrix{T}}}(), Any[])
    register_adaptive_field!(workspace, field.fraction; prolongation = strategy)
    return workspace
end

"""
    InterfaceRefinementIndicator(interface; level, buffer_cells=1,
                                 fraction_floor=sqrt(eps(T)),
                                 t_init=-Inf, t_end=Inf)

Refine blocks intersecting the interface or its cell-halo neighbourhood to a
fixed level. Pure blocks request coarsening. Aligned interfaces between pure
cells are detected through halo fraction jumps as well as mixed cells.
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
function refinement_labels!(
        labels::AbstractVector{Symbol},
        indicator::InterfaceRefinementIndicator;
        time = 0
    )
    field = indicator.field
    mesh = field.mesh
    length(labels) == nblocks(mesh) || throw(DimensionMismatch("one label is required per block"))
    if !_indicator_active(indicator, time)
        fill!(labels, :coarsen)
        return labels
    end
    fill_halo!(field.fraction)
    nx, ny = mesh.block_size
    h, r = mesh.halo, indicator.buffer_cells
    for (b, block) in pairs(mesh.blocks)
        values = @view field.fraction.blocks[b][(h + 1 - r):(h + nx + r), (h + 1 - r):(h + ny + r)]
        low, high = extrema(values)
        floor = indicator.fraction_floor
        near = low <= 1 - floor && high >= floor
        labels[b] = near ? (
                block.key.level < indicator.level ? :refine :
                block.key.level > indicator.level ? :coarsen : :keep
            ) : :coarsen
    end
    return labels
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
coarsening.
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
    for (b, block) in pairs(mesh.blocks)
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
    for (b, block) in pairs(mesh.blocks)
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
