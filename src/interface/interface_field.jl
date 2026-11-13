"""
    InterfaceField(mesh; init=0, fraction_floor=sqrt(eps(T)), ...)
    InterfaceField(fraction::CellField; fraction_floor=sqrt(eps(T)), ...)

Centred liquid volume fractions and derived, cell-local PLIC planes. The liquid
occupies `normal_x*ξ + normal_y*η <= intercept`, with local coordinates in
`[-1/2,1/2]^2`. `reconstruct!` refreshes derived data after modifying fractions.
The field constructor takes ownership of `fraction` (it does not copy it).

When an embedded `geometry` is present, `fraction` is the liquid volume divided
by the complete Cartesian-cell volume.  It therefore obeys
`0 <= fraction <= geometry.open_fraction`.  `conditional_fraction` is the
derived liquid fraction within the open part of the cell and is used by PLIC,
height functions, and the two-phase constitutive laws. The geometry association
is fixed for the lifetime of the field. `copy(field)` copies the fractions and
shares the same geometry object.

`fraction_floor` is the single numerical threshold used by interface cleanup
and compatible scalar recovery.  Conditional liquid fractions no larger than
the floor are set to zero; conditional gas fractions no larger than the floor
are removed by setting the liquid fraction to the available open-cell
fraction.  The default is `sqrt(eps(T))`.

`reconstruction_method=:lhf` uses height-function normals with a WY fallback
for ordinary mixed-cell PLIC planes. `:lhf_ghf` also uses GHF fallback normals;
`:wy` uses WY normals. In embedded triple cells all methods use an
unconstrained solid-aware GHF fit for the PLIC plane when its normal agrees
in orientation with the masked WY seed. Otherwise a nearby fluid LHF normal,
if aligned with that seed, or the masked WY normal is used. Curvature is
estimated with height functions and GHF fits in all modes, so changing the
advection reconstruction does not disable curvature-based refinement or
surface tension.

The default static contact angle is `π/2` at every non-periodic domain wall
and at embedded solid geometry. Both domain-wall and embedded angles constrain
curvature only, not PLIC planes used for advection.
Pass `contact_angle=nothing` to disable the domain-wall curvature condition
and `embedded_contact_angle=nothing` to disable the embedded-wall curvature
condition. A four-tuple orders domain angles as
`(left, right, bottom, top)` and may contain `nothing` entries. For backward
compatibility, a scalar `contact_angle` used together with `geometry` denotes
the embedded contact angle.
"""
struct InterfaceField{T, M <: BlockTreeMesh, G} <: AbstractMeshField{T}
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
    # Exact fraction snapshot and topology keys from the last reconstruction.
    # These make the validity flag safe even when callers edit the public
    # fraction arrays directly, without going through an invalidation hook.
    reconstruction_fraction::CellField{T, M}
    reconstruction_keys::Vector{BlockKey}
    geometry_is_valid::Base.RefValue{Bool}
    reconstructed_geometry_revision::Base.RefValue{UInt}
    # Physical face apertures belong to the reconstructed interface state,
    # not to a particular CFL/projection workspace. Transport copies these
    # into its frozen start-of-step arrays before changing the fractions.
    phase_liquid_aperture::Vector{T}
    phase_gas_aperture::Vector{T}
    phase_apertures_valid::Base.RefValue{Bool}
    phase_aperture_updates::Base.RefValue{UInt}
    # How a face aperture combines the two adjacent PLIC traces: `:average`
    # (default) or `:pure_neighbour`, set by a `MimeticProjectionSolver` on
    # its interface (see `_phase_face_apertures`). `copy` resets it.
    phase_aperture_rule::Base.RefValue{Symbol}
    # Owned flags of the last conditional-fraction refresh, one entry per block
    # column: `(i, value)` of a cell whose fraction left `[0, open]`, or
    # `(0, 0)` when the column is valid. Preallocated so the fused check in the
    # threaded refresh allocates nothing.
    conditional_failure::Vector{Tuple{Int32, T}}
    fraction_floor::T
    reconstruction_method::Symbol
    contact_angles::NTuple{4, Union{Nothing, T}}
    geometry::G
    embedded_contact_angle::Union{Nothing, T}
    # Number of PLIC reconstructions of this field. A geometry's revision adds
    # it (`_interface_geometry_revision`), so a cache that reads the planes
    # sees a re-initialized or reconstructed solid.
    reconstructions::Base.RefValue{UInt}
end
placement(::InterfaceField) = Cell()
Base.eltype(::InterfaceField{T}) where {T} = T
interior(field::InterfaceField, b::Integer) = interior(field.fraction, b)
blockdata(field::InterfaceField, b::Integer) = blockdata(field.fraction, b)
integrate(field::InterfaceField) = integrate(field.fraction)
function InterfaceField(
        fraction::CellField{T, M};
        contact_angle = (pi / 2, pi / 2, pi / 2, pi / 2),
        geometry = nothing, embedded_contact_angle = missing,
        fraction_floor = sqrt(eps(T)), reconstruction_method = :lhf,
    ) where {T <: AbstractFloat, M}
    reconstruction_method in (:lhf, :lhf_ghf, :wy) || throw(
        ArgumentError(
            "reconstruction_method must be :lhf, :lhf_ghf or :wy",
        )
    )
    isnothing(geometry) || (
        hasproperty(geometry, :mesh) && geometry.mesh === fraction.mesh
    ) || throw(ArgumentError("geometry must share the interface mesh"))
    embedded = if isnothing(geometry)
        nothing
    elseif ismissing(embedded_contact_angle)
        contact_angle isa Real ? contact_angle : pi / 2
    else
        embedded_contact_angle
    end
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
    floor = T(fraction_floor)
    isfinite(floor) && zero(T) < floor < T(0.5) || throw(
        ArgumentError("fraction_floor must be finite and in (0, 0.5)"),
    )
    field = InterfaceField{T, M, typeof(geometry)}(
        fraction.mesh, fraction, CellField(fraction.mesh, T),
        CellField(fraction.mesh, T), CellField(fraction.mesh, T), CellField(fraction.mesh, T),
        CellField(fraction.mesh, T; init = NaN), CellField(fraction.mesh, UInt8),
        CellField(fraction.mesh, T), CellField(fraction.mesh, T),
        CellField(fraction.mesh, T), HeightFunctionCache(T),
        CellField(fraction.mesh, T), BlockKey[], Ref(false), Ref(UInt(0)),
        T[], T[], Ref(false), Ref(UInt(0)), Ref(:average),
        Tuple{Int32, T}[],
        floor,
        reconstruction_method, angles,
        geometry, isnothing(embedded) ? nothing : T(embedded),
        Ref(UInt(0)),
    )
    return reconstruct!(field)
end
InterfaceField(
    mesh::BlockTreeMesh; init = 0, T = eltype(mesh.origin),
    contact_angle = (pi / 2, pi / 2, pi / 2, pi / 2), geometry = nothing,
    embedded_contact_angle = missing, fraction_floor = sqrt(eps(T)),
    reconstruction_method = :lhf,
) = begin
    values = CellField(mesh, T; init = isnothing(geometry) ? init : zero(T))
    if !isnothing(geometry)
        init isa Number || throw(
            ArgumentError(
                "geometry-aware InterfaceField init must be a scalar",
            )
        )
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
        values; contact_angle, geometry, embedded_contact_angle, fraction_floor,
        reconstruction_method,
    )
end
Base.copy(field::InterfaceField) = InterfaceField(
    copy(field.fraction); contact_angle = field.contact_angles,
    geometry = field.geometry,
    embedded_contact_angle = field.embedded_contact_angle,
    fraction_floor = field.fraction_floor,
    reconstruction_method = field.reconstruction_method,
)

"""
    interface_height(interface; direction=:y, position=0)

Return the physical coordinate of a single-valued interface at `position`.
`direction` selects the height coordinate (`:x` or `:y`); `position` is the
coordinate in the transverse direction. The height function integrates the
reconstructed 0/1 phase indicator along an adaptive-mesh column and infers
whether liquid lies above or below the interface.

The sampled column must connect opposite pure phases and must not cross
embedded solid geometry. Periodic transverse positions are wrapped into the
domain.
"""
function interface_height(
        field::InterfaceField; direction = :y, position = 0,
    )
    axis = direction === :x ? 1 : direction === :y ? 2 : throw(
            ArgumentError("direction must be :x or :y"),
        )
    return _interface_height(field, axis, position)
end

@inline _conditional_fraction(field::InterfaceField, ref::CellRef) =
    _get(field.conditional_fraction, ref)

"""Throw the bound violation recorded by a conditional-fraction refresh.

The refresh records the first offending cell of each block column in an owned,
preallocated flag and never throws; this runs on the calling task after the
launch, so the exception is the package's own for every thread count.
"""
function _throw_conditional_fraction_failure(
        field::InterfaceField{T}, failures,
    ) where {T}
    index = 0
    for candidate in eachindex(failures)
        if !iszero(@inbounds failures[candidate][1])
            index = candidate
            break
        end
    end
    iszero(index) && return nothing
    ny = field.mesh.block_size[2]
    block = (index - 1) ÷ ny + 1
    j = index - (block - 1) * ny
    flag, value = @inbounds failures[index]
    i = Int(flag)
    h = field.mesh.halo
    open = isnothing(field.geometry) ? nothing :
        @inbounds field.geometry.open_fraction.blocks[block]
    available = isnothing(open) ? one(T) : @inbounds open[i + h, j + h]
    throw(
        ArgumentError(
            "liquid volume fraction $value lies outside [0,$available] " *
                "in cell $(CellRef(block, i, j))",
        ),
    )
end

"""Clamp `fraction` into `[0, open]` and refresh the conditional fraction.

The bounds check is fused into the threaded refresh: it records the offending
cell in an owned, preallocated flag and the throw follows the launch, so the
error is the same for every thread count. Because the check shares its pass
with the clamp, a rejected fraction is not restored: cells visited before the
offending one are already clamped.
"""
function _refresh_conditional_fraction!(field::InterfaceField{T}) where {T}
    geometry = field.geometry
    mesh = field.mesh
    nx, ny = mesh.block_size
    h = mesh.halo
    _resize_interface_cache!(field.conditional_fraction)
    failures = field.conditional_failure
    length(failures) == nblocks(mesh) * ny || resize!(failures, nblocks(mesh) * ny)
    fill!(failures, (Int32(0), zero(T)))
    _parallel_block_columns(mesh) do block, first_j, last_j
        physical = @inbounds field.fraction.blocks[block]
        conditional = @inbounds field.conditional_fraction.blocks[block]
        open = isnothing(geometry) ? nothing :
            @inbounds geometry.open_fraction.blocks[block]
        column_failures = field.conditional_failure
        # Read through the captured field rather than capturing the mesh, so
        # this launch keeps its capture tuple (and allocation) unchanged.
        columns = field.mesh.block_size[2]
        @inbounds for j in first_j:last_j, i in 1:nx
            ii, jj = i + h, j + h
            available = isnothing(open) ? one(T) : open[ii, jj]
            raw = physical[ii, jj]
            tolerance = T(128) * eps(T) * max(one(T), available)
            isfinite(raw) && raw >= -tolerance && raw <= available + tolerance ||
                (column_failures[(block - 1) * columns + j] = (Int32(i), raw))
            value = clamp(raw, zero(T), available)
            ratio = available > zero(T) ? value / available : zero(T)
            if ratio <= field.fraction_floor
                value = zero(T)
                ratio = zero(T)
            elseif ratio >= one(T) - field.fraction_floor
                value = available
                ratio = one(T)
            end
            physical[ii, jj] = value
            conditional[ii, jj] = ratio
        end
    end
    _throw_conditional_fraction_failure(field, failures)
    fill_halo!(field.fraction)
    fill_halo!(field.conditional_fraction)
    isnothing(geometry) || _restrict_conditional_halo!(field, geometry)
    return field.conditional_fraction
end

# A coarse block next to finer blocks receives its halo values by restriction.
# The conditional fraction is a ratio of the liquid fraction to the open
# fraction, so its halo must restrict numerator and denominator with the same
# weights: averaging the per-cell ratios instead counts covered cells, whose
# ratio is zero by construction, as gas and makes every nearly covered halo
# cell look mixed (#148).
function _restrict_conditional_halo!(field::InterfaceField{T}, geometry) where {T}
    mesh = field.mesh
    floor = field.fraction_floor
    fraction = field.fraction
    open_fraction = geometry.open_fraction
    conditional = field.conditional_fraction
    @_parallel for entry_index in 1:length(mesh.halo_entries)
        entry = @inbounds mesh.halo_entries[entry_index]
        entry.physical_boundary && continue
        # A single same-level or coarser donor already carries the ratio
        # itself; only a restriction from several finer donors can differ.
        entry.donor_count > 1 || continue
        liquid = zero(T)
        available = zero(T)
        stop = entry.first_donor + entry.donor_count - 1
        for index in entry.first_donor:stop
            weight = @inbounds mesh.halo_weights[index]
            donor = @inbounds mesh.halo_donors[index]
            liquid += weight * _get(fraction, donor)
            available += weight * _get(open_fraction, donor)
        end
        ratio = available > zero(T) ?
            clamp(liquid, zero(T), available) / available : zero(T)
        ratio <= floor && (ratio = zero(T))
        ratio >= one(T) - floor && (ratio = one(T))
        _set!(conditional, ratio, entry.destination)
    end
    return conditional
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

function _resize_interface_cache!(field::CellField)
    nx, ny = field.mesh.block_size
    h = field.mesh.halo
    while length(field.blocks) < nblocks(field.mesh)
        push!(field.blocks, zeros(eltype(field), nx + 2h, ny + 2h))
    end
    return resize!(field.blocks, nblocks(field.mesh))
end

"""Version of a geometry's reconstructed state for every revision-keyed cache.

`geometry.revision` counts `refresh_geometry!` calls (regrids); the fixed
solid's own reconstruction count adds every rebuild of its PLIC planes, so a
re-initialized or reconstructed solid is a new version even without a refresh.
Both counters only grow, so the sum grows with either of them.
"""
@inline _interface_geometry_revision(geometry) =
    isnothing(geometry) ? UInt(0) :
    UInt(geometry.revision) + geometry.solid.reconstructions[]

"""Whether stored PLIC data still describe the exact current input state."""
function _reconstruction_is_current(field::InterfaceField)
    field.geometry_is_valid[] || return false
    mesh = field.mesh
    length(field.reconstruction_keys) == nblocks(mesh) || return false
    length(field.reconstruction_fraction.blocks) == nblocks(mesh) ||
        return false
    field.reconstructed_geometry_revision[] ==
        _interface_geometry_revision(field.geometry) || return false
    nx, ny = mesh.block_size
    h = mesh.halo
    for block in eachindex(mesh.blocks)
        field.reconstruction_keys[block] == mesh.blocks[block].key ||
            return false
        physical = @inbounds field.fraction.blocks[block]
        snapshot = @inbounds field.reconstruction_fraction.blocks[block]
        @inbounds for j in 1:ny, i in 1:nx
            physical[i + h, j + h] == snapshot[i + h, j + h] ||
                return false
        end
    end
    return true
end

function _mark_reconstruction_current!(field::InterfaceField)
    mesh = field.mesh
    _resize_interface_cache!(field.reconstruction_fraction)
    resize!(field.reconstruction_keys, nblocks(mesh))
    for block in eachindex(mesh.blocks)
        @inbounds field.reconstruction_keys[block] = mesh.blocks[block].key
        copyto!(
            field.reconstruction_fraction.blocks[block],
            field.fraction.blocks[block],
        )
    end
    field.reconstructed_geometry_revision[] =
        _interface_geometry_revision(field.geometry)
    field.geometry_is_valid[] = true
    field.reconstructions[] += UInt(1)
    return field
end

@inline function _clip_curvature_to_mesh(kappa, spacing)
    h = maximum(spacing)
    limit = inv(2h)
    return clamp(kappa, -limit, limit)
end

# An LHF in a nearby fully open mixed cell supplies an angle-independent
# direction when the target GHF plane is unavailable or opposes its WY seed.
# The nearest interface centroid wins; iteration order breaks distance ties.
function _nearby_lhf_normal(field, values, b, i, j, ref, c, seed_x, seed_y)
    hx, hy = field.mesh.blocks[b].spacing
    sx, sy = seed_x / hx, seed_y / hy
    seed_norm = hypot(sx, sy)
    minimum_alignment = sqrt(oftype(c, 0.5)) * seed_norm
    seed_intercept = _open_plane_intercept(field.geometry, ref, c, seed_x, seed_y)
    target_x, target_y = _interface_centroid(seed_x, seed_y, seed_intercept)
    best = nothing
    best_distance = oftype(c, Inf)
    for dj in -2:2, di in -2:2
        iszero(di) && iszero(dj) && continue
        ii, jj = i + 4 + di, j + 4 + dj
        checkbounds(Bool, values, ii, jj) || continue
        fraction = values[ii, jj]
        # Height values exclude cut/covered cells and invalid halo donors.
        isfinite(fraction) && zero(fraction) < fraction < one(fraction) || continue
        fit = _lhf_geometry(values, ii, jj, (hx, hy))
        isnothing(fit) && continue
        scale = hypot(fit[1], fit[2])
        scale > 0 || continue
        nx, ny = fit[1] / scale, fit[2] / scale
        nx * sx + ny * sy >= minimum_alignment || continue
        mx, my = nx * hx, ny * hy
        intercept = _plane_intercept(fraction, mx, my)
        cx, cy = _interface_centroid(mx, my, intercept)
        distance = ((di + cx - target_x) * hx)^2 +
            ((dj + cy - target_y) * hy)^2
        distance < best_distance || continue
        best = (nx, ny)
        best_distance = distance
    end
    return best
end

function _reconstruct_embedded_triple_cell!(field, values, b, i, j, ref, c)
    block = field.mesh.blocks[b]
    # A wider local search supplies fluid height samples when the solid
    # removes the nearest columns; non-triple stencils remain unchanged.
    plane_fit = _ghf_geometry(
        field, values, b, i, j,
        field.previous_normal_x, field.previous_normal_y,
        field.previous_intercept;
        enforce_contact = false,
        sample_radius = 3,
    )
    h = field.mesh.halo
    fraction = field.fraction.blocks[b]
    open = field.geometry.open_fraction.blocks[b]
    seed_x, seed_y = _solid_aware_wy_normal(fraction, open, i + h, j + h)
    sx, sy = seed_x / block.spacing[1], seed_y / block.spacing[2]
    # Keep the fixed masked-WY orientation when an under-resolved GHF fit
    # points into its opposite hemisphere. Compare physical normals so the
    # check is independent of cell aspect ratio.
    accepted = !isnothing(plane_fit) &&
        plane_fit[1] * sx + plane_fit[2] * sy > 0
    if accepted
        mx, my = plane_fit[1] * block.spacing[1], plane_fit[2] * block.spacing[2]
        scale = abs(mx) + abs(my)
        mx, my = mx / scale, my / scale
        method = 2
    else
        neighbor = _nearby_lhf_normal(
            field, values, b, i, j, ref, c, seed_x, seed_y,
        )
        if isnothing(neighbor)
            mx, my = seed_x, seed_y
            method = 3
        else
            mx, my = neighbor[1] * block.spacing[1], neighbor[2] * block.spacing[2]
            scale = abs(mx) + abs(my)
            mx, my = mx / scale, my / scale
            method = 1
        end
    end
    # The contact angle is a curvature boundary condition, not an advection
    # plane constraint. Both fits use the same seed and previous-pass planes.
    raw_curvature = isnothing(plane_fit) ? eltype(field)(NaN) : plane_fit[3]
    if !isnothing(field.embedded_contact_angle)
        curvature_fit = _ghf_geometry(
            field, values, b, i, j,
            field.previous_normal_x, field.previous_normal_y,
            field.previous_intercept;
            enforce_contact = true,
            sample_radius = 3,
        )
        !isnothing(curvature_fit) && (raw_curvature = curvature_fit[3])
    end
    _set!(field.normal_x, mx, ref)
    _set!(field.normal_y, my, ref)
    _set!(field.intercept, _open_plane_intercept(field.geometry, ref, c, mx, my), ref)
    contact_curvature = _contact_curvature(field, values, b, i, j)
    !isnothing(contact_curvature) && (raw_curvature = contact_curvature)
    _set!(field.curvature, _clip_curvature_to_mesh(raw_curvature, block.spacing), ref)
    _set!(field.reconstruction, method, ref)
    return nothing
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
    embedded_triple = !isnothing(field.geometry) &&
        zero(c) < _get(field.geometry.solid.fraction, ref) < one(c)
    embedded_triple && return _reconstruct_embedded_triple_cell!(
        field, values, b, i, j, ref, c,
    )
    geometry = _one_sided_wall_normal(field, values, b, i, j)
    method = 5
    if isnothing(geometry)
        geometry = _lhf_geometry(values, i + 4, j + 4, block.spacing)
        method = 1
    end
    if isnothing(geometry)
        geometry = _ghf_geometry(
            field, values, b, i, j,
            field.previous_normal_x, field.previous_normal_y,
            field.previous_intercept; enforce_contact = false,
        )
        method = 2
    end
    if isnothing(geometry)
        method = 3
        gx = _get(field.normal_x, ref) / block.spacing[1]
        gy = _get(field.normal_y, ref) / block.spacing[2]
        raw_curvature = eltype(field)(NaN)
    else
        gx, gy = geometry[1], geometry[2]
        raw_curvature = length(geometry) == 3 ?
            geometry[3] : eltype(field)(NaN)
    end
    # A GHF fit can supply curvature even when its normal is not selected for
    # advection. In :lhf the initial WY plane remains on those cells.
    use_fitted_plane = field.reconstruction_method == :lhf_ghf ||
        (field.reconstruction_method == :lhf && method in (1, 5))
    if use_fitted_plane
        mx, my = gx * block.spacing[1], gy * block.spacing[2]
        scale = abs(mx) + abs(my)
        mx, my = mx / scale, my / scale
        _set!(field.normal_x, mx, ref)
        _set!(field.normal_y, my, ref)
        intercept = isnothing(field.geometry) ? _plane_intercept(c, mx, my) :
            _open_plane_intercept(field.geometry, ref, c, mx, my)
        _set!(field.intercept, intercept, ref)
    end

    # For domain walls, the contact angle constrains curvature only.
    contact_curvature = _contact_curvature(field, values, b, i, j)
    if !isnothing(contact_curvature)
        raw_curvature = contact_curvature
    elseif !isfinite(raw_curvature) && method == 5
        unconstrained = _ghf_geometry(
            field, values, b, i, j,
            field.previous_normal_x, field.previous_normal_y,
            field.previous_intercept; enforce_contact = false,
        )
        !isnothing(unconstrained) && (raw_curvature = unconstrained[3])
    end
    kappa = _clip_curvature_to_mesh(raw_curvature, block.spacing)
    _set!(field.curvature, kappa, ref)
    _set!(field.reconstruction, use_fitted_plane ? method : 3, ref)
    return nothing
end

"""Reconstruct PLIC planes with the selected `reconstruction_method`.
`reconstruction` records 0 (pure), 1 (centred or nearby LHF), 2 (GHF),
3 (WY), or 5 (one-sided two-point wall LHF). Domain-wall and embedded contact
angles constrain curvature only, not the PLIC planes used for advection.
Curvature is NaN where neither a height stencil nor a GHF fit is available.
The exact fraction and topology snapshot make a repeated call a no-op. Use
`force=true` after intentionally editing derived PLIC fields in place.
"""
function reconstruct!(field::InterfaceField; force = false)
    mesh = field.mesh
    mesh.halo >= 1 || throw(ArgumentError("interface reconstruction requires one halo cell"))
    if !force && _reconstruction_is_current(field)
        return field
    end
    return _reconstruct_interface!(field)
end

function _reconstruct_interface!(field::InterfaceField)
    mesh = field.mesh
    field.geometry_is_valid[] = false
    field.phase_apertures_valid[] = false
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
        physical = @inbounds field.fraction.blocks[b]
        open = isnothing(field.geometry) ? nothing :
            @inbounds field.geometry.open_fraction.blocks[b]
        normal_x = @inbounds field.normal_x.blocks[b]
        normal_y = @inbounds field.normal_y.blocks[b]
        intercept = @inbounds field.intercept.blocks[b]
        @inbounds for j in first_j:last_j, i in 1:nx
            ii, jj = i + h, j + h
            ref = CellRef(b, i, j)
            triple = !isnothing(field.geometry) &&
                zero(fraction[ii, jj]) < fraction[ii, jj] < one(fraction[ii, jj]) &&
                zero(fraction[ii, jj]) <
                _get(field.geometry.solid.fraction, ref) < one(fraction[ii, jj])
            mx, my = triple ?
                _solid_aware_wy_normal(physical, open, ii, jj) :
                _wy_normal(fraction, ii, jj)
            normal_x[ii, jj] = mx
            normal_y[ii, jj] = my
            intercept[ii, jj] = isnothing(field.geometry) ?
                _plane_intercept(fraction[ii, jj], mx, my) :
                _open_plane_intercept(
                    field.geometry, ref, fraction[ii, jj], mx, my,
                )
        end
    end
    # Triple-cell GHF planes need the same fixed-snapshot iterations as LHF.
    # WY mode updates ordinary cells only on the first pass, as without solids.
    passes = field.reconstruction_method == :wy && isnothing(field.geometry) ? 1 : 3
    for pass in 1:passes
        for (current, previous) in (
                (field.normal_x, field.previous_normal_x),
                (field.normal_y, field.previous_normal_y),
                (field.intercept, field.previous_intercept),
            )
            _parallel_for(nblocks(mesh); work = _block_work()) do block
                copyto!(previous.blocks[block], current.blocks[block])
            end
        end
        values = _fill_height_values!(field)
        triple_only = pass > 1 && field.reconstruction_method == :wy
        _parallel_block_columns(mesh) do b, first_j, last_j
            height_values = @inbounds values[b]
            for j in first_j:last_j, i in 1:nx
                if triple_only
                    ref = CellRef(b, i, j)
                    c = _conditional_fraction(field, ref)
                    solid = _get(field.geometry.solid.fraction, ref)
                    (zero(c) < c < one(c) && zero(c) < solid < one(c)) ||
                        continue
                end
                _reconstruct_interface_cell!(field, height_values, b, i, j)
            end
        end
    end
    _mark_reconstruction_current!(field)
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

struct InterfaceProlongation{F, T, S}
    field::F
    planes::Dict{BlockKey, NTuple{3, Matrix{T}}}
    pool::Vector{NTuple{3, Matrix{T}}}
    staggered::Vector{S}
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
        NTuple{3, Matrix{T}}[], _interface_staggered_scalars(field),
        Dict{BlockKey, NTuple{3, Matrix{T}}}(),
    )
    register_adaptive_field!(workspace, field.fraction; prolongation = strategy)
    return workspace
end
