"""Interface for discrete surface-tension force formulations."""
abstract type AbstractSurfaceTensionMethod end

"""Continuum surface tension with the same weighted gradient as pressure."""
struct CSF <: AbstractSurfaceTensionMethod end
"""Ghost-fluid surface tension with a sharp indicator jump.
`GFM()` uses the thesis's conservative C-GFM distance (staggered fraction).
`GFM(conservative=false)` uses geometric centre-to-centre phase distances;
as discussed in the thesis, this variant does not conserve linear momentum.
"""
struct GFM <: AbstractSurfaceTensionMethod
    conservative::Bool
end
GFM(; conservative = true) = GFM(conservative)

"""Select the conservative ghost-fluid surface-tension formulation."""
CGFM() = GFM(conservative = true)

"""Staggered mixture density from the mass-compatible cell-to-face interpolant."""
function mixture_density(field::InterfaceField{T}, density::NTuple{2}) where {T}
    result = Vector{T}(undef, length(field.mesh.faces))
    return _mixture_density!(result, field, density)
end

@inline function _staggered_liquid_fraction(
        field::InterfaceField{T}, ::Nothing, face,
    ) where {T}
    hm = field.mesh.blocks[face.minus.block].spacing[face.axis]
    hp = field.mesh.blocks[face.plus.block].spacing[face.axis]
    return clamp(
        (
            hm * _conditional_fraction(field, face.minus) +
                hp * _conditional_fraction(field, face.plus)
        ) / (hm + hp),
        zero(T), one(T),
    )
end

"""Liquid fraction in one open staggered dual volume.

The cut-cell case is the allocation-free scalar form of `_phase_face_average!`:
each half of the dual volume is weighted by its geometrically open subcell
area. Keeping this interpolation in one helper makes the one- and two-velocity
diffusion inertia identical.
"""
@inline function _staggered_liquid_fraction(
        field::InterfaceField{T}, geometry::GeometryField, face,
    ) where {T}
    mesh = field.mesh
    axis = face.axis
    tangent = 3 - axis
    weighted = volume = zero(T)
    for ref in (face.minus, face.plus)
        center = cellcenter(mesh, ref.block, ref.i, ref.j)
        spacing = mesh.blocks[ref.block].spacing[tangent]
        lo = _wrapped_delta(
            face.center[tangent] - face.area / 2 - center[tangent],
            mesh.extent[tangent], mesh.periodic[tangent],
        ) / spacing
        hi = lo + face.area / spacing
        bounds = axis == 1 ? (-T(0.5), T(0.5), lo, hi) :
            (lo, hi, -T(0.5), T(0.5))
        contribution = T(0.5) * _open_subcell_area(
            geometry.solid, ref, bounds...,
        ) * mesh.blocks[ref.block].volume
        weighted += contribution * _conditional_fraction(field, ref)
        volume += contribution
    end
    return volume > zero(T) ?
        clamp(weighted / volume, zero(T), one(T)) : zero(T)
end

function _mixture_density!(
        result::AbstractVector{T}, field::InterfaceField{T}, density::NTuple{2}
    ) where {T}
    return _mixture_density!(
        result, field, density, field.geometry,
    )
end

function _mixture_density!(
        result::AbstractVector{T}, field::InterfaceField{T},
        density::NTuple{2}, geometry,
    ) where {T}
    all(rho -> isfinite(rho) && rho > 0, density) || throw(ArgumentError("phase densities must be finite and positive"))
    mesh = field.mesh
    length(result) == length(mesh.faces) || throw(
        DimensionMismatch("one density value is required per interior face"),
    )
    @_parallel for i in 1:length(mesh.faces)
        face = mesh.faces[i]
        c = _staggered_liquid_fraction(field, geometry, face)
        result[i] = density[1] * c + density[2] * (1 - c)
    end
    return result
end

function mixture_density(
        field::InterfaceField{T}, density::NTuple{2},
        geometry::GeometryField,
    ) where {T}
    field.mesh === geometry.mesh || throw(
        ArgumentError("interface and geometry must share a mesh"),
    )
    result = Vector{T}(undef, length(field.mesh.faces))
    for (index, face) in pairs(field.mesh.faces)
        fraction = _staggered_liquid_fraction(field, geometry, face)
        @inbounds result[index] = density[1] * fraction +
            density[2] * (one(T) - fraction)
    end
    return result
end
"""Construct the pressure operator using the inverse staggered mixture density."""
function PoissonOperator(
        field::InterfaceField; density = (1.0, 1.0),
        formulation = CGFM(), template = nothing,
        construction_cache = GradientConstructionCache(eltype(field)),
        assemble_matrix = true,
    )
    rho = mixture_density(field, density)
    if formulation isa GFM && !formulation.conservative
        @_parallel for i in 1:length(field.mesh.faces)
            face = field.mesh.faces[i]
            fraction = _gfm_distance_fraction(field, face)
            rho[i] = density[1] * fraction + density[2] * (1 - fraction)
        end
    end
    coefficient = inv.(rho)
    if isnothing(template)
        return PoissonOperator(
            field.mesh; face_coefficient = coefficient, construction_cache,
            assemble_matrix,
            T = eltype(field)
        )
    end
    template isa EllipticOperator && template.mesh === field.mesh ||
        throw(
        ArgumentError(
            "template must be an EllipticOperator on the interface mesh"
        )
    )
    iszero(template.reaction) ||
        throw(ArgumentError("a pressure template must have zero reaction"))
    return _replace_face_coefficient(template, coefficient)
end

"""Two-phase pressure operator on a fixed embedded cut-cell geometry."""
function PoissonOperator(
        field::InterfaceField{T}, geometry::GeometryField{T};
        density = (1.0, 1.0), formulation = CGFM(), template = nothing,
        construction_cache = GradientConstructionCache(T),
        assemble_matrix = true,
    ) where {T}
    field.mesh === geometry.mesh || throw(
        ArgumentError("interface and geometry must share a mesh"),
    )
    rho = mixture_density(field, density, geometry)
    if formulation isa GFM && !formulation.conservative
        @_parallel for index in eachindex(field.mesh.faces)
            face = field.mesh.faces[index]
            fraction = _gfm_distance_fraction(field, face)
            @inbounds rho[index] = density[1] * fraction +
                density[2] * (1 - fraction)
        end
    end
    coefficient = geometry.aperture .^ 2 ./ rho
    if isnothing(template)
        return EllipticOperator(
            field.mesh; reaction = 0, diffusion = 1,
            face_coefficient = coefficient,
            gradient_operator = GradientOperator(
                geometry; construction_cache,
            ),
            volumes = _geometry_cell_volumes(geometry),
            active_cells = geometry.active_cells,
            construction_cache, assemble_matrix, T,
        )
    end
    template isa EllipticOperator && template.mesh === field.mesh || throw(
        ArgumentError("template must be an EllipticOperator on the interface mesh"),
    )
    return _replace_face_coefficient(template, coefficient)
end

function _line_phase_fraction(field, ref, p0, p1)
    c = _conditional_fraction(field, ref)
    (c == 0||c == 1) && return c
    block = field.mesh.blocks[ref.block]
    center = cellcenter(field.mesh, ref.block, ref.i, ref.j)
    nx, ny = _get(field.normal_x, ref), _get(field.normal_y, ref)
    a = _get(field.intercept, ref)
    value(p) = nx * _wrapped_delta(p[1] - center[1], field.mesh.extent[1], field.mesh.periodic[1]) / block.spacing[1] +
        ny * _wrapped_delta(p[2] - center[2], field.mesh.extent[2], field.mesh.periodic[2]) / block.spacing[2] - a
    d0, d1 = value(p0), value(p1)
    d0 <= 0 && d1 <= 0 && return one(c)
    d0 >= 0 && d1 >= 0 && return zero(c)
    crossing = d0 / (d0 - d1)
    return d0 < 0 ? crossing : 1 - crossing
end
function _gfm_distance_fraction(field, face)
    mesh = field.mesh
    p0 = cellcenter(mesh, face.minus.block, face.minus.i, face.minus.j)
    p1 = cellcenter(mesh, face.plus.block, face.plus.i, face.plus.j)
    delta = ntuple(a -> _wrapped_delta(p1[a] - p0[a], mesh.extent[a], mesh.periodic[a]), 2)
    hm = mesh.blocks[face.minus.block].spacing[face.axis]
    hp = mesh.blocks[face.plus.block].spacing[face.axis]
    theta = hm / (hm + hp)
    crossing = ntuple(a -> p0[a] + theta * delta[a], 2)
    return theta * _line_phase_fraction(field, face.minus, p0, crossing) +
        (1 - theta) * _line_phase_fraction(field, face.plus, crossing, p1)
end

function _weighted_jump_force!(
        destination, operator, work, jumps, geometry = nothing,
    )
    if !isnothing(geometry)
        square_root = operator.square_root_face_coefficient
        @_parallel for i in eachindex(jumps)
            @inbounds work.face_input[i] = square_root[i] * jumps[i]
        end
        _mul_inverse_metric!(
            work.face_output,
            operator.gradient_operator.inverse_metric, work.face_input,
        )
        @_parallel for i in eachindex(jumps)
            aperture = @inbounds geometry.aperture[i]
            @inbounds destination.interior[i] = aperture > 0 ?
                square_root[i] / aperture *
                work.face_output[i] : zero(eltype(destination))
        end
        fill!(destination.boundary, zero(eltype(destination)))
        return destination
    end
    @_parallel for i in 1:length(jumps)
        work.face_input[i] = operator.square_root_face_coefficient[i] * jumps[i]
    end
    _mul_inverse_metric!(work.face_output, operator.gradient_operator.inverse_metric, work.face_input)
    @_parallel for i in 1:length(jumps)
        destination.interior[i] = operator.square_root_face_coefficient[i] * work.face_output[i]
    end
    fill!(destination.boundary, zero(eltype(destination)))
    return destination
end
function _curvature_samples(field)
    samples = Tuple{
        NTuple{2, eltype(field)}, eltype(field), eltype(field), eltype(field),
    }[]
    for ref in _cell_references(field.mesh)
        c = _conditional_fraction(field, ref)
        kappa = _get(field.curvature, ref)
        weight = c * (1 - c)
        weight > 0 && isfinite(kappa) || continue
        point = cellcenter(field.mesh, ref.block, ref.i, ref.j)
        spacing = maximum(field.mesh.blocks[ref.block].spacing)
        push!(samples, (point, kappa, weight, spacing))
    end
    return samples
end

function _face_curvature(field, face, samples)
    total = weight = 0.0
    for ref in (face.minus, face.plus)
        c = _conditional_fraction(field, ref);k = _get(field.curvature, ref)
        w = c * (1 - c)
        if w > 0 && isfinite(k)
            total += w * k;weight += w
        end
    end
    weight > 0 && return total / weight

    # A cut-cell conditional fraction can cross an interface on a face while
    # both cell averages are pure. Extend the same c(1-c)-weighted GHF values
    # used by the thesis face interpolant from the local stencil. The radius
    # scales with the larger adjacent cell, including at an AMR transition.
    local_spacing = max(
        maximum(field.mesh.blocks[face.minus.block].spacing),
        maximum(field.mesh.blocks[face.plus.block].spacing),
    )
    radius = 3local_spacing
    for (point, kappa, sample_weight, sample_spacing) in samples
        dx = _wrapped_delta(
            point[1] - face.center[1], field.mesh.extent[1],
            field.mesh.periodic[1],
        )
        dy = _wrapped_delta(
            point[2] - face.center[2], field.mesh.extent[2],
            field.mesh.periodic[2],
        )
        distance = hypot(dx, dy)
        distance <= radius + sample_spacing || continue
        geometric_weight = sample_weight /
            max(distance^2, (sample_spacing / 4)^2)
        total += geometric_weight * kappa
        weight += geometric_weight
    end
    return weight > 0 ? total / weight : nothing
end

"""
    surface_tension!(acceleration, interface, pressure_operator, workspace;
                     sigma, formulation=CGFM(), curvature=nothing)

Overwrite acceleration with CSF or conservative GFM surface tension. The
optional constant `curvature` is for isolating the balanced-force identity in
tests. Otherwise use fraction-weighted LHF/GHF curvature. Missing curvature on
an active interface face raises an error. The same weighted inverse metric as
pressure is applied to the curvature-weighted phase jump, including on AMR.
"""
function surface_tension!(
        destination::FaceVelocity, field::InterfaceField,
        operator::EllipticOperator, work::EllipticWorkspace;
        sigma, formulation = CGFM(), curvature = nothing
    )
    destination.mesh === field.mesh === operator.mesh || throw(ArgumentError("surface tension arguments must share a mesh"))
    formulation isa Union{CSF, GFM} || throw(ArgumentError("formulation must be CSF() or GFM()"))
    isfinite(sigma) && sigma >= 0 || throw(ArgumentError("sigma must be finite and nonnegative"))
    isnothing(curvature) || isfinite(curvature) || throw(ArgumentError("constant curvature must be finite"))
    jumps = operator.gradient_operator.pressure_jump
    curvature_samples = isnothing(curvature) ?
        _curvature_samples(field) : nothing
    for (i, face) in pairs(field.mesh.faces)
        if !isnothing(field.geometry) &&
                iszero(field.geometry.aperture[i])
            jumps[i] = 0
            continue
        end
        cm, cp = _conditional_fraction(field, face.minus),
            _conditional_fraction(field, face.plus)
        jump = formulation isa CSF ? cp - cm : (cp > 0.5) - (cm > 0.5)
        if sigma == 0 || jump == 0
            jumps[i] = 0
            continue
        end
        kappa = isnothing(curvature) ?
            _face_curvature(field, face, curvature_samples) : curvature
        if isnothing(kappa)
            methods = (
                _get(field.reconstruction, face.minus),
                _get(field.reconstruction, face.plus),
            )
            error(
                "surface tension requires resolved curvature on face $i " *
                    "with fractions ($cm, $cp) and reconstruction methods " *
                    "$methods in cells ($(face.minus), $(face.plus))"
            )
        end
        jumps[i] = sigma * kappa * jump * face.area
    end
    return _weighted_jump_force!(
        destination, operator, work, jumps, field.geometry,
    )
end

@inline function _clip_line_interval(lo, hi, slope, offset; keep_less)
    vlo = muladd(slope, lo, offset)
    vhi = muladd(slope, hi, offset)
    inside_lo = keep_less ? vlo <= 0 : vlo >= 0
    inside_hi = keep_less ? vhi <= 0 : vhi >= 0
    inside_lo && inside_hi && return lo, hi
    !inside_lo && !inside_hi && return lo, lo
    crossing = lo + (hi - lo) * vlo / (vlo - vhi)
    return inside_lo ? (lo, crossing) : (crossing, hi)
end

# Physical liquid aperture on a face segment according to one adjacent PLIC.
# In a cut cell the solid half-plane is clipped first, so the result is bounded
# by that cell's open-face aperture rather than merely by one.
@inline _face_aperture(field, face, ref) =
    _face_aperture(field, field.geometry, face, ref)

function _face_aperture(field, geometry, face, ref)
    c = _conditional_fraction(field, ref)
    c <= 0 && return zero(c)
    mesh = field.mesh
    block = mesh.blocks[ref.block]
    center = cellcenter(mesh, ref.block, ref.i, ref.j)
    axis, tangent = face.axis, 3 - face.axis
    normal_coordinate = _wrapped_delta(
        face.center[axis] - center[axis], mesh.extent[axis],
        mesh.periodic[axis],
    ) / block.spacing[axis]
    lo, hi = _face_local_tangent_interval(mesh, ref, face)
    original_width = hi - lo
    if !isnothing(geometry)
        solid = geometry.solid
        solid_fraction = _get(solid.fraction, ref)
        solid_fraction >= one(solid_fraction) && return zero(c)
        if solid_fraction > zero(solid_fraction)
            sn = axis == 1 ? _get(solid.normal_x, ref) :
                _get(solid.normal_y, ref)
            st = tangent == 1 ? _get(solid.normal_x, ref) :
                _get(solid.normal_y, ref)
            lo, hi = _clip_line_interval(
                lo, hi, st,
                sn * normal_coordinate - _get(solid.intercept, ref);
                keep_less = false,
            )
            hi > lo || return zero(c)
        end
    end
    c >= 1 && return clamp((hi - lo) / original_width, zero(c), one(c))
    pn = axis == 1 ? _get(field.normal_x, ref) :
        _get(field.normal_y, ref)
    pt = tangent == 1 ? _get(field.normal_x, ref) :
        _get(field.normal_y, ref)
    lo, hi = _clip_line_interval(
        lo, hi, pt,
        pn * normal_coordinate - _get(field.intercept, ref);
        keep_less = true,
    )
    return clamp((hi - lo) / original_width, zero(c), one(c))
end

function _local_open_polygon(field::InterfaceField{T}, geometry, ref) where {T}
    if isnothing(geometry)
        return NTuple{2, T}[
            (-T(0.5), -T(0.5)), (T(0.5), -T(0.5)),
            (T(0.5), T(0.5)), (-T(0.5), T(0.5)),
        ]
    end
    return _open_cell_polygon(geometry, ref)
end

function _phase_first_moments(field::InterfaceField{T}, geometry) where {T}
    liquid_x = zero(T)
    liquid_y = zero(T)
    open_x = zero(T)
    open_y = zero(T)
    for ref in _cell_references(field.mesh)
        block = field.mesh.blocks[ref.block]
        center = cellcenter(field.mesh, ref.block, ref.i, ref.j)
        open_polygon = _local_open_polygon(field, geometry, ref)
        open_area, open_local_x, open_local_y =
            _polygon_area_moments(open_polygon)
        fraction = _conditional_fraction(field, ref)
        liquid_polygon = if fraction <= zero(T)
            NTuple{2, T}[]
        elseif fraction >= one(T)
            open_polygon
        else
            _clip_polygon(
                open_polygon,
                _get(field.normal_x, ref),
                _get(field.normal_y, ref),
                _get(field.intercept, ref),
            )
        end
        liquid_area, liquid_local_x, liquid_local_y =
            _polygon_area_moments(liquid_polygon)
        volume = block.volume
        dx, dy = block.spacing
        liquid_x += volume * (
            center[1] * liquid_area + dx * liquid_local_x
        )
        liquid_y += volume * (
            center[2] * liquid_area + dy * liquid_local_y
        )
        open_x += volume * (center[1] * open_area + dx * open_local_x)
        open_y += volume * (center[2] * open_area + dy * open_local_y)
    end
    return (liquid_x, liquid_y), (open_x, open_y)
end

function _gravitational_potential_energy(
        field::InterfaceField{T}, geometry;
        density::NTuple{2}, gravity::NTuple{2},
    ) where {T}
    all(iszero, gravity) && return zero(T)
    liquid_moment, open_moment = _phase_first_moments(field, geometry)
    gas_moment = (
        open_moment[1] - liquid_moment[1],
        open_moment[2] - liquid_moment[2],
    )
    return -sum(1:2) do axis
        gravity[axis] * (
            density[1] * liquid_moment[axis] +
                density[2] * gas_moment[axis]
        )
    end
end

"""
    gravity!(acceleration, interface, pressure_operator, workspace;
             density, gravity)

Thesis mimetic gravity model: aperture density times the gradient of
`dot(gravity, interface_centroid)`. Pure cells use their cell centroids.
On mixed-level meshes aperture-weighted jumps enter the pressure's weighted
inverse metric, retaining exact cancellation when a hydrostatic potential
exists. Nonzero gravity along periodic directions is rejected because its
potential is not periodic.
"""
function gravity!(
        destination::FaceVelocity, field::InterfaceField,
        operator::EllipticOperator, work::EllipticWorkspace;
        density::NTuple{2}, gravity::NTuple{2}
    )
    destination.mesh === field.mesh === operator.mesh || throw(ArgumentError("gravity arguments must share a mesh"))
    all(rho -> isfinite(rho)&&rho > 0, density) || throw(ArgumentError("densities must be finite and positive"))
    all(isfinite, gravity) || throw(ArgumentError("gravity must be finite"))
    mesh = field.mesh
    any(a -> mesh.periodic[a]&&!iszero(gravity[a]), 1:2) &&
        throw(ArgumentError("gravity potential requires nonperiodic boundaries along gravity"))
    potential = work.input
    @_parallel for index in 1:length(operator.references)
        ref = operator.references[index]
        x, y = cellcenter(mesh, ref.block, ref.i, ref.j)
        c = _conditional_fraction(field, ref)
        if 0 < c < 1
            mx, my = _get(field.normal_x, ref), _get(field.normal_y, ref)
            cx, cy = _interface_centroid(mx, my, _get(field.intercept, ref))
            hx, hy = mesh.blocks[ref.block].spacing
            x += cx * hx;y += cy * hy
        end
        @inbounds potential[index] = gravity[1] * x + gravity[2] * y
    end
    jumps = operator.gradient_operator.pressure_jump
    # Reuse the same reconstructed liquid opening used by the phase
    # divergence operator. In a two-velocity step this also primes the
    # post-transport aperture cache before pressure assembly.
    liquid_aperture = first(_current_phase_face_apertures!(field))
    @_parallel for i in 1:length(mesh.faces)
        face = mesh.faces[i]
        aperture = @inbounds liquid_aperture[i]
        rho = density[1] * aperture + density[2] * (1 - aperture)
        minus = _cell_index(
            face.minus, mesh.block_size[1], mesh.block_size[2]
        )
        plus = _cell_index(
            face.plus, mesh.block_size[1], mesh.block_size[2]
        )
        jumps[i] = rho * face.area * (potential[plus] - potential[minus])
    end
    return _weighted_jump_force!(
        destination, operator, work, jumps, field.geometry,
    )
end
