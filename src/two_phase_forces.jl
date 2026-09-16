"""Continuum surface tension with the same weighted gradient as pressure."""
struct CSF end
"""Ghost-fluid surface tension with a sharp indicator jump.
`GFM()` uses the thesis's conservative C-GFM distance (staggered fraction).
`GFM(conservative=false)` uses geometric centre-to-centre phase distances;
as discussed in the thesis, this variant does not conserve linear momentum.
"""
struct GFM
    conservative::Bool
end
GFM(; conservative = true) = GFM(conservative)

"""Staggered mixture density from the mass-compatible cell-to-face interpolant."""
function mixture_density(field::InterfaceField{T}, density::NTuple{2}) where {T}
    all(rho -> isfinite(rho) && rho > 0, density) || throw(ArgumentError("phase densities must be finite and positive"))
    mesh = field.mesh
    result = Vector{T}(undef, length(mesh.faces))
    for (i, face) in pairs(mesh.faces)
        hm = mesh.blocks[face.minus.block].spacing[face.axis]
        hp = mesh.blocks[face.plus.block].spacing[face.axis]
        c = (hm * _get(field.fraction, face.minus) + hp * _get(field.fraction, face.plus)) / (hm + hp)
        result[i] = density[1] * c + density[2] * (1 - c)
    end
    return result
end
"""Construct the pressure operator using the inverse staggered mixture density."""
function PoissonOperator(
        field::InterfaceField; density = (1.0, 1.0),
        formulation = CSF(), template = nothing
    )
    rho = mixture_density(field, density)
    if formulation isa GFM && !formulation.conservative
        for (i, face) in pairs(field.mesh.faces)
            fraction = _gfm_distance_fraction(field, face)
            rho[i] = density[1] * fraction + density[2] * (1 - fraction)
        end
    end
    coefficient = inv.(rho)
    if isnothing(template)
        return PoissonOperator(
            field.mesh; face_coefficient = coefficient, T = eltype(field)
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

function _line_phase_fraction(field, ref, p0, p1)
    c = _get(field.fraction, ref)
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

function _weighted_jump_force!(destination, operator, work, jumps)
    for i in eachindex(jumps)
        work.face_input[i] = operator.square_root_face_coefficient[i] * jumps[i]
    end
    _mul_inverse_metric!(work.face_output, operator.gradient_operator.inverse_metric, work.face_input)
    for i in eachindex(jumps)
        destination.interior[i] = operator.square_root_face_coefficient[i] * work.face_output[i]
    end
    fill!(destination.boundary, zero(eltype(destination)))
    return destination
end
function _face_curvature(field, face)
    total = weight = 0.0
    for ref in (face.minus, face.plus)
        c = _get(field.fraction, ref);k = _get(field.curvature, ref)
        w = c * (1 - c)
        if w > 0 && isfinite(k)
            total += w * k;weight += w
        end
    end
    return weight > 0 ? total / weight : nothing
end

"""
    surface_tension!(acceleration, interface, pressure_operator, workspace;
                     sigma, formulation=CSF(), curvature=nothing)

Overwrite acceleration with CSF or conservative GFM surface tension. The
optional constant `curvature` is for isolating the balanced-force identity in
tests. Otherwise use fraction-weighted LHF/GHF curvature. Missing curvature on
an active interface face raises an error. The same weighted inverse metric as
pressure is applied to the curvature-weighted phase jump, including on AMR.
"""
function surface_tension!(
        destination::FaceVelocity, field::InterfaceField,
        operator::EllipticOperator, work::EllipticWorkspace;
        sigma, formulation = CSF(), curvature = nothing
    )
    destination.mesh === field.mesh === operator.mesh || throw(ArgumentError("surface tension arguments must share a mesh"))
    formulation isa Union{CSF, GFM} || throw(ArgumentError("formulation must be CSF() or GFM()"))
    isfinite(sigma) && sigma >= 0 || throw(ArgumentError("sigma must be finite and nonnegative"))
    isnothing(curvature) || isfinite(curvature) || throw(ArgumentError("constant curvature must be finite"))
    jumps = operator.gradient_operator.pressure_jump
    for (i, face) in pairs(field.mesh.faces)
        cm, cp = _get(field.fraction, face.minus), _get(field.fraction, face.plus)
        jump = formulation isa CSF ? cp - cm : (cp > 0.5) - (cm > 0.5)
        if sigma == 0 || jump == 0
            jumps[i] = 0
            continue
        end
        kappa = isnothing(curvature) ? _face_curvature(field, face) : curvature
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
    return _weighted_jump_force!(destination, operator, work, jumps)
end

# Fraction on a face segment according to a single adjacent reconstruction.
function _face_aperture(field, face, ref)
    c = _get(field.fraction, ref)
    (c == 0 || c == 1) && return c
    mesh = field.mesh;block = mesh.blocks[ref.block]
    center = cellcenter(mesh, ref.block, ref.i, ref.j)
    mx, my = _get(field.normal_x, ref), _get(field.normal_y, ref)
    dx = _wrapped_delta(face.center[1] - center[1], mesh.extent[1], mesh.periodic[1]) / block.spacing[1]
    dy = _wrapped_delta(face.center[2] - center[2], mesh.extent[2], mesh.periodic[2]) / block.spacing[2]
    signed = _get(field.intercept, ref) - mx * dx - my * dy
    slope = (face.axis == 1 ? my / block.spacing[2] : mx / block.spacing[1]) * face.area
    abs(slope) <= eps(eltype(field)) && return signed >= 0 ? one(c) : zero(c)
    return clamp(0.5 + signed / abs(slope), zero(c), one(c))
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
    for (index, ref) in pairs(operator.references)
        x, y = cellcenter(mesh, ref.block, ref.i, ref.j)
        c = _get(field.fraction, ref)
        if 0 < c < 1
            mx, my = _get(field.normal_x, ref), _get(field.normal_y, ref)
            cx, cy = _interface_centroid(mx, my, _get(field.intercept, ref))
            hx, hy = mesh.blocks[ref.block].spacing
            x += cx * hx;y += cy * hy
        end
        @inbounds potential[index] = gravity[1] * x + gravity[2] * y
    end
    jumps = operator.gradient_operator.pressure_jump
    for (i, face) in pairs(mesh.faces)
        aperture = (_face_aperture(field, face, face.minus) + _face_aperture(field, face, face.plus)) / 2
        rho = density[1] * aperture + density[2] * (1 - aperture)
        minus = _cell_index(
            face.minus, mesh.block_size[1], mesh.block_size[2]
        )
        plus = _cell_index(
            face.plus, mesh.block_size[1], mesh.block_size[2]
        )
        jumps[i] = rho * face.area * (potential[plus] - potential[minus])
    end
    return _weighted_jump_force!(destination, operator, work, jumps)
end
