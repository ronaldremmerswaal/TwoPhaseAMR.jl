# 8-point Gauss--Legendre rule, stored as positive abscissae and paired weights.
const _LS_GL8_X = (0.1834346424956498, 0.525532409916329, 0.7966664774136267, 0.9602898564975363)
const _LS_GL8_W = (0.362683783378362, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763)

@inline function _levelset_value(phi, x, y)
    value = phi(x, y)
    value isa Real && !(value isa Bool) && isfinite(value) ||
        throw(ArgumentError("level set must return finite real values, not Boolean indicators"))
    return value
end
function _levelset_root(f, a, b, fa, fb, tol)
    iszero(fa) && return a
    iszero(fb) && return b
    (fa < 0) != (fb < 0) || throw(ArgumentError("level-set root is not bracketed"))
    for _ in 1:80
        b - a <= tol && return (a + b) / 2
        mid = (a + b) / 2
        fm = f(mid)
        iszero(fm) && return mid
        if (fm < 0) == (fa < 0)
            a, fa = mid, fm
        else
            b, fb = mid, fm
        end
    end
    error("level-set root did not converge")
end
@inline function _levelset_gauss(f, a, b)
    mid, half = (a + b) / 2, (b - a) / 2
    total = zero(mid)
    for (x, w) in zip(_LS_GL8_X, _LS_GL8_W)
        total += w * (f(mid - half * x) + f(mid + half * x))
    end
    return half * total
end
function _levelset_quadrature(f, a, b, tol, depth, whole = _levelset_gauss(f, a, b))
    mid = (a + b) / 2
    left, right = _levelset_gauss(f, a, mid), _levelset_gauss(f, mid, b)
    fine = left + right
    abs(fine - whole) <= tol && return fine
    depth > 0 || error("level-set quadrature failed to reach atol; increase maxdepth")
    return _levelset_quadrature(f, a, mid, tol / 2, depth - 1, left) +
        _levelset_quadrature(f, mid, b, tol / 2, depth - 1, right)
end

function _levelset_height_area(phi, lo, hi, axis, tol, maxdepth)
    tangent = 3 - axis
    a, b = lo[tangent], hi[tangent]
    bottom, top = lo[axis], hi[axis]
    height = top - bottom
    root_tol = max(
        8eps(typeof(a)) * max(abs(a), abs(b), abs(bottom), abs(top), height),
        tol / (b - a) * 0.001
    )
    value(t, s) = axis == 1 ? _levelset_value(phi, s, t) : _levelset_value(phi, t, s)
    function column(t)
        fb, ft = value(t, bottom), value(t, top)
        fb <= 0 && ft <= 0 && return height
        fb >= 0 && ft >= 0 && return zero(height)
        root = _levelset_root(s -> value(t, s), bottom, top, fb, ft, root_tol)
        return fb < 0 ? root - bottom : top - root
    end
    # Split at intersections with the two transverse edges: the clipped height
    # has a kink there even if the implicit interface is analytic.
    cuts = [a, b]
    for s in (bottom, top)
        previous = a; fp = value(a, s)
        for k in 1:8
            next = a + (b - a) * k / 8; fn = value(next, s)
            if (fp < 0) != (fn < 0) && fp != 0 && fn != 0
                push!(cuts, _levelset_root(t -> value(t, s), previous, next, fp, fn, root_tol))
            elseif fn == 0 && a < next < b
                push!(cuts, next)
            end
            previous, fp = next, fn
        end
    end
    sort!(cuts);unique!(cuts)
    result = zero(height)
    for i in 1:(length(cuts) - 1)
        left, right = cuts[i], cuts[i + 1]
        result += _levelset_quadrature(
            column, left, right,
            tol * (right - left) / (b - a), maxdepth
        )
    end
    return result
end

function _levelset_area(phi, lo, hi, tol, depth, maxdepth, lipschitz, min_depth)
    mid = ((lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
    widths = (hi[1] - lo[1], hi[2] - lo[2]);area = prod(widths)
    fc = _levelset_value(phi, mid...)
    if !isnothing(lipschitz) && abs(fc) >= lipschitz * hypot(widths...) / 2
        return fc < 0 ? area : zero(area)
    end
    # The centre and edge midpoints detect enclosed components which a
    # corner-only method misses. min_depth controls additional feature searches.
    xs = (lo[1], mid[1], hi[1]);ys = (lo[2], mid[2], hi[2])
    values = [_levelset_value(phi, x, y) for x in xs,y in ys]
    low, high = extrema(values)
    # A zero sample is an interface contact, not evidence that the box is
    # single-phase. Treat it as a candidate height graph so planes passing
    # exactly through cell vertices or edge midpoints do not recurse forever.
    sign_scale = max(abs(low), abs(high))
    sign_tolerance = 64eps(typeof(float(sign_scale))) * sign_scale
    same_sign = low > sign_tolerance || high < -sign_tolerance
    if depth >= min_depth && same_sign && isnothing(lipschitz)
        return high <= 0 ? area : zero(area)
    end
    if !same_sign && depth >= min_depth
        gx, gy = values[3, 2] - values[1, 2], values[2, 3] - values[2, 1]
        axes = abs(gx / widths[1]) > abs(gy / widths[2]) ? (1, 2) : (2, 1)
        for axis in axes
            direction = axis == 1 ? sign(gx) : sign(gy)
            iszero(direction) && continue
            monotone = all(1:3) do t
                v = axis == 1 ? @view(values[:, t]) : @view(values[t, :])
                direction * (v[2] - v[1]) > 0 && direction * (v[3] - v[2]) > 0
            end
            monotone || continue
            # Secant slopes alone can hide an extremum close to a box edge
            # (and thus a second crossing on the same ray). Check local
            # derivatives as well, including at the endpoints.
            delta = widths[axis] * 1.0e-5
            derivative_ok = true
            for x in xs,y in ys
                difference = axis == 1 ?
                    _levelset_value(phi, x + delta, y) - _levelset_value(phi, x - delta, y) :
                    _levelset_value(phi, x, y + delta) - _levelset_value(phi, x, y - delta)
                if direction * difference < -64eps(typeof(float(fc))) * max(abs(fc), abs(high), abs(low))
                    derivative_ok = false
                    break
                end
            end
            derivative_ok || continue
            return _levelset_height_area(phi, lo, hi, axis, tol, maxdepth)
        end
    end
    # A Lipschitz bound prevents classifying an unobserved enclosed component
    # as pure. An unresolved box contributes at most its entire area.
    if area <= tol
        return fc <= 0 ? area : zero(area)
    end
    depth < maxdepth || error("level-set geometry remains unresolved; increase maxdepth or supply a better-resolved level set")
    total = zero(area)
    for j in 1:2,i in 1:2
        childlo = (i == 1 ? lo[1] : mid[1], j == 1 ? lo[2] : mid[2])
        childhi = (i == 1 ? mid[1] : hi[1], j == 1 ? mid[2] : hi[2])
        total += _levelset_area(phi, childlo, childhi, tol / 4, depth + 1, maxdepth, lipschitz, min_depth)
    end
    return total
end

"""
    InterfaceField(mesh, levelset; atol=1e-10, maxdepth=16,
                   min_depth=0, lipschitz=nothing, T=eltype(mesh.origin))
    initialize!(interface, levelset; atol=1e-10, maxdepth=16,
                min_depth=0, lipschitz=nothing)

Initialize liquid cell averages from `levelset(x,y) <= 0`. The function need
not be a signed distance. Mixed cells use root-solved height integrals with
adaptive Gaussian quadrature, splitting at cell-edge crossings. Geometry that
cannot be represented as one height graph is subdivided. `atol` is the target
absolute error in each cell's fraction, not in the domain volume.

An optional global Lipschitz bound (`lipschitz=1` for a signed distance) gives a
certified pure-cell test and searches same-sign boxes for enclosed components.
Without a bound, purity/monotonicity tests are sampled and assume resolved,
smooth geometry. Arbitrarily small or oscillatory features cannot be guaranteed
from finitely many function samples; `min_depth` forces a finer initial search.
Quadrature error estimates are numerical, not interval-certified error bounds.
Failures to resolve geometry or quadrature throw rather than silently accepting
an inaccurate fraction. Initialization is transactional: failed quadrature does
not modify the existing fraction field. This is an initialization operation;
mesh adaptation uses geometric PLIC transfer instead of resampling the level set.
"""
function InterfaceField(
        mesh::BlockTreeMesh, levelset;
        T = eltype(mesh.origin), contact_angle = nothing,
        geometry = nothing, embedded_contact_angle = nothing, kwargs...,
    )
    field = InterfaceField(
        mesh; T, contact_angle, geometry, embedded_contact_angle,
    )
    return initialize!(field, levelset; kwargs...)
end
function initialize!(
        field::InterfaceField{T}, levelset;
        atol = 1.0e-10, maxdepth = 16, min_depth = 0, lipschitz = nothing
    ) where {T}
    isfinite(atol) && 0 < atol < 1 || throw(ArgumentError("atol must be finite and in (0,1)"))
    maxdepth isa Integer && min_depth isa Integer && 0 <= min_depth <= maxdepth <= 30 ||
        throw(ArgumentError("require integer depths 0 <= min_depth <= maxdepth <= 30"))
    isnothing(lipschitz) || (isfinite(lipschitz) && lipschitz > 0) ||
        throw(ArgumentError("lipschitz must be finite and positive"))
    if !isnothing(field.geometry)
        # Initialize the two implicit surfaces independently to the requested
        # quadrature tolerance, then intersect their PLIC polygons exactly.
        # This avoids integrating max(phi_interface,-phi_solid), whose kink at
        # a triple point defeats otherwise efficient height quadrature.
        temporary = InterfaceField(field.mesh; T)
        initialize!(
            temporary, levelset; atol, maxdepth, min_depth, lipschitz,
        )
        mesh = field.mesh
        nx, ny = mesh.block_size
        h = mesh.halo
        _parallel_block_columns(mesh) do block, first_j, last_j
            destination = @inbounds field.fraction.blocks[block]
            for j in first_j:last_j, i in 1:nx
                ref = CellRef(block, i, j)
                open_polygon = _open_cell_polygon(field.geometry, ref)
                open_area = _polygon_area(open_polygon)
                physical_fraction = if open_area <= zero(T)
                    zero(T)
                else
                    c = _conditional_fraction(temporary, ref)
                    liquid_area = if c <= zero(T)
                        zero(T)
                    elseif c >= one(T)
                        open_area
                    else
                        liquid_polygon = _clip_polygon(
                            open_polygon,
                            _get(temporary.normal_x, ref),
                            _get(temporary.normal_y, ref),
                            _get(temporary.intercept, ref); keep_less = true,
                        )
                        _polygon_area(liquid_polygon)
                    end
                    clamp(liquid_area, zero(T), open_area)
                end
                @inbounds destination[i + h, j + h] = physical_fraction
            end
        end
        return reconstruct!(field)
    end
    mesh = field.mesh;nx, ny = mesh.block_size
    h = mesh.halo
    result = CellField(mesh, T)
    endpoint_tolerance = 64eps(T)
    _parallel_block_columns(mesh) do b, first_j, last_j
        block = @inbounds mesh.blocks[b]
        values = @inbounds result.blocks[b]
        dx, dy = block.spacing
        tolerance = atol * block.volume
        @inbounds for j in first_j:last_j, i in 1:nx
            x = block.origin[1] + (i - 1 / 2) * dx
            y = block.origin[2] + (j - 1 / 2) * dy
            lo = (x - dx / 2, y - dy / 2)
            hi = (x + dx / 2, y + dy / 2)
            area = _levelset_area(
                levelset, lo, hi, tolerance, 0, maxdepth,
                lipschitz, min_depth,
            )
            fraction = clamp(T(area / block.volume), zero(T), one(T))
            # A geometrically pure cell can divide to `1 - O(eps)` because its
            # area and volume follow different floating-point operation paths.
            # Leaving that roundoff sliver marks every cell in a pure phase as
            # an interface cell and can feed meaningless normals into curvature
            # and force stencils. This only removes endpoint roundoff; the much
            # larger `sqrt(eps(T))` floor used to recover intrinsic phase values
            # remains an advection concern and does not erase resolved
            # geometric slivers.
            fraction = fraction <= endpoint_tolerance ? zero(T) :
                fraction >= one(T) - endpoint_tolerance ? one(T) : fraction
            values[i + h, j + h] = fraction
        end
    end
    _parallel_for(nblocks(mesh); grain = 4) do b
        copyto!(field.fraction.blocks[b], result.blocks[b])
    end
    return reconstruct!(field)
end
