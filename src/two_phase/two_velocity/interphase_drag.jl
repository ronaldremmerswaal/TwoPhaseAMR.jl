"""
    InterphaseDrag(; component = :tangential)

Viscous coupling of the two intrinsic phase velocities of `TwoVelocityModel`
across the interface, opt-in through `TwoVelocityModel(; interphase_drag =
InterphaseDrag())` (the default `nothing` couples nothing). It is off by
default because it smears the vortex sheet with which the two-velocity model
represents an unresolved shear layer into a grid-scale layer: the viscous
Kelvin--Helmholtz case with a shear layer of half a cell decays with the
drag (growth-rate error 3.3 % without, 182 % with it, `MimeticProjection`).

The phase-mass weighted mixture viscous step gives both phases the same
increment, so on its own it never damps the jump `u_gas - u_liquid` of a face
whose staggered control volume holds both phases. The drag adds, on every such
face, the interface shear of the jump: the force `W (u_gas - u_liquid)` on the
liquid and its negative on the gas, with the conductance
`W = mu_Gamma |Gamma_f| w_f / delta_f`. The control volume is the one of the
staggered liquid fraction that gives the masses: each adjacent cell
contributes its open part of the face's strip with the cell's phase fractions,
so its PLIC chord and phase centroids enter with the weight of that strip.
`|Gamma_f|` is the weighted chord length, `delta_f` the distance of the two
phase centroids along the weighted mean interface normal (at least a quarter
of the mean cell width along the face normal), and between a pure liquid and a
pure gas cell the face itself is the interface. `mu_Gamma` is the phase
viscosity mixture of the staggered liquid fraction under the stress operator's
averaging rule (harmonic by default). With the staggered phase masses
`m_liquid`, `m_gas` the jump decays at the rate
`lambda = W (1/m_liquid + 1/m_gas)`; the step applies the exact solution
`jump * exp(-lambda dt)` at fixed mixture momentum after the mixture solve.
It conserves the mixture momentum of every face, does not increase the
kinetic energy, and does nothing at rest, on faces where a phase has no
volume, and when both viscosities are zero.

`component = :tangential` (the default) takes the component weight
`w_f = t_k^2 / (t_1^4 + t_2^4)` with the interface tangent `t` and the face
axis `k`: a jump along the interface then dissipates
`sum_k t_k^2 w_k = 1` per unit length for every orientation, the same as
`:full` (`w_f = 1`), while a jump normal to the interface is not coupled where
the interface is aligned with the grid (`2 t_1^2 t_2^2 / (t_1^4 + t_2^4)`,
zero at 0 and 90 degrees, one at 45 degrees, where the face components cannot
separate the two).
"""
struct InterphaseDrag
    component::Symbol

    function InterphaseDrag(component::Symbol)
        component in (:full, :tangential) || throw(
            ArgumentError("InterphaseDrag component must be :full or :tangential"),
        )
        return new(component)
    end
end

InterphaseDrag(; component::Symbol = :tangential) = InterphaseDrag(component)

# One directed boundary segment p -> q of a polygon: accumulate twice its area
# and six times its first moments (Green's theorem).
@inline function _green_segment(accumulator, p, q)
    twice_area, moment_x, moment_y = accumulator
    cross = p[1] * q[2] - q[1] * p[2]
    return (
        twice_area + cross,
        moment_x + (p[1] + q[1]) * cross,
        moment_y + (p[2] + q[2]) * cross,
    )
end

@inline function _plic_crossing(a, b, value_a, value_b)
    fraction = value_a / (value_a - value_b)
    return (
        muladd(fraction, b[1] - a[1], a[1]),
        muladd(fraction, b[2] - a[2], a[2]),
    )
end

"""Clip the convex polygon `vertices[1:count]` (a fixed-size tuple, at most
eight vertices) to `normal . x <= intercept` (`keep_less`) or `>=`, without
allocation. Returns the clipped vertices, their count and the two points
where the line enters and leaves the polygon (equal if it does not cut it)."""
@inline function _clip_convex(vertices, count, normal_x, normal_y, intercept, keep_less)
    T = typeof(intercept)
    result = vertices
    kept = 0
    first_cut = second_cut = (zero(T), zero(T))
    cuts = 0
    count == 0 && return result, 0, first_cut, second_cut
    previous = vertices[count]
    previous_value = muladd(normal_x, previous[1], muladd(normal_y, previous[2], -intercept))
    for k in 1:count
        current = vertices[k]
        current_value = muladd(normal_x, current[1], muladd(normal_y, current[2], -intercept))
        previous_inside = keep_less ? previous_value <= zero(T) : previous_value >= zero(T)
        current_inside = keep_less ? current_value <= zero(T) : current_value >= zero(T)
        if previous_inside != current_inside
            point = _plic_crossing(previous, current, previous_value, current_value)
            kept += 1
            result = Base.setindex(result, point, kept)
            cuts += 1
            cuts == 1 ? (first_cut = point) : (second_cut = point)
        end
        if current_inside
            kept += 1
            result = Base.setindex(result, current, kept)
        end
        previous = current
        previous_value = current_value
    end
    cuts < 2 && (second_cut = first_cut)
    return result, kept, first_cut, second_cut
end

@inline function _tuple_polygon_area_moments(vertices, count, ::Type{T}) where {T}
    accumulator = (zero(T), zero(T), zero(T))
    count < 3 && return accumulator
    previous = vertices[count]
    for k in 1:count
        accumulator = _green_segment(accumulator, previous, vertices[k])
        previous = vertices[k]
    end
    return accumulator[1] / 2, accumulator[2] / 6, accumulator[3] / 6
end

"""Open and liquid geometry of one cell in its normalized coordinates
`[-1/2, 1/2]^2`: open area and first moments, liquid area and first moments,
and the vector of the PLIC chord in the open cell (zero for a pure cell).
The open part is the cell outside an embedded solid's plane, the liquid part
its intersection with the PLIC half-plane, allocation-free."""
@inline function _cell_phase_geometry(field::InterfaceField{T}, ref) where {T}
    half = T(0.5)
    vertices = (
        (-half, -half), (half, -half), (half, half), (-half, half),
        (zero(T), zero(T)), (zero(T), zero(T)), (zero(T), zero(T)), (zero(T), zero(T)),
    )
    count = 4
    geometry = field.geometry
    if !isnothing(geometry)
        solid = geometry.solid
        solid_fraction = _get(solid.fraction, ref)
        if solid_fraction >= one(T)
            count = 0
        elseif solid_fraction > zero(T)
            vertices, count, _, _ = _clip_convex(
                vertices, count, _get(solid.normal_x, ref), _get(solid.normal_y, ref),
                _get(solid.intercept, ref), false,
            )
        end
    end
    open = _tuple_polygon_area_moments(vertices, count, T)
    chord = (zero(T), zero(T))
    fraction = _conditional_fraction(field, ref)
    liquid = if fraction <= zero(T) || count == 0
        (zero(T), zero(T), zero(T))
    elseif fraction >= one(T)
        open
    else
        clipped, kept, first_cut, second_cut = _clip_convex(
            vertices, count, _get(field.normal_x, ref), _get(field.normal_y, ref),
            _get(field.intercept, ref), true,
        )
        chord = (second_cut[1] - first_cut[1], second_cut[2] - first_cut[2])
        _tuple_polygon_area_moments(clipped, kept, T)
    end
    return open, liquid, chord
end

"""Open area of the strip of `face` in cell `ref`, normalized by the cell
volume: the weight of the cell in the staggered liquid fraction."""
@inline function _strip_open_area(field::InterfaceField{T}, ref, face, lo, hi) where {T}
    geometry = field.geometry
    isnothing(geometry) && return hi - lo
    half = T(0.5)
    return face.axis == 1 ?
        _open_subcell_area(geometry.solid, ref, -half, half, lo, hi) :
        _open_subcell_area(geometry.solid, ref, lo, hi, -half, half)
end

"""Interface length, phase-centroid distance and component weight of the
staggered control volume of `face`, consistent with its staggered liquid
fraction and volume.

Each adjacent cell contributes half of the open part of the face's strip
(`_strip_open_area`), with the cell's phase fractions; its open-cell PLIC
chord and phase centroids are scaled by the ratio of that contribution to the
open cell area, so the liquid area of the volume is the staggered fraction
times its volume. `|Gamma_f|` is the sum of the scaled chords, the interface
normal their length-weighted mean normal, and `delta_f` the distance of the
liquid and gas centroids along it, bounded below by a quarter of the mean
cell width along the face normal. If no cell contributes a chord (a pure
liquid cell next to a pure gas cell), the face itself is the interface:
length `face.area`, normal along the face normal. Returns
`(|Gamma_f|, delta_f, w_f, liquid fraction of the volume)`, with zero length
only if the volume holds one phase.
"""
@inline function _interphase_contact(field::InterfaceField{T}, face, tangential) where {T}
    mesh = field.mesh
    axis = face.axis
    tangent = 3 - axis
    liquid = (zero(T), zero(T), zero(T))  # area, first moments about the face centre
    total = (zero(T), zero(T), zero(T))
    chord_length = zero(T)
    normal_sum = (zero(T), zero(T))
    width = zero(T)
    for (side, ref) in ((-1, face.minus), (1, face.plus))
        block = mesh.blocks[ref.block]
        spacing = (T(block.spacing[1]), T(block.spacing[2]))
        width += spacing[axis]
        lo, hi = _face_local_tangent_interval(mesh, ref, face)
        lo, hi = T(lo), T(hi)
        open, phase, chord = _cell_phase_geometry(field, ref)
        open[1] > zero(T) || continue
        # Cell contribution (normalized) over the open cell area: the scale of
        # its whole-cell geometry in this control volume.
        scale = _strip_open_area(field, ref, face, lo, hi) / (2open[1])
        scale > zero(T) || continue
        jacobian = spacing[1] * spacing[2]
        offset_axis = side * spacing[axis] / 2
        offset_tangent = -(lo + hi) / 2 * spacing[tangent]
        offset = axis == 1 ? (offset_axis, offset_tangent) :
            (offset_tangent, offset_axis)
        weight = scale * jacobian
        total = (
            muladd(weight, open[1], total[1]),
            muladd(weight, muladd(open[2], spacing[1], open[1] * offset[1]), total[2]),
            muladd(weight, muladd(open[3], spacing[2], open[1] * offset[2]), total[3]),
        )
        liquid = (
            muladd(weight, phase[1], liquid[1]),
            muladd(weight, muladd(phase[2], spacing[1], phase[1] * offset[1]), liquid[2]),
            muladd(weight, muladd(phase[3], spacing[2], phase[1] * offset[2]), liquid[3]),
        )
        piece = scale * hypot(chord[1] * spacing[1], chord[2] * spacing[2])
        piece > zero(T) || continue
        # Plane normal in physical coordinates, pointing into the gas.
        gradient = (
            _get(field.normal_x, ref) / spacing[1], _get(field.normal_y, ref) / spacing[2],
        )
        magnitude = hypot(gradient[1], gradient[2])
        chord_length += piece
        normal_sum = (
            muladd(piece, gradient[1] / magnitude, normal_sum[1]),
            muladd(piece, gradient[2] / magnitude, normal_sum[2]),
        )
    end
    gas_area = total[1] - liquid[1]
    liquid[1] > zero(T) && gas_area > zero(T) ||
        return (zero(T), zero(T), zero(T), liquid[1] / total[1])
    normal_norm = hypot(normal_sum[1], normal_sum[2])
    if !(chord_length > zero(T) && normal_norm > zero(T))
        # Pure liquid next to pure gas (or degenerate chords): the face.
        chord_length = T(face.area)
        normal_sum = axis == 1 ? (one(T), zero(T)) : (zero(T), one(T))
        normal_norm = one(T)
    end
    normal = (normal_sum[1] / normal_norm, normal_sum[2] / normal_norm)
    separation = (
        (total[2] - liquid[2]) / gas_area - liquid[2] / liquid[1],
        (total[3] - liquid[3]) / gas_area - liquid[3] / liquid[1],
    )
    distance = max(
        abs(muladd(separation[1], normal[1], separation[2] * normal[2])),
        width / 8,
    )
    # Tangent component along the face axis: t = (-n_2, n_1), t_axis^2 = n_tangent^2.
    component_weight = tangential ?
        normal[tangent]^2 / (normal[1]^4 + normal[2]^4) : one(T)
    return chord_length, distance, component_weight, liquid[1] / total[1]
end

"""Decay rate `lambda = W (1/m_liquid + 1/m_gas)` of the phase-velocity jump
on `face` with staggered liquid fraction `fraction`, or zero if the face is not
mixed (see [`InterphaseDrag`](@ref))."""
@inline function _interphase_drag_rate(
        field::InterfaceField{T}, face, fraction, volume, density, viscosity,
        averaging, tangential,
    ) where {T}
    volume > zero(T) && zero(T) < fraction < one(T) || return zero(T)
    chord_length, distance, weight, _ = _interphase_contact(field, face, tangential)
    chord_length > zero(T) && weight > zero(T) || return zero(T)
    mu = T(_phase_dynamic_viscosity(fraction, viscosity, averaging))
    conductance = mu * chord_length * weight / distance
    return conductance / volume * (
        inv(density[1] * fraction) + inv(density[2] * (one(T) - fraction))
    )
end

_interphase_drag!(velocity, ::Nothing, model, operator, dt) = velocity

"""Relax the phase-velocity jump of every mixed face by `exp(-lambda dt)` at
fixed mixture momentum (see [`InterphaseDrag`](@ref)), with the staggered
masses and volumes of the mixture solve in `operator`."""
function _interphase_drag!(
        velocity::TwoPhaseVelocity, drag::InterphaseDrag, model, operator, dt,
    )
    all(iszero, model.dynamic_viscosity) && return velocity
    iszero(dt) && return velocity
    interface = model.interface
    reconstruct!(interface)
    _interphase_drag!(
        velocity[1].interior, velocity[2].interior, interface,
        model.density, model.dynamic_viscosity, operator.averaging,
        operator.volumes, drag.component === :tangential,
        eltype(interface)(dt),
    )
    return velocity
end

function _interphase_drag!(
        liquid::AbstractVector{T}, gas::AbstractVector{T},
        interface::InterfaceField{T}, density, viscosity,
        averaging::AbstractViscosityAveraging, volumes, tangential, dt,
    ) where {T}
    faces = interface.mesh.faces
    rho_liquid, rho_gas = density
    @_parallel for index in 1:length(faces)
        @inbounds begin
            liquid_value = liquid[index]
            gas_value = gas[index]
            isfinite(liquid_value) && isfinite(gas_value) || continue
            jump = gas_value - liquid_value
            iszero(jump) && continue
            face = faces[index]
            fraction = _staggered_liquid_fraction(interface, interface.geometry, face)
            rate = _interphase_drag_rate(
                interface, face, fraction, volumes[index], density, viscosity,
                averaging, tangential,
            )
            rate > zero(T) || continue
            liquid_mass = rho_liquid * fraction
            gas_mass = rho_gas * (one(T) - fraction)
            # Equal and opposite momentum changes; -expm1 keeps small decays exact.
            change = -expm1(-rate * dt) * jump / (liquid_mass + gas_mass)
            liquid[index] = muladd(gas_mass, change, liquid_value)
            gas[index] = muladd(-liquid_mass, change, gas_value)
        end
    end
    return liquid, gas
end
