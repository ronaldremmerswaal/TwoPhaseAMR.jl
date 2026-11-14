"""
    InterphaseDrag(; component = :tangential)

Viscous coupling of the two intrinsic phase velocities of `TwoVelocityModel`
across the interface (the default of that model; pass `interphase_drag =
nothing` to disable it).

The phase-mass weighted mixture viscous step gives both phases the same
increment, so on its own it never damps the jump `u_gas - u_liquid` of a face
whose staggered control volume holds both phases. The drag adds, on every such
face, the interface shear of the jump: the force `W (u_gas - u_liquid)` on the
liquid and its negative on the gas, with the conductance
`W = mu_Gamma |Gamma_f| / delta_f`. `|Gamma_f|` is the PLIC interface length
in the staggered control volume of the face, `delta_f` the distance of the two
phase centroids in that volume along the interface normal (at least a quarter
of the mean cell width along the face normal) and `mu_Gamma` the phase
viscosity mixture of the staggered liquid fraction under the stress operator's
averaging rule (harmonic by default). With the staggered phase masses
`m_liquid`, `m_gas` the jump decays at the rate
`lambda = W (1/m_liquid + 1/m_gas)`; the step applies the exact solution
`jump * exp(-lambda dt)` at fixed mixture momentum after the mixture solve.
It conserves the mixture momentum of every face, does not increase the
kinetic energy, and does nothing at rest, on faces where a phase has no
volume, and when both viscosities are zero.

`component = :tangential` (the default) weights the coupling of a face
component by `|n_f . tau_Gamma|`, the share of that component tangential to
the interface; `:full` couples the whole component. The two agree to a few
percent in the capillary-wave and Kelvin--Helmholtz studies, with
`:tangential` slightly closer to the reference.
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

"""Liquid part `normal_x*x + normal_y*y <= intercept` of the rectangle
`[x0, x1] x [y0, y1]` (normalized cell coordinates): its area, its first
moments and the vector of the clipped interface chord, allocation-free."""
@inline function _plic_rectangle_clip(normal_x, normal_y, intercept, x0, x1, y0, y1)
    T = typeof(intercept)
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    accumulator = (zero(T), zero(T), zero(T))
    exit = entry = (zero(T), zero(T))
    exits = entries = false
    for k in 1:4
        a = corners[k]
        b = corners[k == 4 ? 1 : k + 1]
        value_a = muladd(normal_x, a[1], muladd(normal_y, a[2], -intercept))
        value_b = muladd(normal_x, b[1], muladd(normal_y, b[2], -intercept))
        if value_a <= zero(T)
            if value_b <= zero(T)
                accumulator = _green_segment(accumulator, a, b)
            else
                exit = _plic_crossing(a, b, value_a, value_b)
                exits = true
                accumulator = _green_segment(accumulator, a, exit)
            end
        elseif value_b <= zero(T)
            entry = _plic_crossing(a, b, value_a, value_b)
            entries = true
            accumulator = _green_segment(accumulator, entry, b)
        end
    end
    chord = (zero(T), zero(T))
    if exits && entries
        accumulator = _green_segment(accumulator, exit, entry)
        chord = (entry[1] - exit[1], entry[2] - exit[2])
    end
    twice_area, moment_x, moment_y = accumulator
    return twice_area / 2, moment_x / 6, moment_y / 6, chord
end

"""Interface length, phase-centroid distance and component weight of the
staggered control volume of `face`.

The control volume is the one of the staggered liquid fraction: half of each
adjacent cell, restricted to the tangential strip of the face (the whole
strip in the face-normal direction, weighted by one half, so its liquid area
is the staggered fraction times its volume). The PLIC plane of each mixed cell
is clipped to its strip; `|Gamma_f|` is half the sum of the two chord
lengths, the interface normal the chord-length weighted mean of the two plane
normals, and `delta_f` the distance of the liquid and gas centroids along it,
bounded below by a quarter of the mean cell width along the face normal.
Between a pure liquid and a pure gas cell the interface is the face itself
(length `face.area`, normal along the face normal, `delta_f` the cell-centre
distance). An embedded solid is ignored in this geometry (the masses and the
fraction do account for it). Returns zeros if the volume holds no interface
or only one phase.
"""
@inline function _interphase_contact(field::InterfaceField{T}, face, tangential) where {T}
    mesh = field.mesh
    axis = face.axis
    tangent = 3 - axis
    half = T(0.5)
    liquid = (zero(T), zero(T), zero(T))  # area, first moments
    total = (zero(T), zero(T), zero(T))
    chord_length = zero(T)
    normal_sum = (zero(T), zero(T))
    width = zero(T)
    mixed_cells = 0
    for (side, ref) in ((-1, face.minus), (1, face.plus))
        block = mesh.blocks[ref.block]
        spacing = (T(block.spacing[1]), T(block.spacing[2]))
        width += spacing[axis]
        lo, hi = _face_local_tangent_interval(mesh, ref, face)
        lo, hi = T(lo), T(hi)
        # Cell centre relative to the face centre.
        offset_axis = side * spacing[axis] / 2
        offset_tangent = -(lo + hi) / 2 * spacing[tangent]
        offset = axis == 1 ? (offset_axis, offset_tangent) :
            (offset_tangent, offset_axis)
        x0, x1, y0, y1 = axis == 1 ? (-half, half, lo, hi) : (lo, hi, -half, half)
        jacobian = spacing[1] * spacing[2]
        local_area = (x1 - x0) * (y1 - y0)
        local_center = ((x0 + x1) / 2, (y0 + y1) / 2)
        cell_area = half * jacobian * local_area
        total = (
            total[1] + cell_area,
            total[2] + cell_area * muladd(local_center[1], spacing[1], offset[1]),
            total[3] + cell_area * muladd(local_center[2], spacing[2], offset[2]),
        )
        fraction = _conditional_fraction(field, ref)
        fraction <= zero(T) && continue
        if fraction >= one(T)
            liquid = (
                liquid[1] + cell_area,
                liquid[2] + cell_area * muladd(local_center[1], spacing[1], offset[1]),
                liquid[3] + cell_area * muladd(local_center[2], spacing[2], offset[2]),
            )
            continue
        end
        mixed_cells += 1
        normal_x = _get(field.normal_x, ref)
        normal_y = _get(field.normal_y, ref)
        area, moment_x, moment_y, chord = _plic_rectangle_clip(
            normal_x, normal_y, _get(field.intercept, ref), x0, x1, y0, y1,
        )
        area_weight = half * jacobian
        liquid = (
            liquid[1] + area_weight * area,
            liquid[2] + area_weight * muladd(moment_x, spacing[1], area * offset[1]),
            liquid[3] + area_weight * muladd(moment_y, spacing[2], area * offset[2]),
        )
        piece = half * hypot(chord[1] * spacing[1], chord[2] * spacing[2])
        piece > zero(T) || continue
        # Plane normal in physical coordinates, pointing into the gas.
        gradient = (normal_x / spacing[1], normal_y / spacing[2])
        norm = hypot(gradient[1], gradient[2])
        chord_length += piece
        normal_sum = (
            muladd(piece, gradient[1] / norm, normal_sum[1]),
            muladd(piece, gradient[2] / norm, normal_sum[2]),
        )
    end
    gas_area = total[1] - liquid[1]
    nothing_coupled = (zero(T), zero(T), zero(T))
    liquid[1] > zero(T) && gas_area > zero(T) || return nothing_coupled
    if iszero(mixed_cells)
        # Pure liquid next to pure gas: the face is the interface.
        chord_length = T(face.area)
        normal_sum = axis == 1 ? (chord_length, zero(T)) : (zero(T), chord_length)
    end
    chord_length > zero(T) || return nothing_coupled
    normal_norm = hypot(normal_sum[1], normal_sum[2])
    minimum_distance = width / 8
    normal_norm > zero(T) ||
        return (chord_length, minimum_distance, tangential ? zero(T) : one(T))
    normal = (normal_sum[1] / normal_norm, normal_sum[2] / normal_norm)
    separation = (
        (total[2] - liquid[2]) / gas_area - liquid[2] / liquid[1],
        (total[3] - liquid[3]) / gas_area - liquid[3] / liquid[1],
    )
    distance = max(
        abs(muladd(separation[1], normal[1], separation[2] * normal[2])),
        minimum_distance,
    )
    return chord_length, distance, tangential ? abs(normal[tangent]) : one(T)
end

"""Decay rate `lambda = W (1/m_liquid + 1/m_gas)` of the phase-velocity jump
on `face`, or zero if the face is not mixed (see [`InterphaseDrag`](@ref))."""
@inline function _interphase_drag_rate(
        field::InterfaceField{T}, face, volume, density, viscosity, averaging,
        tangential,
    ) where {T}
    volume > zero(T) || return zero(T)
    fraction = _staggered_liquid_fraction(field, field.geometry, face)
    zero(T) < fraction < one(T) || return zero(T)
    chord_length, distance, weight = _interphase_contact(field, face, tangential)
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
            rate = _interphase_drag_rate(
                interface, face, volumes[index], density, viscosity,
                averaging, tangential,
            )
            rate > zero(T) || continue
            fraction = _staggered_liquid_fraction(interface, interface.geometry, face)
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
