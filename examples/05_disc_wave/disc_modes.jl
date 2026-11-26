# Exact standing waves of the wave equation u_tt = c²Δu on the disc r < R
# with u = 0 on the circle: the reference solution of the disc wave example,
# shared by disc_wave.jl, its figure script and test/integration/disc_wave.jl.
# Plain Julia and SpecialFunctions; it knows nothing of the mesh or solver.
using SpecialFunctions: besselj

"""
    bessel_zero(m, k)

The `k`-th positive zero `j_{m,k}` of the Bessel function `J_m`. Sign changes
are bracketed on a grid of spacing `1/20`, starting where `J_m` is still
positive (consecutive zeros are more than two apart), and bisected to
round-off.
"""
function bessel_zero(m::Integer, k::Integer)
    m >= 0 && k >= 1 || throw(ArgumentError("bessel_zero needs m >= 0 and k >= 1"))
    found = 0
    left = max(float(m), 1 / 20)
    left_value = besselj(m, left)
    while true
        right = left + 1 / 20
        right_value = besselj(m, right)
        if iszero(right_value) || signbit(right_value) != signbit(left_value)
            found += 1
            if found == k
                iszero(right_value) && return right
                while true
                    middle = (left + right) / 2
                    left < middle < right || return middle
                    middle_value = besselj(m, middle)
                    iszero(middle_value) && return middle
                    if signbit(middle_value) == signbit(left_value)
                        left, left_value = middle, middle_value
                    else
                        right = middle
                    end
                end
            end
        end
        left, left_value = right, right_value
    end
    return
end

"""
    DiscMode(axis, m, k; sine=false, amplitude=1)

A standing Dirichlet eigenmode of the component `axis` (`1` for `x`, `2` for
`y`) on the disc `r < R`, in polar coordinates about its centre:

    amplitude * J_m(j_{m,k} r/R) * cos(mθ) * cos(ωt),    ω = c j_{m,k}/R,

with `sin(mθ)` for `sine=true`. It vanishes on the circle and solves the
wave equation `u_tt = c²Δu` for all time, starting at rest.
"""
struct DiscMode
    axis::Int
    m::Int
    k::Int
    sine::Bool
    amplitude::Float64
    root::Float64
end

function DiscMode(axis, m, k; sine = false, amplitude = 1.0)
    axis in (1, 2) || throw(ArgumentError("a disc mode belongs to axis 1 or 2"))
    return DiscMode(axis, m, k, sine, amplitude, bessel_zero(m, k))
end

# u_x: the axisymmetric fundamental (0,1) and the (2,1) mode; u_y: the (1,1)
# mode and the (3,2) mode, which has a nodal circle.
const DISC_WAVE_MODES = (
    DiscMode(1, 0, 1), DiscMode(1, 2, 1; amplitude = 0.6),
    DiscMode(2, 1, 1; sine = true), DiscMode(2, 3, 2; amplitude = 0.8),
)

"""Angular frequency `c j_{m,k}/R` of `mode`."""
mode_frequency(mode::DiscMode; radius = 1.0, speed = 1.0) = speed * mode.root / radius

"""Spatial shape of `mode`, with its amplitude, at `(x, y)` about the disc centre."""
function mode_shape(mode::DiscMode, x, y, radius)
    angle = atan(y, x)
    angular = mode.sine ? sin(mode.m * angle) : cos(mode.m * angle)
    return mode.amplitude * besselj(mode.m, mode.root * hypot(x, y) / radius) * angular
end

"""
    disc_wave_exact(modes, x, y, time, axis; radius=1, speed=1)

Exact displacement of component `axis` and its rate `∂u/∂t` at `(x, y)`: the
sum of the `modes` of that component.
"""
function disc_wave_exact(modes, x, y, time, axis; radius = 1.0, speed = 1.0)
    displacement = 0.0
    rate = 0.0
    for mode in modes
        mode.axis == axis || continue
        omega = mode_frequency(mode; radius, speed)
        shape = mode_shape(mode, x, y, radius)
        displacement += shape * cos(omega * time)
        rate -= omega * shape * sin(omega * time)
    end
    return displacement, rate
end
