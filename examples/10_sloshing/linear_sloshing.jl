using LinearAlgebra: Diagonal, LAPACK, Symmetric, eigen

"""
    sloshing_tank(; baffle_height=0, baffle_thickness=0.01)

Tank, fluids, baffle and wall-probe width shared by the example, its figures
and the linear theory: a 1 m × 1 m closed tank holding 0.5 m of water under
air, SI units. The probes average over the `1/64 m` strip next to a wall.
"""
function sloshing_tank(; baffle_height = 0.0, baffle_thickness = 0.01)
    return (;
        length = 1.0, height = 1.0, depth = 0.5,
        density = (1000.0, 1.2), dynamic_viscosity = (1.0e-3, 1.8e-5),
        surface_tension = 0.0728, gravity = 9.81,
        baffle_height = Float64(baffle_height),
        baffle_thickness = Float64(baffle_thickness), probe_width = 1 / 64,
    )
end

# Restoring stiffness and two-layer inertia of a surface mode with liquid and
# gas Dirichlet-to-Neumann eigenvalues `κl` and `κg` and wavenumber `k`:
# (ρl/κl + ρg/κg) a'' + ((ρl - ρg) g + σk²) a = (ρl - ρg) ξ f(t).
function _mode_coefficients(tank, κl, κg, k)
    ρl, ρg = tank.density
    inertia = ρl / κl + ρg / κg
    frequency = sqrt(((ρl - ρg) * tank.gravity + tank.surface_tension * k^2) / inertia)
    return frequency, (ρl - ρg) / inertia
end

_gas_eigenvalue(tank, k) = k * tanh(k * (tank.height - tank.depth))

"""
    plain_tank_modes(tank; count=100)

Antisymmetric linear sloshing modes `cos(nπx/L)`, `n = 1, 3, …`, of the
tank without a baffle. Each entry holds the natural frequency, the mean of the
mode over the left-wall probe strip, the response coefficient of the uniform
horizontal body force and the component of the initial shape `cos(πx/L)`.
"""
function plain_tank_modes(tank; count = 100)
    return map(1:count) do m
        n = 2m - 1
        k = n * pi / tank.length
        frequency, inertia_ratio = _mode_coefficients(
            tank, k * tanh(k * tank.depth), _gas_eigenvalue(tank, k), k,
        )
        # Cosine coefficient of x - L/2 on [0, L].
        projection = -4tank.length / (n * pi)^2
        (;
            frequency, probe = sinc(k * tank.probe_width / pi),
            forcing = inertia_ratio * projection, initial = Float64(m == 1),
        )
    end
end

# Mean over [0, width] of the piecewise-linear interpolant of samples v at x.
function _strip_mean(x, v, width)
    total = zero(eltype(v))
    for i in 1:(length(x) - 1)
        x[i] < width || break
        right = min(x[i + 1], width)
        value = v[i] + (v[i + 1] - v[i]) * (right - x[i]) / (x[i + 1] - x[i])
        total += (right - x[i]) * (v[i] + value) / 2
    end
    return total / width
end

"""
    baffled_tank_modes(tank; count=20, spacing=0.0025)

Antisymmetric linear sloshing modes of the tank with its central bottom-mounted
baffle, from a bilinear finite-element Steklov eigenproblem for the velocity
potential on the left half of the liquid (zero potential on the symmetry line
above the baffle). The surface nodes are condensed by a banded LU
factorization of the interior block. The gas layer and surface tension enter
through the plain-tank wavenumber of the same mode number, a correction of
about 0.1% in frequency. The probe value is the mean over the left-wall probe
strip. The baffle edges must lie on the `spacing` lattice; a zero
`baffle_height` gives the plain tank.
"""
function baffled_tank_modes(tank; count = 20, spacing = 0.0025)
    nx = round(Int, tank.length / 2spacing)
    ny = round(Int, tank.depth / spacing)
    face = round(Int, (tank.length - tank.baffle_thickness) / 2spacing)
    top = round(Int, tank.baffle_height / spacing)
    aligned(value, cells) = isapprox(value, cells * spacing; atol = 1.0e-12)
    aligned(tank.length / 2, nx) && aligned(tank.depth, ny) &&
        aligned((tank.length - tank.baffle_thickness) / 2, face) &&
        aligned(tank.baffle_height, top) && 0 <= top < ny || throw(
        ArgumentError("the baffle must lie on the finite-element lattice and below the surface"),
    )
    node(i, j) = j * (nx + 1) + i + 1
    total = (nx + 1) * (ny + 1)
    bandwidth = nx + 2
    # Upper band storage: band[bandwidth + 1 + a - b, b] = K[a, b] for a <= b.
    band = zeros(bandwidth + 1, total)
    element = [4 -1 -2 -1; -1 4 -1 -2; -2 -1 4 -1; -1 -2 -1 4] ./ 6
    active = falses(total)
    for j in 0:(ny - 1), i in 0:(nx - 1)
        i >= face && j < top && continue
        nodes = (node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1))
        for (p, a) in enumerate(nodes), (q, b) in enumerate(nodes)
            a <= b && (band[bandwidth + 1 + a - b, b] += element[p, q])
        end
        active[collect(nodes)] .= true
    end
    # Nodes inside the baffle and the antisymmetry line become identity rows.
    for a in 1:total
        symmetry = a % (nx + 1) == 0 && (a - 1) ÷ (nx + 1) >= top
        active[a] && !symmetry && continue
        for b in max(1, a - bandwidth):min(total, a + bandwidth)
            band[bandwidth + 1 + min(a, b) - max(a, b), max(a, b)] = 0
        end
        band[bandwidth + 1, a] = 1
    end
    interior = total - (nx + 1)
    surface = nx + 1
    coupling = zeros(interior, surface)
    stiffness = zeros(surface, surface)
    for b in (interior + 1):total, a in max(1, b - bandwidth):b
        value = band[bandwidth + 1 + a - b, b]
        if a <= interior
            coupling[a, b - interior] = value
        else
            stiffness[a - interior, b - interior] = value
            stiffness[b - interior, a - interior] = value
        end
    end
    # LAPACK general band storage of the interior block (its LU needs `bandwidth`
    # extra rows): A[a, b] is stored at factor[2bandwidth + 1 + a - b, b].
    factor = zeros(3bandwidth + 1, interior)
    for b in 1:interior, a in max(1, b - bandwidth):b
        value = band[bandwidth + 1 + a - b, b]
        factor[2bandwidth + 1 + a - b, b] = value
        factor[2bandwidth + 1 + b - a, a] = value
    end
    _, pivots = LAPACK.gbtrf!(bandwidth, bandwidth, interior, factor)
    condensed = LAPACK.gbtrs!(
        'N', bandwidth, bandwidth, interior, factor, pivots, copy(coupling),
    )
    # Drop the surface node on the symmetry line, where the potential is zero.
    keep = 1:(surface - 1)
    schur = Symmetric((stiffness - coupling' * condensed)[keep, keep])
    weights = fill(spacing, surface - 1)
    weights[1] = spacing / 2
    decomposition = eigen(schur, Matrix(Diagonal(weights)))
    x = (0:(surface - 2)) .* spacing
    shape = cos.(pi .* x ./ tank.length)
    return map(1:count) do m
        κ = decomposition.values[m]
        v = decomposition.vectors[:, m]
        norm = sum(weights .* v .^ 2)
        k = (2m - 1) * pi / tank.length
        frequency, inertia_ratio = _mode_coefficients(
            tank, κ, _gas_eigenvalue(tank, k), k,
        )
        (;
            frequency, probe = _strip_mean(x, v, tank.probe_width),
            forcing = inertia_ratio * sum(weights .* (x .- tank.length / 2) .* v) / norm,
            initial = sum(weights .* shape .* v) / norm,
        )
    end
end

"""
    modal_elevation(time, modes; initial_amplitude=0, forcing_amplitude=0,
                    forcing_frequency=1)

Linear probe elevation for an initial surface `initial_amplitude*cos(πx/L)`
plus the response to the tank displacement `-A sin(ωt)` with
`A = forcing_amplitude` and `ω = forcing_frequency`. Liquid and gas start at
rest while the tank starts impulsively with velocity `-Aω`: in the tank frame
an initial relative velocity `Aω` and the body force `-Aω² sin(ωt)`. A mode
with forcing coefficient `q` and frequency `ωn` then responds as
`qAω(ωn sin(ωn t) - ω sin(ωt))/(ωn² - ω²)`.
"""
function modal_elevation(
        time, modes; initial_amplitude = 0.0, forcing_amplitude = 0.0,
        forcing_frequency = 1.0,
    )
    ω = forcing_frequency
    return sum(modes) do mode
        ωn = mode.frequency
        # The resonant limit replaces the quotient when ωn ≈ ω.
        response = abs(ωn - ω) > 1.0e-8ωn ?
            ω * (ωn * sin(ωn * time) - ω * sin(ω * time)) / (ωn^2 - ω^2) :
            (sin(ω * time) + ω * time * cos(ω * time)) / 2
        mode.probe * (
            initial_amplitude * mode.initial * cos(ωn * time) +
                mode.forcing * forcing_amplitude * response
        )
    end
end
