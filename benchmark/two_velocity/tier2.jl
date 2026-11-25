# Tier 2: short time-dependent flows (local only, about five minutes per method).
# Every two-velocity method and the one-velocity model as the reference run
# the same cases: hydrostatic rest, the static drop in both label orders, the
# capillary standing wave against Prosperetti's solution, the
# Kelvin–Helmholtz linear growth (grid-line, shifted, larger-amplitude and
# moving-frame members), the normally translated sheet (Galilean), and the
# exchange of the phase labels. The setups follow the integration tests and
# the MM study harnesses; the KH problem reuses its integration-test fixture.

include(joinpath(@__DIR__, "..", "..", "test", "integration", "interfacial_instability_problem.jl"))
include(joinpath(@__DIR__, "..", "..", "examples", "06_capillary_wave", "prosperetti_amplitude.jl"))

phase_fields(velocity) = velocity isa FaceVelocity ? (velocity,) : velocity

max_speed(velocity) = maximum(finite_maximum(field.interior) for field in phase_fields(velocity))

"""Largest value on a mixed face (mean conditional fraction of its cells in
(0, 1)) whose phase aperture exceeds 0.05: a resolved current, not the
intrinsic velocity of a vanishing phase (the static-drop test's gate)."""
function resolved_speed(velocity, interface)
    apertures = TP._current_phase_face_apertures!(interface)
    largest = 0.0
    for (index, face) in pairs(interface.mesh.faces)
        c = (TP._conditional_fraction(interface, face.minus) + TP._conditional_fraction(interface, face.plus)) / 2
        0 < c < 1 || continue
        for (phase, field) in enumerate(phase_fields(velocity))
            # The single field of the one-velocity model is gated by the
            # aperture of the larger phase.
            aperture = velocity isa FaceVelocity ? max(apertures[1][index], apertures[2][index]) : apertures[phase][index]
            aperture > 0.05 || continue
            value = field.interior[index]
            isfinite(value) && (largest = max(largest, abs(value)))
        end
    end
    return largest
end

kinetic_energy(state) = diagnostic_value(KineticEnergy(), state)

"""Per-step speeds and kinetic energy of a `simulate!` run."""
mutable struct SpeedObserver <: AbstractObserver
    times::Vector{Float64}
    raw::Vector{Float64}
    resolved::Vector{Float64}
    energy::Vector{Float64}
end

SpeedObserver() = SpeedObserver(Float64[], Float64[], Float64[], Float64[])

function TwoPhaseAMR.observe!(observer::SpeedObserver, event::Symbol, state, info)
    event === :step || return nothing
    push!(observer.times, state.time)
    push!(observer.raw, max_speed(state.velocity))
    push!(observer.resolved, resolved_speed(state.velocity, state.interface))
    push!(observer.energy, kinetic_energy(state))
    return nothing
end

window_maximum(values, times, keep) = maximum((v for (v, t) in zip(values, times) if keep(t)); init = 0.0)

two_phase_velocity(m::Method, liquid, gas, levelset) = two_velocity(m) ? (liquid, gas) :
    (x, y, t, axis) -> (levelset(x, y) <= 0 ? liquid : gas)(x, y, t, axis)

# ---------------------------------------------------------------- hydrostatic
METRIC_DESCRIPTIONS["hydrostatic/max_u"] = "largest value over 50 steps of Δt = 1e-3, exact rest"

"""Flat pool at y = ½ + h/3 under gravity g = 1, periodic in x, free-slip
walls, DR 1000; `swap` labels the light phase as phase 1."""
function tier2_hydrostatic!(suite, m)
    n = 32
    for swap in (false, true)
        case_name = "pool DR=1000 " * (swap ? "swapped" : "normal")
        guarded(suite, 2, "hydrostatic", case_name, n, m) do
            y0 = 0.5 + 1 / (3n)
            sign = swap ? -1 : 1
            mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (8, 8), periodic = (true, false))
            interface = InterfaceField(mesh, (x, y) -> sign * (y - y0); lipschitz = 1)
            physics = TwoPhase(
                interface; density = swap ? (1.0, 1000.0) : (1000.0, 1.0), dynamic_viscosity = (0.0, 0.0),
                surface_tension = 0.0, gravity = (0.0, -1.0),
            )
            flow = FlowSimulation(mesh, physics; velocity_model = velocity_model(m), boundary = FreeSlip())
            peak = 0.0
            for _ in 1:50
                checked(step!(flow, 1.0e-3; tolerance = 1.0e-11, max_cycles = 1000))
                peak = max(peak, max_speed(flow.velocity))
            end
            record!(suite, 2, "hydrostatic", case_name, n, m, "max_u", peak; atol = 1.0e-9)
        end
    end
    return
end

# ---------------------------------------------------------------- static drop
METRIC_DESCRIPTIONS["static drop/late_u"] = "largest value for t > T/2 (all present values)"
METRIC_DESCRIPTIONS["static drop/late_u_resolved"] = "largest value for t > T/2 on mixed faces with phase aperture > 0.05"
METRIC_DESCRIPTIONS["static drop/K_max"] = "largest kinetic energy (staggered phase masses)"
METRIC_DESCRIPTIONS["static drop/K_growth"] = "largest kinetic energy of the last third over that of the first third"
METRIC_DESCRIPTIONS["static drop/steps"] = "time steps of `simulate!` (package step controller)"

"""Inviscid static drop R = 0.2 at the box centre, σ = 1 (CGFM), free-slip
walls, n = 32, t = 0.4 (4.5 capillary times); `swap` labels the outer fluid
as phase 1 (level set and densities exchanged)."""
function tier2_static_drop!(suite, m)
    n, final_time = 32, 0.4
    for ratio in (1.0, 1000.0), swap in (false, true)
        case_name = "R=0.2 DR=$(round(Int, ratio)) " * (swap ? "swapped" : "normal")
        guarded(suite, 2, "static drop", case_name, n, m) do
            sign = swap ? -1 : 1
            mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (8, 8), periodic = (false, false))
            interface = InterfaceField(mesh, (x, y) -> sign * (hypot(x - 0.5, y - 0.5) - 0.2); lipschitz = 1)
            density = swap ? (1 / ratio, 1.0) : (1.0, 1 / ratio)
            physics = TwoPhase(
                interface; density, dynamic_viscosity = (0.0, 0.0),
                surface_tension = SurfaceTension(1.0; method = CGFM()),
            )
            flow = FlowSimulation(mesh, physics; velocity_model = velocity_model(m), boundary = FreeSlip())
            observer = SpeedObserver()
            result = simulate!(flow, final_time; tolerance = 1.0e-9, max_cycles = 1000, observer)
            late(t) = t > final_time / 2
            early_energy = window_maximum(observer.energy, observer.times, t -> t <= final_time / 3)
            late_energy = window_maximum(observer.energy, observer.times, t -> t >= 2final_time / 3)
            record!(suite, 2, "static drop", case_name, n, m, "late_u", window_maximum(observer.raw, observer.times, late))
            record!(
                suite, 2, "static drop", case_name, n, m, "late_u_resolved",
                window_maximum(observer.resolved, observer.times, late),
            )
            record!(suite, 2, "static drop", case_name, n, m, "K_max", maximum(observer.energy))
            record!(suite, 2, "static drop", case_name, n, m, "K_growth", late_energy / early_energy)
            record!(suite, 2, "static drop", case_name, n, m, "steps", result.total_steps; goal = :info)
        end
    end
    return
end

# ---------------------------------------------------------------- capillary wave
METRIC_DESCRIPTIONS["capillary wave/error"] = "time-L2 error of the normalised amplitude against Prosperetti over a quarter period"
METRIC_DESCRIPTIONS["capillary wave/steps"] = "time steps of `simulate!` (package step controller)"

"""Cosine moment of the VOF field: the height amplitude on a unit period."""
function cosine_amplitude(interface)
    mesh = interface.mesh
    amplitude = 0.0
    for ref in TP._cell_references(mesh)
        x, _ = cellcenter(mesh, ref.block, ref.i, ref.j)
        hx, hy = mesh.blocks[ref.block].spacing
        amplitude += TP._get(interface.fraction, ref) * hy * (sinpi(2 * (x + hx / 2)) - sinpi(2 * (x - hx / 2))) / pi
    end
    return amplitude
end

mutable struct AmplitudeObserver <: AbstractObserver
    interface::Any
    times::Vector{Float64}
    amplitudes::Vector{Float64}
end

function TwoPhaseAMR.observe!(observer::AmplitudeObserver, event::Symbol, state, info)
    event === :step || return nothing
    push!(observer.times, state.time)
    push!(observer.amplitudes, cosine_amplitude(observer.interface))
    return nothing
end

"""The thesis standing capillary wave of the integration test (λ = 1, depth
1.5 on each side, mean height 0.25 on a cell edge, a₀ = λ/100, La = 3000 with
equal kinematic viscosities, CGFM, free slip) over a quarter period."""
function tier2_capillary_wave!(suite, m)
    for ratio in (1.0, 1000.0), n in (16, 32, 64)
        case_name = "La=3000 DR=$(round(Int, ratio))"
        guarded(suite, 2, "capillary wave", case_name, n, m) do
            rho_l, rho_g = 1.0, 1 / ratio
            nu = sqrt(1.0 * rho_l / 3000)
            mesh = BlockTreeMesh(;
                base_mesh = (n, 3n), block_size = (8, 8), origin = (0.0, -1.5), extent = (1.0, 3.0),
                periodic = (true, false),
            )
            levelset(x, y) = y - 0.25 - 0.01 * cospi(2x)
            interface = InterfaceField(mesh, levelset; lipschitz = 1)
            physics = TwoPhase(
                interface; density = (rho_l, rho_g), dynamic_viscosity = (rho_l * nu, rho_g * nu),
                surface_tension = SurfaceTension(1.0; method = CGFM()),
            )
            rest = (x, y, t, axis) -> 0.0
            flow = FlowSimulation(
                mesh, physics; velocity_model = velocity_model(m), boundary = FreeSlip(),
                velocity = two_velocity(m) ? (rest, rest) : nothing,
            )
            omega0 = sqrt((2pi)^3 / (rho_l + rho_g))
            initial = cosine_amplitude(interface)
            observer = AmplitudeObserver(interface, [0.0], [initial])
            result = simulate!(
                flow, 0.25 * 2pi / omega0; tolerance = 1.0e-9, max_cycles = two_velocity(m) ? 1000 : 300, observer,
            )
            errors = map(observer.times, observer.amplitudes) do t, a
                a / initial - prosperetti_amplitude(
                    t; liquid_density = rho_l, gas_density = rho_g, kinematic_viscosity = nu,
                    gravity = 0.0, surface_tension = 1.0, wavenumber = 2pi,
                )
            end
            times = observer.times
            integral = sum((times[i + 1] - times[i]) * (errors[i]^2 + errors[i + 1]^2) / 2 for i in 1:(length(times) - 1))
            record!(suite, 2, "capillary wave", case_name, n, m, "error", sqrt(integral / (last(times) - first(times))))
            record!(suite, 2, "capillary wave", case_name, n, m, "steps", result.total_steps; goal = :info)
        end
    end
    return
end

# ---------------------------------------------------------------- Kelvin–Helmholtz
METRIC_DESCRIPTIONS["Kelvin-Helmholtz/error"] = "normalised time-L2 error of the complex mode amplitude against linear theory"
METRIC_DESCRIPTIONS["Kelvin-Helmholtz/grid_mean"] = "mean error of the five grid-line members (amplitude ×1, ×(1 ± 0.0025), ×(1 ± 0.005))"
METRIC_DESCRIPTIONS["Kelvin-Helmholtz/grid_spread"] = "largest minus smallest error of the grid-line members"

"""Inviscid KH (liquid below, DR 1000, We 0.5, n = 32 per wavelength, LW
advection, CFL 0.35) after `two_velocity_instability_setup`, with the mean
interface at `height` cell spacings above the grid line y = 0, the amplitude
scaled by `factor` and the whole flow moving at `mean` (a Galilean boost)."""
function kh_run(m::Method, n; height = 0.0, amplitude = DEFAULT_WAVE_AMPLITUDE, mean = 0.0)
    dispersion = instability_dispersion(;
        density_ratio = 1000.0, weber = 0.5, bond = 0.0, mean_interface_velocity = mean, liquid_above = false,
    )
    mesh = _instability_mesh(n)
    k = dispersion.k
    y0 = height / n
    levelset(x, y) = y - y0 - amplitude * cos(k * x)
    interface = InterfaceField(mesh, levelset; lipschitz = hypot(1, amplitude * k), reconstruction_method = :lhf)
    phase_speed = dispersion.mean_frequency / k
    liquid_base, gas_base = dispersion.tangential_velocity
    liquid = _phase_velocity_initializer(liquid_base, amplitude * (phase_speed - liquid_base), false, y0, k)
    gas = _phase_velocity_initializer(gas_base, amplitude * (phase_speed - gas_base), true, y0, k)
    physics = TwoPhase(
        interface; density = dispersion.density, dynamic_viscosity = (0.0, 0.0),
        surface_tension = dispersion.surface_tension, gravity = (0.0, dispersion.vertical_gravity),
    )
    flow = FlowSimulation(
        mesh, physics; velocity_model = velocity_model(m), velocity = two_phase_velocity(m, liquid, gas, levelset),
        boundary = _free_slip_boundaries(), numerics = NavierStokesOptions(advection = AdvectionOptions(; scheme = LaxWendroff())),
    )
    mode() = _instability_wave_amplitude(interface; liquid_above = false)
    add_diagnostic!(flow, :amplitude_real, state -> real(mode()))
    add_diagnostic!(flow, :amplitude_imaginary, state -> imag(mode()))
    case = (; flow, dispersion, initial_amplitude = amplitude, default_final_time = _default_instability_final_time(dispersion))
    result = _run_instability!(
        case, _instability_target(case, nothing); cfl = 0.35, tolerance = 1.0e-7, max_cycles = 3000, timestep_factor = 1.0,
    )
    reference = [instability_linear_amplitude(t, dispersion; initial_amplitude = amplitude) for t in result.times]
    return normalized_amplitude_error(result.times, result.amplitudes, reference), flow.steps
end

const KH_MEMBERS = [
    ("grid ×1", (;)),
    ("grid ×0.9975", (; amplitude = 0.9975DEFAULT_WAVE_AMPLITUDE)),
    ("grid ×1.0025", (; amplitude = 1.0025DEFAULT_WAVE_AMPLITUDE)),
    ("grid ×0.995", (; amplitude = 0.995DEFAULT_WAVE_AMPLITUDE)),
    ("grid ×1.005", (; amplitude = 1.005DEFAULT_WAVE_AMPLITUDE)),
    ("shift 0.333h", (; height = 1 / 3)),
    ("amplitude 0.04", (; amplitude = 0.04)),
    ("moving U=2", (; mean = 2.0)),
]

function tier2_kelvin_helmholtz!(suite, m)
    # One velocity field cannot carry the sheet's velocity jump: its error
    # against the two-fluid linear theory is O(1) (7.7 for the grid member), so
    # the one-velocity model is not a reference here.
    two_velocity(m) || return
    n = 32
    grid = Float64[]
    for (member, parameters) in KH_MEMBERS
        case_name = "DR=1000 We=0.5 " * member
        guarded(suite, 2, "Kelvin-Helmholtz", case_name, n, m) do
            error, steps = kh_run(m, n; parameters...)
            record!(suite, 2, "Kelvin-Helmholtz", case_name, n, m, "error", error)
            record!(suite, 2, "Kelvin-Helmholtz", case_name, n, m, "steps", steps; goal = :info)
            startswith(member, "grid") && push!(grid, error)
        end
    end
    if length(grid) == 5
        record!(suite, 2, "Kelvin-Helmholtz", "DR=1000 We=0.5 grid ensemble", n, m, "grid_mean", sum(grid) / 5)
        record!(
            suite, 2, "Kelvin-Helmholtz", "DR=1000 We=0.5 grid ensemble", n, m, "grid_spread",
            maximum(grid) - minimum(grid); goal = :info,
        )
    end
    return
end

# ---------------------------------------------------------------- translation
METRIC_DESCRIPTIONS["translation/tangential"] = "largest error of a tangential value with phase aperture > 0 (exact ∓½)"
METRIC_DESCRIPTIONS["translation/normal"] = "largest error of a normal value with phase aperture > 0 (exact 1)"
METRIC_DESCRIPTIONS["translation/fraction"] = "largest error of the volume fraction against the translated layer"

"""A material sheet translated normally at speed 1 with tangential streams
∓½ (an exact inviscid solution), periodic in both directions (two
interfaces), DR 1000, until t = 2h: `film` gives the light layer 1.5 cells,
`resolved` half the box. Only values with a phase aperture (all of them for
the one-velocity model: aperture 1) are compared."""
function tier2_translation!(suite, m)
    n = 32
    for film in (false, true)
        case_name = "sheet DR=1000 " * (film ? "film 1.5h" : "resolved")
        guarded(suite, 2, "translation", case_name, n, m) do
            h = 1 / n
            yc, halfwidth = 0.5 + h / 3, film ? 0.75h : 0.25
            mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (8, 8), periodic = (true, true))
            distance(x, y) = halfwidth - abs(mod(y - yc + 0.5, 1.0) - 0.5)   # positive inside the light layer
            interface = InterfaceField(mesh, distance; lipschitz = 1)
            heavy = (x, y, t, a) -> a == 1 ? -0.5 : 1.0
            light = (x, y, t, a) -> a == 1 ? 0.5 : 1.0
            physics = TwoPhase(interface; density = (1000.0, 1.0), dynamic_viscosity = (0.0, 0.0), surface_tension = 0.0)
            flow = FlowSimulation(
                mesh, physics; velocity_model = velocity_model(m), boundary = FreeSlip(),
                velocity = two_phase_velocity(m, heavy, light, distance),
            )
            final_time = 2h
            tangential = normal = fraction = 0.0
            while flow.time < final_time - 16eps(final_time)
                dt = min(stable_timestep(flow; cfl = 0.35), final_time - flow.time)
                checked(step!(flow, dt; tolerance = 1.0e-10, max_cycles = 1000))
                apertures = TP._current_phase_face_apertures!(interface)
                for phase in 1:2
                    field = two_velocity(m) ? flow.velocity[phase] : flow.velocity
                    threshold = two_velocity(m) ? 1.0e-12 : 1 - 1.0e-12
                    for (i, face) in pairs(mesh.faces)
                        apertures[phase][i] > threshold || continue
                        v = field.interior[i]
                        if face.axis == 1
                            tangential = max(tangential, abs(v - (phase == 1 ? -0.5 : 0.5)))
                        else
                            normal = max(normal, abs(v - 1))
                        end
                    end
                end
                centre = yc + flow.time
                for (b, block) in pairs(mesh.blocks)
                    values = interior(interface.fraction, b)
                    hy = block.spacing[2]
                    for j in axes(values, 2)
                        _, y = cellcenter(mesh, b, 1, j)
                        light_part = sum(
                            max(0.0, min(y + hy / 2, centre + halfwidth + s) - max(y - hy / 2, centre - halfwidth + s))
                                for s in -1:1
                        ) / hy
                        for i in axes(values, 1)
                            fraction = max(fraction, abs(values[i, j] - (1 - light_part)))
                        end
                    end
                end
            end
            record!(suite, 2, "translation", case_name, n, m, "tangential", tangential; atol = 1.0e-9)
            record!(suite, 2, "translation", case_name, n, m, "normal", normal; atol = 1.0e-9)
            record!(suite, 2, "translation", case_name, n, m, "fraction", fraction; atol = 1.0e-9)
        end
    end
    return
end

# ---------------------------------------------------------------- label symmetry
METRIC_DESCRIPTIONS["label symmetry/velocity"] = "largest difference of the exchanged values over their largest value (Inf: a value present in one labelling only)"
METRIC_DESCRIPTIONS["label symmetry/fraction"] = "largest |C + C' − 1| of the two labellings"

"""Two flows that differ only by the phase labels: level set negated,
densities and viscosities exchanged."""
function relabelled_flows(m, n, levelset, density, viscosity; periodic, boundary)
    return map((false, true)) do swapped
        mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (8, 8), periodic)
        interface = InterfaceField(mesh, swapped ? (x, y) -> -levelset(x, y) : levelset; lipschitz = 1.5)
        physics = TwoPhase(
            interface; density = swapped ? reverse(density) : density,
            dynamic_viscosity = swapped ? reverse(viscosity) : viscosity,
            surface_tension = SurfaceTension(1.0; method = CGFM()),
        )
        FlowSimulation(mesh, physics; boundary, velocity_model = velocity_model(m))
    end
end

function relabelling_defect(normal, swapped)
    defect = scale = 0.0
    a_fields, b_fields = phase_fields(normal.velocity), phase_fields(swapped.velocity)
    pairs_of_fields = length(a_fields) == 1 ? ((a_fields[1], b_fields[1]),) :
        ((a_fields[1], b_fields[2]), (a_fields[2], b_fields[1]))
    for (fa, fb) in pairs_of_fields, (a, b) in zip(fa.interior, fb.interior)
        isnan(a) && isnan(b) && continue
        defect = isnan(a) || isnan(b) ? Inf : max(defect, abs(a - b))
        isfinite(a) && (scale = max(scale, abs(a)))
    end
    fraction = maximum(TP._cell_references(normal.interface.mesh)) do ref
        abs(TP._get(normal.interface.fraction, ref) + TP._get(swapped.interface.fraction, ref) - 1)
    end
    return defect / scale, fraction
end

"""Eight steps of Δt = 5e-4 from rest of a static drop (off the cell
diagonals) and a capillary wave, in both labellings; the phase roles must
follow the geometry and the densities, not the index (issue #200)."""
function tier2_label_symmetry!(suite, m)
    cases = (
        "static drop" => (
            ratio -> (
                32, (x, y) -> hypot(x - 0.5 - 0.21 / 32, y - 0.5 - 0.37 / 32) - 0.2,
                (1.0, 1 / ratio), (0.0129, 0.0129 / sqrt(ratio)), (false, false),
            )
        ),
        "capillary wave" => (ratio -> (32, (x, y) -> y - 0.5 - 0.02cos(2pi * x), (1.0, 1 / ratio), (0.0, 0.0), (true, false))),
    )
    for (name, setup) in cases, ratio in (1.0, 1000.0)
        case_name = name * " DR=$(round(Int, ratio))"
        n = 32
        guarded(suite, 2, "label symmetry", case_name, n, m) do
            _, levelset, density, viscosity, periodic = setup(ratio)
            normal, swapped = relabelled_flows(m, n, levelset, density, viscosity; periodic, boundary = FreeSlip())
            worst = fraction = 0.0
            for _ in 1:8
                for flow in (normal, swapped)
                    checked(step!(flow, 5.0e-4; tolerance = 1.0e-12, max_cycles = 2000))
                end
                defect, change = relabelling_defect(normal, swapped)
                worst, fraction = max(worst, defect), max(fraction, change)
            end
            record!(suite, 2, "label symmetry", case_name, n, m, "velocity", worst; atol = 1.0e-8)
            record!(suite, 2, "label symmetry", case_name, n, m, "fraction", fraction; atol = 1.0e-12)
        end
    end
    return
end

# Marks of the summary table (see the tier-1 marks in tier1.jl).
MARK_TEXT["E"] = "exact by construction (round-off or projection tolerance)"
MARK_TEXT["1V"] = "one velocity field cannot carry the sheet's tangential jump: O(1) by construction"
for m in (:cmdgfm, :cmdgfm_constant, :mimetic, :one_velocity)
    push!(TABLE_MARKS, ("hydrostatic", r"", r"", m, "E"))
end
for m in (:cmdgfm, :cmdgfm_constant, :mimetic)
    push!(TABLE_MARKS, ("translation", r"", r"^(tangential|normal)$", m, "E"))
end
push!(
    TABLE_MARKS,
    ("translation", r"", r"^tangential$", :one_velocity, "1V"),
    ("label symmetry", r"", r"^velocity$", :mimetic, "E"),
)

const TIER2_TESTS = (
    tier2_hydrostatic!, tier2_static_drop!, tier2_capillary_wave!, tier2_kelvin_helmholtz!,
    tier2_translation!, tier2_label_symmetry!,
)

function run_tier2!(suite; tests = TIER2_TESTS)
    for m in suite.methods
        println("Tier 2, ", m.label)
        flush(stdout)
        for test in tests
            test(suite, m)
        end
    end
    record_orders!(suite, 2)
    return suite
end
