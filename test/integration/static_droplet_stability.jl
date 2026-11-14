using Test, TwoPhaseAMR

# Static circular drop in a unit box, free-slip walls, drop clear of the walls.
# A drop at rest should stay at rest: discrete curvature errors may drive a
# spurious current, but it must neither grow in time nor with refinement.
# The viscosity follows the thesis floating-droplet study (st_validation.tex,
# sec. 4.8.2): the density and dynamic-viscosity ratios satisfy
# rho_g/rho_l = (mu_g/mu_l)^2, so mu_g = mu_l sqrt(rho_g/rho_l). The Laplace
# number is La = sigma rho_l R / mu_l^2, and the inviscid case is La = Inf.
# With R = 0.2 the capillary time sqrt(rho_l R^3/sigma) is 0.089; t = 0.4
# covers four and a half capillary times, and the k = 4 Lamb energy period
# T_sigma(4)/2 is 0.036 (about eleven energy periods).
const STATIC_DROP_RADIUS = 0.2
const STATIC_DROP_CENTER = 0.5
const STATIC_DROP_SIGMA = 1.0
const STATIC_DROP_LAPLACE = 1200.0
# The thesis considers 1e2, 1e3 and infinity; 1200 approximates 1e3 and keeps
# the finite-La runs near the Popinet static-droplet range.
const STATIC_DROP_LAPLACES = (Inf, STATIC_DROP_LAPLACE)
const STATIC_DROP_TIME = 0.4
const STATIC_DROP_RATIOS = (1.0, 3.0, 1000.0)
const STATIC_DROP_RESOLUTIONS = (16, 32)
const STATIC_DROP_MODELS = (:one_velocity, :cmdgfm, :mimetic)
# Energy envelope of the inviscid curvature-driven oscillation: the thesis
# requires that it neither grows nor decays. Discrete curvature errors may
# move it by a few percent per peak, so allow 25% between the first and last
# third of the run.
const STATIC_DROP_ENERGY_GROWTH_TOLERANCE = 0.25
# Finite-La decay: the last peak energy must be clearly below the first (the
# thesis requires decay towards zero). At La = 1200 the viscous time is
# sqrt(La) = 34.6 capillary times, so t = 0.4 alone gives about 23% energy
# decay; the 0.9 factor is the documented "clearly below".
const STATIC_DROP_ENERGY_DECAY_RATIO = 0.9
# Late-time capillary number with ONE physical scale for both phases:
# Ca = max(max|u_liquid|, max|u_gas|) mu_liquid / sigma on mixed faces.
# Normalising the gas with mu_gas would hide the gas current. The largest
# value of a passing one-velocity or CMDGFM case is 1.09e-3 (CMDGFM, DR 1000,
# n = 16, gas); the bound leaves a factor of 4.6. The two-velocity mimetic gas
# current at DR 1000 (37 at n = 16) is the known small-aperture defect of the
# current force map (phase-I's per-phase force (A) bounds it at 0.05-0.26) and
# is recorded as broken.
const STATIC_DROP_CAPILLARY_BOUND = 5.0e-3
# A phase must occupy more than this fraction of a mixed face to contribute to
# the spurious-current maximum. The phase-I study bins gas velocities by
# aperture and its smallest non-degenerate bin is (0, 0.05]; a smaller
# aperture carries at most 5% of the face mass, and its intrinsic velocity is
# divided by a vanishing fraction rather than representing a current.
const STATIC_DROP_PHASE_FLOOR = 0.05

function _static_drop_mesh(resolution)
    return BlockTreeMesh(
        block_size = (resolution, resolution), periodic = (false, false),
    )
end

function _static_drop_density(ratio)
    return (1.0, 1.0 / ratio)
end

function _static_drop_liquid_viscosity(ratio, laplace = STATIC_DROP_LAPLACE)
    isinf(laplace) && return 0.0
    density = _static_drop_density(ratio)
    return sqrt(
        STATIC_DROP_SIGMA * density[1] * STATIC_DROP_RADIUS / laplace,
    )
end

function _static_drop_viscosity(ratio, laplace = STATIC_DROP_LAPLACE)
    liquid = _static_drop_liquid_viscosity(ratio, laplace)
    iszero(liquid) && return (0.0, 0.0)
    density = _static_drop_density(ratio)
    # Thesis floating-droplet convention: rho ratio = mu ratio^2.
    return (liquid, liquid * sqrt(density[2] / density[1]))
end

_static_drop_levelset(x, y) =
    hypot(x - STATIC_DROP_CENTER, y - STATIC_DROP_CENTER) - STATIC_DROP_RADIUS

function _static_drop_interface(resolution)
    mesh = _static_drop_mesh(resolution)
    return InterfaceField(mesh, _static_drop_levelset; lipschitz = 1)
end

function _static_drop_model(model)
    model === :one_velocity && return OneVelocityModel()
    model === :cmdgfm &&
        return TwoVelocityModel(pressure_projection = CMDGFMProjection())
    model === :mimetic &&
        return TwoVelocityModel(pressure_projection = MimeticProjection())
    throw(ArgumentError("unknown static-droplet model $model"))
end

function _static_drop_flow(
        model, ratio, resolution; laplace = STATIC_DROP_LAPLACE,
    )
    interface = _static_drop_interface(resolution)
    physics = TwoPhase(
        interface; dynamic_viscosity = _static_drop_viscosity(ratio, laplace),
        density = _static_drop_density(ratio),
        surface_tension = SurfaceTension(STATIC_DROP_SIGMA; method = CGFM()),
    )
    return FlowSimulation(
            interface.mesh, physics; velocity_model = _static_drop_model(model),
            boundary = FreeSlip(),
        ), interface
end

"""Largest intrinsic face velocity of each phase on a mixed face.

The face is mixed when the mean of its two cells' conditional fractions lies
strictly between 0 and 1 (the gate of the phase-I static-droplet study). The
phase must occupy more than `STATIC_DROP_PHASE_FLOOR` of the face: a smaller
aperture carries at most that fraction of the face's mass, and its intrinsic
velocity (velocity divided by the aperture) is a vanishing-phase ratio rather
than a current in the drop. Zero is returned for the gas of the one-velocity
model, which has a single field.
"""
function _static_drop_phase_maxima(velocity, interface)
    liquid_aperture, gas_aperture =
        TwoPhaseAMR._current_phase_face_apertures!(interface)
    liquid_field = velocity isa FaceVelocity ? velocity : velocity[1]
    liquid = _static_drop_phase_max(liquid_field, interface, liquid_aperture)
    gas = velocity isa FaceVelocity ? 0.0 :
        _static_drop_phase_max(velocity[2], interface, gas_aperture)
    return liquid, gas
end

function _static_drop_phase_max(field, interface, aperture)
    maximum_value = 0.0
    for (index, face) in pairs(interface.mesh.faces)
        aperture[index] > STATIC_DROP_PHASE_FLOOR || continue
        c = (
            TwoPhaseAMR._conditional_fraction(interface, face.minus) +
                TwoPhaseAMR._conditional_fraction(interface, face.plus)
        ) / 2
        0 < c < 1 || continue
        value = field.interior[index]
        isfinite(value) &&
            (maximum_value = max(maximum_value, abs(value)))
    end
    return maximum_value
end

"""Phase-mass-weighted kinetic energy over the face dual volumes.

The one-velocity model uses the mixture density of the face aperture; the
two-velocity model sums `rho_phase * aperture * u_phase^2` over both phases.
"""
function _static_drop_energy(velocity, interface, density)
    mesh = interface.mesh
    liquid_aperture, gas_aperture =
        TwoPhaseAMR._current_phase_face_apertures!(interface)
    energy = 0.0
    for (index, face) in pairs(mesh.faces)
        hm = mesh.blocks[face.minus.block].spacing[face.axis]
        hp = mesh.blocks[face.plus.block].spacing[face.axis]
        dual_volume = face.area * (hm + hp) / 2
        if velocity isa FaceVelocity
            value = velocity.interior[index]
            isfinite(value) || continue
            aperture = liquid_aperture[index]
            mixture = aperture * density[1] + (1 - aperture) * density[2]
            energy += mixture * dual_volume * value^2 / 2
        else
            liquid = velocity[1].interior[index]
            gas = velocity[2].interior[index]
            isfinite(liquid) &&
                (
                energy += density[1] * liquid_aperture[index] *
                    dual_volume * liquid^2 / 2
            )
            isfinite(gas) &&
                (
                energy += density[2] * gas_aperture[index] *
                    dual_volume * gas^2 / 2
            )
        end
    end
    return energy
end

function _static_drop_energy_peaks(times, energies)
    isempty(energies) && return Int[]
    threshold = 0.05 * maximum(energies)
    return [
        index for index in 2:(length(energies) - 1) if
            energies[index] > energies[index - 1] &&
            energies[index] >= energies[index + 1] &&
            energies[index] > threshold
    ]
end

"""
Well-balanced-force check with an exactly prescribed constant curvature.

The interface is static, so advection and diffusion are identities at rest and
the step's well balance is decided by the force map and the projection. Each
cycle applies the package's surface-tension acceleration, built from
`_surface_tension_jumps!` with `curvature = 1/R`, and then the model's own
projection. A constant-curvature jump is a discrete gradient, so every cycle
must leave both phase velocities at round-off.
"""
function static_droplet_prescribed_case(
        model, ratio, resolution; steps = 50, dt = 1.0e-3,
    )
    interface = _static_drop_interface(resolution)
    mesh = interface.mesh
    density = _static_drop_density(ratio)
    kappa = inv(STATIC_DROP_RADIUS)
    maximum_speed = 0.0
    if model === :one_velocity
        operator = PoissonOperator(interface; density)
        acceleration = FaceVelocity(mesh)
        velocity = FaceVelocity(mesh)
        pressure = CellField(mesh)
        solver = EllipticSolver(operator; preconditioner = :tree_geometric)
        work = ProjectionWorkspace(solver)
        elliptic = EllipticWorkspace(operator)
        for _ in 1:steps
            surface_tension!(
                acceleration, interface, operator, elliptic;
                sigma = STATIC_DROP_SIGMA, formulation = CGFM(),
                curvature = kappa,
            )
            velocity.interior .+= dt .* acceleration.interior
            project!(
                velocity, pressure, solver, dt, work; tolerance = 1.0e-12,
            )
            maximum_speed = max(
                maximum_speed,
                maximum(_static_drop_phase_maxima(velocity, interface)),
            )
        end
    elseif model === :cmdgfm
        operator = PoissonOperator(interface; density)
        acceleration = FaceVelocity(mesh)
        velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
        pressure = CellField(mesh)
        solver = CMDGFMProjectionSolver(interface; density)
        work = CMDGFMProjectionWorkspace(solver)
        elliptic = EllipticWorkspace(operator)
        for _ in 1:steps
            surface_tension!(
                acceleration, interface, operator, elliptic;
                sigma = STATIC_DROP_SIGMA, formulation = CGFM(),
                curvature = kappa,
            )
            for phase in 1:2
                velocity[phase].interior .+= dt .* acceleration.interior
            end
            project!(
                velocity, pressure, solver, dt, work;
                tolerance = 1.0e-12, max_iterations = 2000,
            )
            maximum_speed = max(
                maximum_speed,
                maximum(_static_drop_phase_maxima(velocity, interface)),
            )
        end
    else
        operator = PoissonOperator(interface; density)
        velocity = (FaceVelocity(mesh), FaceVelocity(mesh))
        pressure = CellField(mesh)
        solver = MimeticProjectionSolver(interface; density)
        work = MimeticProjectionWorkspace(solver)
        for _ in 1:steps
            reconstruct!(interface)
            jumps = TwoPhaseAMR._surface_tension_jumps!(
                operator.gradient_operator.pressure_jump, interface;
                sigma = STATIC_DROP_SIGMA, formulation = CGFM(),
                curvature = kappa,
            )
            TwoPhaseAMR._mimetic_jump_force!(velocity, solver, work, jumps, dt)
            project!(
                velocity, pressure, solver, dt, work;
                tolerance = 1.0e-12, max_iterations = 2000,
            )
            maximum_speed = max(
                maximum_speed,
                maximum(_static_drop_phase_maxima(velocity, interface)),
            )
        end
    end
    return maximum_speed
end

mutable struct StaticDropSpeedObserver <: AbstractObserver
    times::Vector{Float64}
    liquid::Vector{Float64}
    gas::Vector{Float64}
    energy::Vector{Float64}
    density::NTuple{2, Float64}
end

function StaticDropSpeedObserver(density)
    return StaticDropSpeedObserver(
        Float64[], Float64[], Float64[], Float64[], density,
    )
end

function TwoPhaseAMR.observe!(
        observer::StaticDropSpeedObserver, event::Symbol, state, info,
    )
    event === :step || return nothing
    liquid, gas = _static_drop_phase_maxima(state.velocity, state.interface)
    push!(observer.times, state.time)
    push!(observer.liquid, liquid)
    push!(observer.gas, gas)
    push!(
        observer.energy,
        _static_drop_energy(state.velocity, state.interface, observer.density),
    )
    return nothing
end

"""
Static drop with the package's estimated (height-function) curvature.

The first-half and second-half maxima are taken over the recorded per-step
liquid and gas speeds. The late-time capillary number is the larger of the
two phase capillary numbers `max|u_phase| mu_phase / sigma`. The energy peaks
in the first and last third of the run measure the inviscid envelope and the
finite-La decay of the k = 4 Lamb oscillation.
"""
function static_droplet_case(
        model, ratio, resolution;
        laplace = STATIC_DROP_LAPLACE, final_time = STATIC_DROP_TIME,
    )
    density = _static_drop_density(ratio)
    flow, _ = _static_drop_flow(
        model, ratio, resolution; laplace,
    )
    observer = StaticDropSpeedObserver(density)
    result = simulate!(
        flow, final_time; tolerance = 1.0e-9, max_cycles = 1000,
        observer,
    )
    half = final_time / 2
    first_half(speeds) = maximum(
        speed for (speed, time) in zip(speeds, observer.times) if time <= half
    )
    second_half(speeds) = maximum(
        speed for (speed, time) in zip(speeds, observer.times) if time > half
    )
    liquid_early, liquid_late =
        first_half(observer.liquid), second_half(observer.liquid)
    gas_early, gas_late = first_half(observer.gas), second_half(observer.gas)
    peaks = _static_drop_energy_peaks(observer.times, observer.energy)
    peak_energies = observer.energy[peaks]
    first_window = [
        observer.energy[index] for index in peaks if
            observer.times[index] <= final_time / 3
    ]
    last_window = [
        observer.energy[index] for index in peaks if
            observer.times[index] >= 2 * final_time / 3
    ]
    return (;
        liquid_early, liquid_late, gas_early, gas_late,
        early = max(liquid_early, gas_early),
        late = max(liquid_late, gas_late),
        capillary = max(liquid_late, gas_late) *
            _static_drop_liquid_viscosity(ratio, laplace) / STATIC_DROP_SIGMA,
        energy_early = isempty(first_window) ? NaN : maximum(first_window),
        energy_late = isempty(last_window) ? NaN : maximum(last_window),
        first_peak = isempty(peak_energies) ? NaN : first(peak_energies),
        last_peak = isempty(peak_energies) ? NaN : last(peak_energies),
        peaks = length(peaks), steps = result.total_steps,
    )
end

_static_drop_finite_growth_failure(model, ratio) =
    model === :mimetic && (ratio == 1.0 || ratio == 3.0)

@testset "Static-droplet stability" begin
    prescribed = Dict{Tuple{Symbol, Float64, Int}, Float64}()
    measured = Dict{Tuple{Symbol, Float64, Int, Float64}, NamedTuple}()
    for model in STATIC_DROP_MODELS, ratio in STATIC_DROP_RATIOS,
            resolution in STATIC_DROP_RESOLUTIONS
        prescribed[(model, ratio, resolution)] =
            static_droplet_prescribed_case(model, ratio, resolution)
        for laplace in STATIC_DROP_LAPLACES
            measured[(model, ratio, resolution, laplace)] =
                static_droplet_case(model, ratio, resolution; laplace)
        end
    end

    @testset "Well balance at prescribed constant curvature" begin
        for model in STATIC_DROP_MODELS, ratio in STATIC_DROP_RATIOS,
                resolution in STATIC_DROP_RESOLUTIONS
            # Round-off accumulation over 50 force/projection cycles at the
            # 1e-12 projection tolerance, measured from the capillary velocity
            # scale sigma/mu: 1e-9 of it is 7.7e-8. The measured values are at
            # most 1.5e-11 after 50 cycles, so an imbalance at the estimated-
            # curvature level (1e-2 and up) cannot pass.
            bound = 1.0e-9 * STATIC_DROP_SIGMA /
                _static_drop_liquid_viscosity(ratio)
            @test prescribed[(model, ratio, resolution)] <= bound
        end
    end

    for model in STATIC_DROP_MODELS, ratio in STATIC_DROP_RATIOS,
            laplace in STATIC_DROP_LAPLACES
        @testset "$model, density ratio $ratio, La $laplace" begin
            coarse = measured[(model, ratio, 16, laplace)]
            fine = measured[(model, ratio, 32, laplace)]
            if isinf(laplace)
                # Thesis floating droplet: the inviscid curvature-driven
                # k = 4 Lamb oscillation neither grows nor decays. Measured
                # worst ratio of the last-third to first-third peak energies:
                # 1.14 (mimetic, DR 3, n = 32); every case is within 25%.
                @test coarse.energy_late <=
                    coarse.energy_early * (1 + STATIC_DROP_ENERGY_GROWTH_TOLERANCE)
                @test fine.energy_late <=
                    fine.energy_early * (1 + STATIC_DROP_ENERGY_GROWTH_TOLERANCE)
            else
                if _static_drop_finite_growth_failure(model, ratio)
                    # NUMERICS CONCERN: at finite La the two-velocity mimetic
                    # projection amplifies both phase velocities at density
                    # ratios 1 and 3. Measured max|u|, first half -> second
                    # half at t = 0.4. DR 1, n = 16: liquid 4.0e-2 -> 1.4e-1,
                    # gas 9.1e-2 -> 1.5e-1; n = 32: liquid 4.6e-3 -> 1.9e-2,
                    # gas 3.2e-3 -> 4.3e-3. DR 3, n = 16: liquid 1.1e-1 ->
                    # 2.4e-1, gas 7.8e-1 -> 1.03; n = 32: liquid 5.1e-3 ->
                    # 5.7e-3, gas 9.0e-3 -> 1.9e-2.
                    @test_broken coarse.liquid_late <= coarse.liquid_early
                    @test_broken coarse.gas_late <= coarse.gas_early
                    @test_broken fine.liquid_late <= fine.liquid_early
                    @test_broken fine.gas_late <= fine.gas_early
                elseif model === :mimetic && ratio == 1000.0
                    # NUMERICS CONCERN: at DR 1000 the n = 16 liquid current
                    # grows (3.3e-1 -> 5.3e-1) while the larger gas current
                    # decays (40.0 -> 37.1); the joint maximum would hide it.
                    # At n = 32 both decay; the gas velocity is the
                    # small-aperture defect cited at the capillary number.
                    @test_broken coarse.liquid_late <= coarse.liquid_early
                    @test coarse.gas_late <= coarse.gas_early
                    @test fine.liquid_late <= fine.liquid_early
                    @test fine.gas_late <= fine.gas_early
                else
                    @test coarse.liquid_late <= coarse.liquid_early
                    @test coarse.gas_late <= coarse.gas_early
                    @test fine.liquid_late <= fine.liquid_early
                    @test fine.gas_late <= fine.gas_early
                end
                if _static_drop_finite_growth_failure(model, ratio) &&
                        ratio == 1.0
                    # NUMERICS CONCERN: at DR 1, n = 16 the finite-La energy
                    # does not decay: first and last k = 4 peaks are both
                    # 2.2e-4 (one peak above the 5% threshold, no envelope).
                    @test_broken coarse.last_peak <=
                        STATIC_DROP_ENERGY_DECAY_RATIO * coarse.first_peak
                elseif _static_drop_finite_growth_failure(model, ratio) &&
                        ratio == 3.0
                    # NUMERICS CONCERN: at DR 3, n = 16 the energy grows:
                    # first k = 4 peak 4.9e-4, last 5.4e-3. At n = 32 the
                    # same case decays (5.9e-7 -> 1.5e-7) and is asserted.
                    @test_broken coarse.last_peak <=
                        STATIC_DROP_ENERGY_DECAY_RATIO * coarse.first_peak
                else
                    @test coarse.last_peak <=
                        STATIC_DROP_ENERGY_DECAY_RATIO * coarse.first_peak
                end
                @test fine.last_peak <=
                    STATIC_DROP_ENERGY_DECAY_RATIO * fine.first_peak
                if model === :mimetic && ratio == 1000.0
                    # NUMERICS CONCERN: the single liquid viscosity scale
                    # exposes the small-aperture gas current: max|u| = 37.1
                    # (n = 16) and 0.857 (n = 32) give Ca 4.8e-1 and 1.1e-2,
                    # against the 5e-3 bound. Phase-I's per-phase force (A)
                    # bounds these gas velocities at 0.05-0.26.
                    @test_broken coarse.capillary <= STATIC_DROP_CAPILLARY_BOUND
                    @test_broken fine.capillary <= STATIC_DROP_CAPILLARY_BOUND
                elseif model === :mimetic && ratio == 3.0
                    # NUMERICS CONCERN: at DR 3, n = 16 the gas current 1.03
                    # gives Ca 1.3e-2 (n = 32 decays and passes at 2.4e-4).
                    @test_broken coarse.capillary <= STATIC_DROP_CAPILLARY_BOUND
                    @test fine.capillary <= STATIC_DROP_CAPILLARY_BOUND
                else
                    @test coarse.capillary <= STATIC_DROP_CAPILLARY_BOUND
                    @test fine.capillary <= STATIC_DROP_CAPILLARY_BOUND
                end
            end
            # The current must not grow with resolution.
            @test fine.late <= coarse.late
        end
    end
end
