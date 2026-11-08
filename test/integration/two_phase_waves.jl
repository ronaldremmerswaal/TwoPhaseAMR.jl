using Test, TwoPhaseAMR

include(joinpath(@__DIR__, "..", "..", "examples", "06_capillary_wave", "prosperetti_amplitude.jl"))

"""Cosine-mode amplitude obtained by integrating the geometric VOF field."""
function _cosine_wave_amplitude(interface)
    mesh = interface.mesh
    amplitude = 0.0
    for ref in TwoPhaseAMR._cell_references(mesh)
        x, _ = cellcenter(mesh, ref.block, ref.i, ref.j)
        hx, hy = mesh.blocks[ref.block].spacing
        # Twice the cosine moment is the height amplitude on a unit-period
        # domain. Integrating the basis over each cell avoids point sampling.
        cosine_integral = (
            sinpi(2 * (x + hx / 2)) -
                sinpi(2 * (x - hx / 2))
        ) / pi
        amplitude += TwoPhaseAMR._get(
            interface.fraction, ref
        ) * hy * cosine_integral
    end
    return amplitude
end

function _two_phase_wave_mesh(kind, resolution)
    kind in (:uniform, :mixed, :amr) ||
        throw(ArgumentError("unknown mesh kind $kind"))
    if kind == :uniform
        iszero(resolution % 8) || throw(
            ArgumentError("wave resolution must be a multiple of eight")
        )
        return BlockTreeMesh(
            base_mesh = (resolution, 3resolution), block_size = (8, 8),
            origin = (0.0, -1.5), extent = (1.0, 3.0),
            periodic = (true, false)
        )
    end
    ratio = resolution / 8
    level = round(Int, log2(ratio))
    8 * 2^level == resolution || throw(
        ArgumentError("mixed-grid resolution must be 8*2^level")
    )
    return BlockTreeMesh(
        base_mesh = (8, 24), block_size = (8, 8), max_level = level,
        origin = (0.0, -1.5), extent = (1.0, 3.0),
        periodic = (true, false),
        # The mean interface is at y=0.25, inside the upper level-one row.
        # Refining that row leaves a fixed coarse/fine transition away from
        # the wave while giving the requested resolution at the interface.
        refine = (key, origin, extent) -> kind == :mixed &&
            key.level < level &&
            origin[2] < 0.4 && origin[2] + extent[2] > 0.1
    )
end

function _trapezoidal_l2(times, errors)
    integral = sum(
        (times[i + 1] - times[i]) *
            (errors[i]^2 + errors[i + 1]^2) / 2
            for i in 1:(length(times) - 1)
    )
    return sqrt(integral / (last(times) - first(times)))
end

function _wave_volume_tolerance(interface, steps)
    mesh = interface.mesh
    cells_per_block = prod(mesh.block_size)
    domain_volume = sum(
        cells_per_block * block.volume for block in mesh.blocks
    )
    # Each of the two directional sweeps can snap at most
    # fraction_floor times the domain volume. Add the accumulated flux-sum
    # roundoff separately.
    wisp_bound = 2 * steps * interface.fraction_floor * domain_volume
    roundoff_bound =
        16 * steps * length(mesh.faces) * eps(eltype(interface))
    return wisp_bound + roundoff_bound
end

"""Run a quarter-period refinement case for the thesis standing wave.

The thesis uses a three-wavelength-deep domain and `delta0/lambda=1e-2`.
Capillary cases use `La=3000`; gravity cases use `Ga=10000`. Both choices
have equal phase kinematic viscosities, as required by Prosperetti's solution.
"""
function wave_case(
        kind, resolution, density_ratio, mode;
        formulation = CSF(), periods = 0.25,
        velocity_model = :one_velocity,
        interface_transport = FrozenInterfaceFluxes(),
        two_velocity_projection = CMDGFMProjection(),
        timestep_factor = 1.0,
        solver_tolerance = 1.0e-9,
    )
    mode in (:capillary, :gravity, :combined) ||
        throw(ArgumentError("unknown wave mode $mode"))
    velocity_model in (:one_velocity, :two_velocity) ||
        throw(ArgumentError("unknown velocity model $velocity_model"))
    isfinite(timestep_factor) && 0 < timestep_factor <= 1 || throw(
        ArgumentError("timestep_factor must be finite and in (0, 1]"),
    )
    wavelength = 1.0
    wavenumber = 2pi / wavelength
    # Keep the unperturbed interface on the same cell-edge position throughout
    # dyadic refinement. Otherwise grid-alignment oscillations contaminate the
    # observed rate of this small-amplitude problem.
    mean_height = 0.25
    initial_amplitude = wavelength / 100
    liquid_density = 1.0
    gas_density = liquid_density / density_ratio
    surface_tension = mode == :gravity ? 0.0 : 1.0
    gravity = mode == :capillary ? 0.0 : 1.0
    kinematic_viscosity = if surface_tension > 0
        sqrt(surface_tension * liquid_density * wavelength / 3000)
    else
        sqrt(gravity * wavelength^3 / 1.0e4)
    end
    viscosity = (
        liquid_density * kinematic_viscosity,
        gas_density * kinematic_viscosity,
    )
    levelset = (x, y) ->
    y - mean_height - initial_amplitude * cospi(2x / wavelength)
    mesh = _two_phase_wave_mesh(kind, resolution)
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    indicator = kind == :amr ?
        InterfaceRefinementIndicator(interface; level = mesh.max_level) : nothing
    if kind == :amr
        adaptation = AMRWorkspace(mesh, interface)
        for step in 1:mesh.max_level
            adapt!(adaptation, indicator; step, force = true)
            # Exact reinitialization is confined to construction of the test
            # mesh. Time-dependent AMR uses geometric PLIC transfer.
            initialize!(interface, levelset; lipschitz = 1)
        end
    end
    initial_velocity = velocity_model === :one_velocity ? nothing : (
            (x, y, time, axis) -> 0.0,
            (x, y, time, axis) -> 0.0,
        )
    flow = FlowSimulation(
        mesh, TwoPhase(
            interface; dynamic_viscosity = viscosity,
            density = (liquid_density, gas_density),
            surface_tension = SurfaceTension(surface_tension; method = formulation),
            gravity = (0.0, -gravity)
        );
        velocity = initial_velocity,
        velocity_model = velocity_model === :one_velocity ? OneVelocityModel() :
            TwoVelocityModel(;
                interface_transport,
                pressure_projection = two_velocity_projection
            ),
        boundary = FreeSlip(),
        adaptivity = Adaptivity(criteria = kind == :amr ? (indicator,) : ()),
    )
    omega0_squared = (
        (liquid_density - gas_density) * gravity * wavenumber +
            surface_tension * wavenumber^3
    ) / (liquid_density + gas_density)
    omega0 = sqrt(omega0_squared)
    initial_moment = _cosine_wave_amplitude(interface)
    initial_volume = integrate(interface)

    # Equal-density gravity is the zero-frequency limit. Its body force is a
    # pressure gradient and the mimetic model must preserve the initial state.
    final_time = iszero(omega0) ? 1.0 : periods * 2pi / omega0
    cycle_limit = velocity_model === :two_velocity ? 1000 : 300
    add_diagnostic!(
        flow, :amplitude,
        state -> _cosine_wave_amplitude(interface) / initial_moment,
    )
    add_diagnostic!(flow, :volume, state -> integrate(interface))
    add_diagnostic!(
        flow, :maximum_speed,
        state -> begin
            phase_velocities = state.velocity isa FaceVelocity ?
                (state.velocity,) : state.velocity
            return maximum(
                maximum(abs, phase.interior; init = 0.0)
                    for phase in phase_velocities
            )
        end,
    )
    selected_dt = timestep_factor == 1 ? nothing :
        timestep_factor * stable_timestep(flow)
    simulate!(
        flow, final_time; dt = selected_dt,
        tolerance = solver_tolerance, max_cycles = cycle_limit,
    )
    amplitude_history = diagnostic_history(flow, :amplitude)
    sample_times = amplitude_history.times
    amplitudes = amplitude_history.values
    references = [
        iszero(omega0) ? 1.0 : prosperetti_amplitude(
                time; liquid_density, gas_density, kinematic_viscosity,
                gravity, surface_tension, wavenumber,
            )
            for time in sample_times
    ]
    errors = amplitudes .- references
    maximum_speed = maximum(diagnostic_history(flow, :maximum_speed).values)
    maximum_volume_error = maximum(
        volume -> abs(volume - initial_volume),
        diagnostic_history(flow, :volume).values,
    )
    @test maximum_volume_error <=
        _wave_volume_tolerance(interface, flow.steps)
    @test all(isfinite, amplitudes)
    error = _trapezoidal_l2(sample_times, errors)
    method_name = velocity_model === :one_velocity ? :one_velocity :
        interface_transport isa FrozenInterfaceFluxes ? :frozen : :projected
    println(
        "wave model=$velocity_model mesh=$kind n=$resolution ",
        "interface_transport=$method_name DR=$density_ratio mode=$mode ",
        "normalized_amplitude_L2_error=$error"
    )
    if velocity_model === :two_velocity
        @test flow.workspace.two_phase.diffusion_solver.stats.converged
    end
    return (;
        error, amplitudes, references, times = sample_times,
        maximum_speed, maximum_volume_error, omega0, solver_tolerance,
        steps = flow.steps,
    )
end

@testset "Two-phase wave integration" begin
    @testset "Prosperetti standing waves on uniform and AMR meshes" begin
        @test prosperetti_amplitude(
            0.0; liquid_density = 1.0, gas_density = 1.0,
            kinematic_viscosity = sqrt(1 / 3000), gravity = 0.0,
            surface_tension = 1.0
        ) ≈ 1.0 atol = 128eps()

        # Uniform-grid order studies cover both requested density ratios and
        # gravity with and without surface tension. The thesis reports second
        # order for finite-La capillary waves and first order for its mimetic
        # gravity-wave discretization.
        studies = (
            (1.0, :capillary, 1.8),
            (1.0e3, :capillary, 1.5),
            # The explicit gravity restriction scales as dt=O(sqrt(h)); its
            # first-order force splitting therefore limits this study to
            # O(sqrt(h)) before the advective CFL becomes active.
            (1.0e3, :gravity, 0.4),
            (1.0e3, :combined, 1.5),
        )
        for (density_ratio, mode, minimum_rate) in studies
            resolutions = (8, 16, 32)
            results = [
                wave_case(
                    :uniform, n, density_ratio, mode;
                    formulation = CGFM()
                )
                    for n in resolutions
            ]
            errors = getproperty.(results, :error)
            rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
            endpoint_rate = log2(first(errors) / last(errors)) /
                (length(errors) - 1)
            println(
                "wave convergence mesh=uniform DR=$density_ratio ",
                "mode=$mode n=$resolutions errors=$errors rates=$rates ",
                "endpoint_rate=$endpoint_rate",
            )
            @test all(diff(errors) .< 0)
            @test endpoint_rate >= minimum_rate

            mode == :capillary || continue
            one_velocity_time_errors = vcat(
                results[2].error,
                [
                    wave_case(
                        :uniform, 16, density_ratio, mode;
                        formulation = CGFM(), timestep_factor,
                    ).error
                        for timestep_factor in (0.5, 0.25)
                ],
            )
            println(
                "capillary temporal refinement DR=$density_ratio ",
                "model=one_velocity factors=(1.0, 0.5, 0.25) ",
                "errors=$one_velocity_time_errors",
            )
            @test all(diff(one_velocity_time_errors) .<= 0)
            for interface_transport in (
                    FrozenInterfaceFluxes(), ProjectedLiquidVelocity(),
                )
                two_velocity_results = [
                    wave_case(
                        :uniform, n, density_ratio, mode;
                        formulation = CGFM(), velocity_model = :two_velocity,
                        interface_transport,
                    )
                        for n in resolutions
                ]
                two_velocity_errors = getproperty.(
                    two_velocity_results, :error,
                )
                two_velocity_rates = log2.(
                    two_velocity_errors[1:(end - 1)] ./
                        two_velocity_errors[2:end],
                )
                endpoint_rate = log2(
                    first(two_velocity_errors) / last(two_velocity_errors),
                ) / (length(two_velocity_errors) - 1)
                method_name = interface_transport isa FrozenInterfaceFluxes ?
                    :frozen : :projected
                println(
                    "two-velocity capillary convergence DR=$density_ratio ",
                    "method=$method_name n=$resolutions ",
                    "theory_errors=$two_velocity_errors ",
                    "theory_rates=$two_velocity_rates ",
                    "endpoint_rate=$endpoint_rate",
                )
                @test all(diff(two_velocity_errors) .< 0)
                @test endpoint_rate >= minimum_rate
                temporal_errors = vcat(
                    two_velocity_results[2].error,
                    [
                        wave_case(
                            :uniform, 16, density_ratio, mode;
                            formulation = CGFM(),
                            velocity_model = :two_velocity,
                            interface_transport, timestep_factor,
                        ).error
                            for timestep_factor in (0.5, 0.25)
                    ],
                )
                println(
                    "capillary temporal refinement DR=$density_ratio ",
                    "model=$method_name factors=(1.0, 0.5, 0.25) ",
                    "errors=$temporal_errors",
                )
                # At fixed spatial resolution temporal and spatial errors can
                # cancel: DR=1 frozen gave 0.006525, 0.007361, 0.010889 as
                # dt was halved twice (maximum/base = 1.67). Guard against
                # more than a factor-two increase rather than monotonicity.
                @test all(isfinite, temporal_errors)
                @test maximum(temporal_errors) <= 2first(temporal_errors)
            end
        end

        # At density ratio one a gravity-only perturbation has zero frequency.
        # Check the exact equilibrium instead of assigning an order to
        # roundoff-sized errors.
        equilibrium = wave_case(:uniform, 16, 1.0, :gravity)
        @test iszero(equilibrium.omega0)
        @test maximum(abs.(equilibrium.amplitudes .- 1)) <=
            equilibrium.solver_tolerance
        @test equilibrium.maximum_speed <= equilibrium.solver_tolerance

        # Repeat a physical high-contrast wave across fixed mixed-level and
        # dynamically adapted trees. These are refinement studies, not loose
        # one-grid tolerances.
        for kind in (:mixed, :amr)
            resolutions = (8, 16, 32)
            errors = [
                wave_case(
                    kind, n, 1.0e3, :combined;
                    formulation = CGFM()
                ).error
                    for n in resolutions
            ]
            rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
            endpoint_rate = log2(first(errors) / last(errors)) /
                (length(errors) - 1)
            println(
                "wave convergence mesh=$kind DR=1000 mode=combined ",
                "n=$resolutions errors=$errors rates=$rates ",
                "endpoint_rate=$endpoint_rate",
            )
            @test all(diff(errors) .< 0)
            @test endpoint_rate >= 1.5
        end
    end
end
