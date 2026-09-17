using Test, LinearAlgebra, SpecialFunctions, TwoPhaseAMR

"""Cosine-mode amplitude obtained by integrating the geometric VOF field."""
function wave_amplitude(interface)
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

function _quartic_roots(coefficients::NTuple{5, Float64})
    coefficients[1] == 1 ||
        throw(ArgumentError("the quartic must be monic"))
    companion = zeros(ComplexF64, 4, 4)
    companion[2, 1] = companion[3, 2] = companion[4, 3] = 1
    companion[:, 4] .= -reverse(collect(coefficients[2:end]))
    return eigvals(companion)
end

"""Prosperetti's equal-kinematic-viscosity linear amplitude solution.

This is the formula reproduced in the thesis, with the polynomial coefficient
`-4beta*sqrt(nu*k^2)` used by its reference implementation. The returned
quantity is normalized by the initial amplitude.
"""
function prosperetti_amplitude(
        time; liquid_density, gas_density, kinematic_viscosity,
        gravity, surface_tension, wavenumber = 2pi
    )
    rho_l, rho_g = liquid_density, gas_density
    nu, k = kinematic_viscosity, wavenumber
    omega0_squared = (
        gravity * k * (rho_l - rho_g) + surface_tension * k^3
    ) / (rho_l + rho_g)
    omega_nu = nu * k^2
    beta = rho_l * rho_g / (rho_l + rho_g)^2
    roots = _quartic_roots(
        (
            1.0,
            -4beta * sqrt(omega_nu),
            2(1 - 6beta) * omega_nu,
            4(1 - 3beta) * omega_nu^(3 / 2),
            (1 - 4beta) * omega_nu^2 + omega0_squared,
        )
    )
    products = [
        prod(roots[j] - roots[i] for j in eachindex(roots) if i != j)
            for i in eachindex(roots)
    ]
    result = 4(1 - 4beta) * omega_nu^2 /
        (8(1 - 4beta) * omega_nu^2 + omega0_squared) *
        erfc(sqrt(omega_nu * time))
    for i in eachindex(roots)
        shifted = roots[i]^2 - omega_nu
        result += roots[i] / products[i] * omega0_squared / shifted *
            exp(shifted * time) * erfc(roots[i] * sqrt(time))
    end
    return real(result)
end

function _wave_mesh(kind, resolution)
    kind in (:uniform, :mixed, :amr) ||
        throw(ArgumentError("unknown mesh kind $kind"))
    if kind == :uniform
        return BlockTreeMesh(
            block_size = (resolution, 3resolution),
            origin = (0.0, -1.5), extent = (1.0, 3.0),
            periodic = (true, false)
        )
    end
    resolution % 4 == 0 ||
        throw(ArgumentError("mixed-grid resolution must be divisible by four"))
    base = resolution ÷ 4
    return BlockTreeMesh(
        block_size = (base, 3base), max_level = 2,
        origin = (0.0, -1.5), extent = (1.0, 3.0),
        periodic = (true, false),
        # The mean interface is at y=0.25, inside the upper level-one row.
        # Refining that row leaves a fixed coarse/fine transition away from
        # the wave while giving the requested resolution at the interface.
        refine = (key, origin, extent) -> kind == :mixed &&
            (key.level == 0 || (key.level == 1 && key.j == 1))
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

"""Run a quarter-period refinement case for the thesis standing wave.

The thesis uses a three-wavelength-deep domain and `delta0/lambda=1e-2`.
Capillary cases use `La=3000`; gravity cases use `Ga=10000`. Both choices
have equal phase kinematic viscosities, as required by Prosperetti's solution.
"""
function wave_case(
        kind, resolution, density_ratio, mode;
        formulation = CSF(), periods = 0.25
    )
    mode in (:capillary, :gravity, :combined) ||
        throw(ArgumentError("unknown wave mode $mode"))
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
    mesh = _wave_mesh(kind, resolution)
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    indicator = kind == :amr ?
        InterfaceRefinementIndicator(interface; level = 2) : nothing
    if kind == :amr
        adaptation = AMRWorkspace(mesh, interface)
        for step in 1:2
            adapt!(adaptation, indicator; step, force = true)
            # Exact reinitialization is confined to construction of the test
            # mesh. Time-dependent AMR uses geometric PLIC transfer.
            initialize!(interface, levelset; lipschitz = 1)
        end
    end
    flow = FlowSimulation(
        mesh; interface, dynamic_viscosity = viscosity,
        density = (liquid_density, gas_density),
        surface_tension, gravity = (0.0, -gravity),
        surface_tension_formulation = formulation,
        boundary = FreeSlip(),
        refinement_indicators = kind == :amr ? (indicator,) : ()
    )
    omega0_squared = (
        (liquid_density - gas_density) * gravity * wavenumber +
            surface_tension * wavenumber^3
    ) / (liquid_density + gas_density)
    omega0 = sqrt(omega0_squared)
    initial_moment = wave_amplitude(interface)
    initial_volume = integrate(interface)

    # Equal-density gravity is the zero-frequency limit. Its body force is a
    # pressure gradient and the mimetic model must preserve the initial state.
    final_time = iszero(omega0) ? 1.0 : periods * 2pi / omega0
    sample_times = collect(range(0.0, final_time; length = 9))
    errors = Float64[0]
    amplitudes = Float64[1]
    maximum_speed = 0.0
    # The explicit capillary/gravity update is first order in time. Use
    # dt=O(h^2) so the refinement study measures the spatial method rather
    # than the force splitting error.
    temporal_steps = max(80, resolution^2 ÷ 2)
    for target in sample_times[2:end]
        dt = min(final_time / temporal_steps, 0.8stable_timestep(flow))
        simulate!(
            flow, target; dt, tolerance = 1.0e-9, max_cycles = 300
        )
        measured = wave_amplitude(interface) / initial_moment
        reference = iszero(omega0) ? 1.0 : prosperetti_amplitude(
                flow.time; liquid_density, gas_density,
                kinematic_viscosity, gravity, surface_tension, wavenumber
            )
        push!(amplitudes, measured)
        push!(errors, measured - reference)
        maximum_speed = max(
            maximum_speed, maximum(abs, flow.velocity.interior; init = 0.0)
        )
        @test integrate(interface) ≈ initial_volume atol = 4096eps()
        @test isfinite(measured)
    end
    error = _trapezoidal_l2(sample_times, errors)
    println(
        "wave mesh=$kind n=$resolution DR=$density_ratio mode=$mode ",
        "normalized_amplitude_L2_error=$error"
    )
    return (; error, amplitudes, maximum_speed, omega0)
end

if abspath(PROGRAM_FILE) == @__FILE__
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
            (1.0e3, :gravity, 0.8),
            (1.0e3, :combined, 1.5),
        )
        for (density_ratio, mode, minimum_rate) in studies
            resolutions = (8, 16, 32)
            results = [
                wave_case(
                    :uniform, n, density_ratio, mode;
                    formulation = GFM()
                )
                    for n in resolutions
            ]
            errors = getproperty.(results, :error)
            rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
            println(
                "wave convergence mesh=uniform DR=$density_ratio ",
                "mode=$mode n=$resolutions errors=$errors rates=$rates"
            )
            @test all(diff(errors) .< 0)
            @test last(rates) >= minimum_rate
        end

        # At density ratio one a gravity-only perturbation has zero frequency.
        # Check the exact equilibrium instead of assigning an order to
        # roundoff-sized errors.
        equilibrium = wave_case(:uniform, 16, 1.0, :gravity)
        @test iszero(equilibrium.omega0)
        @test all(isapprox.(equilibrium.amplitudes, 1; atol = 4096eps()))
        @test equilibrium.maximum_speed <= 4096eps()

        # Repeat a physical high-contrast wave across fixed mixed-level and
        # dynamically adapted trees. These are refinement studies, not loose
        # one-grid tolerances.
        for kind in (:mixed, :amr)
            resolutions = (8, 16, 32)
            errors = [
                wave_case(
                    kind, n, 1.0e3, :combined;
                    formulation = GFM()
                ).error
                    for n in resolutions
            ]
            rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
            println(
                "wave convergence mesh=$kind DR=1000 mode=combined ",
                "n=$resolutions errors=$errors rates=$rates"
            )
            @test all(diff(errors) .< 0)
            @test last(rates) >= 1.5
        end
    end
end
