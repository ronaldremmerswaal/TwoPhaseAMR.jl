using Test, TwoPhaseAMR

_kh_parameters(density_ratio, weber) = (;
    density_ratio, weber, bond = 0.0, liquid_above = false,
)

_rt_parameters(density_ratio, bond) = (;
    density_ratio, weber = 0.0, bond, liquid_above = true,
)

const COMBINED_PARAMETERS = (
    density_ratio = 3.0,
    weber = 0.5,
    bond = 0.75,
    mean_interface_velocity = 0.1,
    liquid_above = true,
)

const INSTABILITY_INTERFACE_TRANSPORTS = (
    FrozenInterfaceFluxes(), ProjectedLiquidVelocity(),
)

_instability_transport_name(method) =
    method isa FrozenInterfaceFluxes ? :frozen : :projected

function _test_instability_diagnostics(result)
    # A session contains its initial sample and every completed time step. If
    # the final step lands within roundoff of the requested target, it also
    # records the exact-final completion event documented by
    # VisualizationSession.
    @test length(result.times) in (
        result.flow.steps + 1, result.flow.steps + 2,
    )
    @test all(diff(result.times) .> 0)
    @test first(result.times) == 0
    @test last(result.times) == result.final_time
    return nothing
end

function _instability_refinement(
        parameters;
        interface_transport = FrozenInterfaceFluxes(),
        pressure_projection = CMDGFMProjection(),
    )
    resolutions = (8, 16, 32)
    results = map(resolutions) do resolution
        result = interfacial_instability(
            ; cells_per_wavelength = resolution, parameters...,
            tolerance = 1.0e-7, max_cycles = 3000,
            interface_transport,
            pressure_projection,
        )
        stable = abs(imag(result.dispersion.offset)) <= sqrt(eps())
        expected_ratio = stable ? 1.0 : 2.0
        @test abs(result.analytical[end]) / result.initial_amplitude ≈
            expected_ratio atol = 64eps()
        @test result.final_time ≈ result.default_final_time
        error = normalized_amplitude_error(
            result.times, result.amplitudes, result.analytical,
        )
        @test isfinite(error)
        @test result.flow.steps > 0
        @test typeof(result.interface_transport) === typeof(interface_transport)
        _test_instability_diagnostics(result)
        (; result, error)
    end
    errors = getproperty.(results, :error)
    rates = [
        log(errors[index] / errors[index + 1]) /
            log(resolutions[index + 1] / resolutions[index])
            for index in 1:(length(errors) - 1)
    ]
    endpoint_rate = log(first(errors) / last(errors)) /
        log(last(resolutions) / first(resolutions))
    return (; resolutions, results, errors, rates, endpoint_rate)
end

function _instability_temporal_refinement(
        parameters, spatial_result;
        interface_transport = FrozenInterfaceFluxes(),
        pressure_projection = CMDGFMProjection(),
    )
    factors = (1.0, 0.5, 0.25)
    refined_errors = map((0.5, 0.25)) do timestep_factor
        result = interfacial_instability(
            ; cells_per_wavelength = 16, parameters...,
            timestep_factor, tolerance = 1.0e-7, max_cycles = 3000,
            interface_transport, pressure_projection,
        )
        _test_instability_diagnostics(result)
        normalized_amplitude_error(
            result.times, result.amplitudes, result.analytical,
        )
    end
    errors = [spatial_result.results[2].error, refined_errors...]
    return (; factors, errors)
end

@testset "General two-velocity dispersion relation" begin
    flexible_uniform = _instability_mesh(
        16; block_size = (4, 8), mixed_level = false,
    )
    @test flexible_uniform.block_size == (4, 8)
    @test flexible_uniform.base_mesh == (16, 48)
    flexible_adaptive = _instability_mesh(
        16; block_size = (4, 4), mixed_level = true,
    )
    @test flexible_adaptive.block_size == (4, 4)
    @test flexible_adaptive.base_mesh == (4, 12)
    @test flexible_adaptive.max_level == 2

    kh = instability_dispersion(; _kh_parameters(1.0, 0.5)...)
    @test kh.gravity_term == 0
    @test kh.capillary_term ≈ 1
    @test kh.shear_term ≈ 0.5
    @test kh.radicand ≈ 1 - 0.5
    @test _default_instability_final_time(kh) ≈
        2pi / abs(real(kh.offset))

    rt = instability_dispersion(; _rt_parameters(3.0, 0.5)...)
    @test rt.shear_term == 0
    @test rt.capillary_term ≈ 1
    @test rt.gravity_term ≈ -0.5
    @test rt.radicand ≈ 1 - 0.5

    stable_gravity = instability_dispersion(
        density_ratio = 3.0, weber = 0.0, bond = 0.5,
        liquid_above = false,
    )
    @test stable_gravity.gravity_term ≈ 0.5
    @test stable_gravity.radicand ≈ 1 + 0.5

    combined = instability_dispersion(; COMBINED_PARAMETERS...)
    @test combined.capillary_term ≈ 1
    @test combined.shear_term ≈ COMBINED_PARAMETERS.weber
    @test combined.gravity_term ≈ -COMBINED_PARAMETERS.bond
    @test combined.radicand ≈ 1 - COMBINED_PARAMETERS.weber -
        COMBINED_PARAMETERS.bond
    @test combined.mean_frequency ≈
        combined.k * COMBINED_PARAMETERS.mean_interface_velocity
    @test _default_instability_final_time(combined) ≈
        acosh(2) / abs(imag(combined.offset))
    rho_liquid, rho_gas = combined.density
    liquid_velocity, gas_velocity = combined.tangential_velocity
    @test (
        rho_liquid * liquid_velocity + rho_gas * gas_velocity
    ) / (rho_liquid + rho_gas) ≈ COMBINED_PARAMETERS.mean_interface_velocity
    @test combined.weber ≈ rho_liquid * rho_gas *
        (gas_velocity - liquid_velocity)^2 /
        (combined.surface_tension * combined.k * (rho_liquid + rho_gas))

    @test_throws ArgumentError instability_dispersion(
        density_ratio = 1.0, bond = 0.5,
    )
    @test_throws ArgumentError instability_dispersion(weber = -eps())
    @test_throws ArgumentError instability_dispersion(bond = -eps())
end

@testset "Two-velocity Kelvin--Helmholtz dispersion convergence" begin
    # These coarse grids are controlled by the convective WY restriction, so
    # the first-order split time error is expected. At finer resolution the
    # capillary restriction crosses over to dt ~ h^(3/2), as in the thesis.
    for parameters in (
            _kh_parameters(1.0, 0.5),
            _kh_parameters(1000.0, 0.5),
            _kh_parameters(1.0, 5.0),
        )
        for interface_transport in INSTABILITY_INTERFACE_TRANSPORTS
            method_name = _instability_transport_name(interface_transport)
            result = _instability_refinement(
                parameters; interface_transport,
            )
            println(
                "two_velocity_KH parameters=$parameters " *
                    "method=$method_name n=$(result.resolutions) " *
                    "errors=$(result.errors) rates=$(result.rates) " *
                    "endpoint_rate=$(result.endpoint_rate)",
            )
            @test all(<(0), diff(collect(result.errors)))
            @test result.endpoint_rate >= 0.8

            # As in the capillary-wave study, refine time independently on
            # the middle grid for the two density ratios. These inviscid
            # cases isolate interface transport and pressure projection from
            # the two-velocity diffusion update.
            parameters.weber == 0.5 || continue
            temporal = _instability_temporal_refinement(
                parameters, result; interface_transport,
            )
            println(
                "two_velocity_KH temporal parameters=$parameters " *
                    "method=$method_name factors=$(temporal.factors) " *
                    "errors=$(temporal.errors)",
            )
            @test all(diff(temporal.errors) .<= 0)
        end
    end
end

@testset "Two-velocity Rayleigh--Taylor dispersion convergence" begin
    for parameters in (
            _rt_parameters(3.0, 0.5),
            _rt_parameters(1000.0, 0.5),
            _rt_parameters(3.0, 2.0),
        )
        result = _instability_refinement(parameters)
        println(
            "two_velocity_RT parameters=$parameters method=frozen " *
                "n=$(result.resolutions) errors=$(result.errors) " *
                "rates=$(result.rates) " *
                "endpoint_rate=$(result.endpoint_rate)",
        )
        @test all(<(0), diff(collect(result.errors)))
        @test result.endpoint_rate >= 1.7
    end
end

@testset "Combined KH/RT dispersion convergence" begin
    result = _instability_refinement(COMBINED_PARAMETERS)
    println(
        "two_velocity_combined_instability " *
            "parameters=$COMBINED_PARAMETERS method=frozen " *
            "n=$(result.resolutions) errors=$(result.errors) " *
            "rates=$(result.rates) " *
            "endpoint_rate=$(result.endpoint_rate)",
    )
    @test all(<(0), diff(collect(result.errors)))
    @test result.endpoint_rate >= 0.8
end

@testset "Two-velocity instability solver on fixed mixed levels" begin
    for parameters in (
            _kh_parameters(1.0, 0.5),
            _rt_parameters(3.0, 0.5),
            COMBINED_PARAMETERS,
        )
        result = interfacial_instability(
            ; cells_per_wavelength = 32, mixed_level = true,
            parameters...,
            tolerance = 1.0e-7, max_cycles = 3000,
            pressure_projection = CMDGFMProjection(),
        )
        stable = abs(imag(result.dispersion.offset)) <= sqrt(eps())
        expected_ratio = stable ? 1.0 : 2.0
        @test abs(result.analytical[end]) / result.initial_amplitude ≈
            expected_ratio atol = 64eps()
        @test result.final_time ≈ result.default_final_time
        levels = unique(
            block.key.level for block in result.flow.solver.mesh.blocks
        )
        @test length(levels) > 1
        @test result.interface_indicator isa InterfaceRefinementIndicator
        @test result.flow.solver.refinement_indicators ==
            (result.interface_indicator,)
        @test nblocks(result.mesh) <
            prod(result.mesh.root.layout) * 4^result.mesh.max_level
        @test result.flow.steps > 0
        _test_instability_diagnostics(result)
        @test isfinite(
            normalized_amplitude_error(
                result.times, result.amplitudes, result.analytical,
            )
        )
    end
end
