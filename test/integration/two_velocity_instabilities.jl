using Test, TwoPhaseAMR

include(joinpath(@__DIR__, "..", "..", "examples", "two_velocity_instabilities.jl"))

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

function _instability_refinement(
        parameters;
        two_velocity_projection = MimeticProjection(),
    )
    resolutions = (8, 16, 32)
    errors = map(resolutions) do resolution
        result = two_velocity_instability(
            ; cells_per_wavelength = resolution, parameters...,
            tolerance = 1.0e-7, max_cycles = 3000,
            two_velocity_projection,
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
        error
    end
    rates = [
        log(errors[index] / errors[index + 1]) /
            log(resolutions[index + 1] / resolutions[index])
            for index in 1:(length(errors) - 1)
    ]
    return (; resolutions, errors, rates)
end

@testset "General two-velocity dispersion relation" begin
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
        result = _instability_refinement(parameters)
        println(
            "two_velocity_KH parameters=$parameters " *
                "n=$(result.resolutions) errors=$(result.errors) " *
                "rates=$(result.rates)",
        )
        @test all(<(0), diff(collect(result.errors)))
        @test result.rates[end] >= 0.8
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
            "two_velocity_RT parameters=$parameters " *
                "n=$(result.resolutions) errors=$(result.errors) " *
                "rates=$(result.rates)",
        )
        @test all(<(0), diff(collect(result.errors)))
        @test result.rates[end] >= 1.7
    end
end

@testset "Combined KH/RT dispersion convergence" begin
    result = _instability_refinement(COMBINED_PARAMETERS)
    println(
        "two_velocity_combined_instability parameters=$COMBINED_PARAMETERS " *
            "n=$(result.resolutions) errors=$(result.errors) " *
            "rates=$(result.rates)",
    )
    @test all(<(0), diff(collect(result.errors)))
    @test result.rates[end] >= 0.8
end

@testset "Two-velocity instability solver on fixed mixed levels" begin
    for parameters in (
            _kh_parameters(1.0, 0.5),
            _rt_parameters(3.0, 0.5),
            COMBINED_PARAMETERS,
        )
        result = two_velocity_instability(
            ; cells_per_wavelength = 32, mixed_level = true,
            parameters...,
            tolerance = 1.0e-7, max_cycles = 3000,
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
        @test result.flow.steps > 0
        @test isfinite(
            normalized_amplitude_error(
                result.times, result.amplitudes, result.analytical,
            )
        )
    end
end
