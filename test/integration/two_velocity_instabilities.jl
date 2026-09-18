using Test, TwoPhaseAMR

include(joinpath(@__DIR__, "..", "..", "examples", "two_velocity_instabilities.jl"))

function _instability_refinement(case, parameters)
    resolutions = (8, 16, 32)
    errors = map(resolutions) do resolution
        result = case(
            ; cells_per_wavelength = resolution, parameters...,
            final_time = 0.05, tolerance = 1.0e-7,
            max_cycles = 3000,
        )
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

@testset "Two-velocity Kelvin--Helmholtz dispersion convergence" begin
    # These coarse grids are controlled by the convective WY restriction, so
    # the first-order split time error is expected.  At finer resolution the
    # capillary restriction crosses over to dt ~ h^(3/2), as in the thesis.
    for parameters in (
            (; density_ratio = 1.0, weber = 0.5),
            (; density_ratio = 1000.0, weber = 0.5),
            (; density_ratio = 1.0, weber = 5.0),
        )
        result = _instability_refinement(kelvin_helmholtz, parameters)
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
            (; density_ratio = 3.0, bond = 0.5),
            (; density_ratio = 1000.0, bond = 0.5),
            (; density_ratio = 3.0, bond = 2.0),
        )
        result = _instability_refinement(rayleigh_taylor, parameters)
        println(
            "two_velocity_RT parameters=$parameters " *
                "n=$(result.resolutions) errors=$(result.errors) " *
                "rates=$(result.rates)",
        )
        @test all(<(0), diff(collect(result.errors)))
        @test result.rates[end] >= 1.7
    end
end

@testset "Two-velocity instability solver on fixed mixed levels" begin
    for case in (kelvin_helmholtz, rayleigh_taylor)
        result = case(
            ; cells_per_wavelength = 32, mixed_level = true,
            final_time = 0.02, tolerance = 1.0e-7, max_cycles = 3000,
        )
        levels = unique(
            block.key.level for block in result.flow.solver.mesh.blocks
        )
        @test length(levels) > 1
        @test result.flow.steps > 0
        @test isfinite(normalized_amplitude_error(
            result.times, result.amplitudes, result.analytical,
        ))
    end
end
