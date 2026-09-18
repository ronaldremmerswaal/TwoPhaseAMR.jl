using Test, TwoPhaseAMR

include(joinpath(@__DIR__, "..", "..", "examples", "two_velocity_instabilities.jl"))

function _kh_parameters(density_ratio, weber; velocity_jump = 1.0)
    density = (1.0, inv(float(density_ratio)))
    surface_tension = density[1] * density[2] * velocity_jump^2 /
        (2pi * weber * sum(density))
    return (;
        density, tangential_velocity = (-velocity_jump / 2, velocity_jump / 2),
        surface_tension, vertical_gravity = 0.0, liquid_above = false,
    )
end

function _rt_parameters(density_ratio, bond; gravity = 1.0)
    density = (1.0, inv(float(density_ratio)))
    surface_tension = (density[1] - density[2]) * gravity /
        (4pi^2 * bond)
    return (;
        density, tangential_velocity = (0.0, 0.0), surface_tension,
        vertical_gravity = -gravity, liquid_above = true,
    )
end

const COMBINED_PARAMETERS = (
    density = (1.0, 1 / 3),
    tangential_velocity = (-0.25, 0.25),
    surface_tension = 0.04,
    vertical_gravity = -1.0,
    liquid_above = true,
)

function _instability_refinement(parameters)
    resolutions = (8, 16, 32)
    errors = map(resolutions) do resolution
        result = two_velocity_instability(
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

@testset "General two-velocity dispersion relation" begin
    kh = instability_dispersion(; _kh_parameters(1.0, 0.5)...)
    @test kh.gravity_term == 0
    @test kh.radicand ≈ kh.capillary_term * (1 - 0.5)

    rt = instability_dispersion(; _rt_parameters(3.0, 0.5)...)
    @test rt.shear_term == 0
    @test rt.radicand ≈ rt.capillary_term * (1 - 0.5)

    combined = instability_dispersion(; COMBINED_PARAMETERS...)
    @test combined.capillary_term > 0
    @test combined.shear_term > 0
    @test combined.gravity_term < 0
    @test combined.radicand ≈ combined.capillary_term -
        combined.shear_term + combined.gravity_term
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
            parameters..., final_time = 0.02,
            tolerance = 1.0e-7, max_cycles = 3000,
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
