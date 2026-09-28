using Test
using TwoPhaseAMR

function run_test_file(path)
    if get(ENV, "TWOPHASE_TEST_PROGRESS", "false") == "true"
        println("TEST_FILE ", path)
        flush(stdout)
    end
    return @testset "$path" begin
        include(joinpath(@__DIR__, path))
    end
end

@testset verbose = true showtiming = true "TwoPhaseAMR" begin
    @testset "Mesh" begin
        run_test_file("mesh/topology.jl")
        run_test_file("mesh/variables.jl")
        run_test_file("threading.jl")
        run_test_file("mesh/adaptivity.jl")
        run_test_file("mesh/amr_connectivity.jl")
    end

    @testset "Indicators and geometry" begin
        run_test_file("indicator/phase/anisotropic_cells.jl")
        run_test_file("indicator/phase/levelset_initialization.jl")
        run_test_file("indicator/phase/height_functions.jl")
        run_test_file("indicator/geometry.jl")
    end

    @testset "Generic operators" begin
        run_test_file("operators/advection/cell_advection.jl")
        run_test_file("operators/gradient.jl")
        run_test_file("operators/advection/staggered_advection.jl")
        run_test_file("operators/staggered_diffusion.jl")
        run_test_file("operators/poisson.jl")
    end

    @testset "Two-phase methods" begin
        run_test_file("two_phase/transport.jl")
        run_test_file("two_phase/two_velocity/inactive_velocity.jl")
        run_test_file(
            "two_phase/two_velocity/transport/two_velocity.jl",
        )
        run_test_file(
            "two_phase/two_velocity/projection/mimetic_projection.jl",
        )
        run_test_file(
            "two_phase/two_velocity/projection/cmdgfm_projection.jl",
        )
        run_test_file(
            "two_phase/two_velocity/projection/cmdgfm_pressure_reuse.jl",
        )
        run_test_file(
            "two_phase/two_velocity/projection/mimetic_projection_variants.jl",
        )
        run_test_file(
            "two_phase/two_velocity/transport/two_velocity_wy_reference.jl",
        )
        run_test_file("two_phase/variable_staggered_diffusion.jl")
        run_test_file("two_phase/forces.jl")
        run_test_file("two_phase/two_velocity/diffusion.jl")
    end

    @testset "Navier--Stokes and diagnostics" begin
        run_test_file("navier_stokes/body_force.jl")
        run_test_file("navier_stokes/two_phase_navier_stokes.jl")
        run_test_file("navier_stokes/navier_stokes.jl")
        run_test_file("navier_stokes/high_level_api.jl")
        run_test_file("navier_stokes/configuration.jl")
        run_test_file("diagnostics.jl")
        run_test_file("readme.jl")
    end

    @testset "Examples" begin
        # The shared initialization helpers intentionally load before the
        # example smoke files that call them.
        run_test_file("examples/helpers.jl")
        run_test_file("examples/basic_examples.jl")
        run_test_file("examples/cylinder_example.jl")
        run_test_file("examples/stokes3_example.jl")
        run_test_file("examples/lgpi_example.jl")
        run_test_file("examples/two_phase_examples.jl")
    end

    @testset "Linear solvers" begin
        run_test_file("solvers.jl")
    end

    @testset "Visualization" begin
        run_test_file("visualization/fields.jl")
        run_test_file("visualization/embedded_geometry.jl")
        run_test_file("visualization/session.jl")
    end
end
