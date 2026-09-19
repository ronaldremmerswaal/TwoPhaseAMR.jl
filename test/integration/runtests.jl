using Test
using TwoPhaseAMR

@testset verbose = true showtiming = true "Integration" begin
    include("amr_manufactured_problem.jl")
    include("vortex_problem.jl")
    include("interfacial_instability_problem.jl")
    include("cylinder_reference_problem.jl")

    include("wave_equation.jl")
    include("staggered_wave_equation.jl")
    include("amr_manufactured_flow.jl")
    include("two_phase_vortex.jl")
    include("two_velocity_transport.jl")
    include("two_velocity_rotation.jl")
    include("two_velocity_vortex.jl")
    include("two_velocity_instabilities.jl")
    include("two_phase_waves.jl")
    include("two_phase_droplet.jl")
    include("two_phase_sessile_droplet.jl")
    include("anisotropic_cells.jl")
    include("single_phase_cylinder.jl")
end
