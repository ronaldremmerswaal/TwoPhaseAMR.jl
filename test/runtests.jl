using Test
using TwoPhaseAMR

include("mesh.jl")
include("fields_and_transfer.jl")
include("adaptivity.jl")
include("advection.jl")
include("gradient.jl")
include("wave_equation.jl")
include("staggered_advection.jl")
include("staggered_diffusion.jl")
include("staggered_wave_equation.jl")
include("poisson.jl")
include("navier_stokes.jl")
include("amr_manufactured_flow.jl")
include("external_preconditioners.jl")
include("visualization.jl")
include("visualization_session.jl")
include("performance.jl")
