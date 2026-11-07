isempty(ARGS) && push!(ARGS, "integration")
include(joinpath(@__DIR__, "..", "runtests.jl"))
