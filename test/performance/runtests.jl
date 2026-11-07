isempty(ARGS) && push!(ARGS, "performance")
include(joinpath(@__DIR__, "..", "runtests.jl"))
