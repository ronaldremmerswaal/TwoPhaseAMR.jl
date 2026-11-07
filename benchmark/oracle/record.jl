include("cases.jl")

baseline = record_cases()
open(joinpath(@__DIR__, "baseline.toml"), "w") do io
    TOML.print(io, baseline; sorted = true)
end
println("Wrote baseline.toml")
