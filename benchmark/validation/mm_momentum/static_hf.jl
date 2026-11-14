# Inviscid static drop (test/integration/static_droplet_stability.jl, La = Inf,
# t = 0.4, package HF curvature, MimeticProjection) at n = 32, 64 and DR 1/3/1000,
# per transport design. MM_ACTIVITY=volume|aperture.
include(joinpath(@__DIR__, "common.jl"))
using Test
let src = read(joinpath(WT, "test", "integration", "static_droplet_stability.jl"), String)
    include_string(Main, src[1:(first(findfirst("@testset \"Static-droplet stability\"", src)) - 1)])
end

open(joinpath(OUT, "static_hf" * SUFFIX * ".txt"), "w") do io
    say(s) = (println(io, s); flush(io); println(s); flush(stdout))
    say("## activity=$(ACTIVITY) ties=$(TIES); speeds on mixed faces (aperture > 0.05), first -> second half; energy peaks last/first third")
    for n in (32, 64), ratio in (1.0, 3.0, 1000.0), (design, match) in ((:D0, false), (:DI, false), (:DP, true))
        set_design!(design)
        TP._MIMETIC_MASS_MATCH[] = match
        tag = string(design, match ? "+mm" : "")
        wall = @elapsed r = try
            static_droplet_case(:mimetic, ratio, n; laplace = Inf)
        catch err
            say(@sprintf("%-6s n=%2d DR=%-4d ERROR %s", tag, n, Int(ratio), first(sprint(showerror, err), 140)))
            nothing
        end
        isnothing(r) && continue
        say(@sprintf("%-6s n=%2d DR=%-4d liq %.2e -> %.2e  gas %.2e -> %.2e  E %.2e / %.2e (x%.2f)  steps %d  (%.0f s)",
            tag, n, Int(ratio), r.liquid_early, r.liquid_late, r.gas_early, r.gas_late,
            r.energy_early, r.energy_late, r.energy_late / r.energy_early, r.steps, wall))
    end
end
set_design!(:D0)
TP._MIMETIC_MASS_MATCH[] = false
