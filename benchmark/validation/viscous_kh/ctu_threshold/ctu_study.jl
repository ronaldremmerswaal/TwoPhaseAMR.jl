# Effect of the CTU threshold and of the bulk momentum flux on the two-velocity
# tangential jump. No src/ changes: the threshold is not reachable from the public
# FlowSimulation options, so the WY advection workspace is rebuilt with the public
# `WYAdvectionWorkspace(interface; ctu_threshold)` constructor and stored in the
# (mutable) flow workspaces (see `set_ctu_threshold!` in ../kh_run.jl).
#
#   julia --project=.. ctu_study.jl reproducer   # part 1 → reproducer.csv
#   julia --project=.. ctu_study.jl kh <scheme>  # part 2 → runs/ (r1_thin, two-velocity)
#   julia --project=.. ctu_report.jl             # results.md, jump_vs_h.png
using Printf
include(joinpath(@__DIR__, "..", "jump_reproducer.jl"))
include(joinpath(@__DIR__, "..", "kh_run.jl"))

const HERE = @__DIR__
const THRESHOLDS = (0.5, 0.1, 0.01, 1.0e-12)
const SCHEMES = Dict("Upwind" => Upwind(), "LaxWendroff" => LaxWendroff(), "MC" => MonotonizedCentral())
const SCHEME_ORDER = ("Upwind", "LaxWendroff", "MC")
const NS = (32, 64, 128)                 # δ/h = 1.6, 3.2, 6.4 (δ = 0.05)
const KH_NS = (64, 128, 256)

function reproducer()
    open(joinpath(HERE, "reproducer.csv"), "w") do io
        println(io, "scheme,threshold,n,delta_over_h,jump1,jump160,umin,umax,fmin,fmax,failure")
        for s in SCHEME_ORDER, θ in THRESHOLDS, n in NS
            r = jump_history(n; scheme = SCHEMES[s], ctu_threshold = θ, steps = (1, 160))
            j1 = length(r.jumps) >= 1 ? r.jumps[1] : NaN
            j160 = length(r.jumps) >= 2 ? r.jumps[2] : NaN
            @printf(
                io, "%s,%g,%d,%.2f,%.6g,%.6g,%.8g,%.8g,%.3g,%.12g,\"%s\"\n", s, θ, n, 0.05n, j1, j160,
                r.umin, r.umax, r.fmin, r.fmax, replace(r.failure, '"' => '\'', '\n' => ' ')
            )
            @printf(
                "%-12s θ = %-6g δ/h = %4.1f: jump(1) = %.4f, jump(160) = %.4f, u ∈ [%.6f, %.6f], f ∈ [%.2e, 1%+.1e] %s\n",
                s, θ, 0.05n, j1, j160, r.umin, r.umax, r.fmin, r.fmax - 1, r.failure
            )
            flush(stdout)
        end
    end
    return nothing
end

function kh(scheme_name)
    c = CASES["r1_thin"]
    for θ in (0.5, 1.0e-12), n in KH_NS
        tag = @sprintf("_%s_ctu%g", scheme_name, θ)
        run_case(
            c, TwoVelocityModel(), n; tag, scheme = SCHEMES[scheme_name], ctu_threshold = θ,
            runs = joinpath(HERE, "runs"), skip_existing = true
        )
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    ARGS[1] == "reproducer" && reproducer()
    ARGS[1] == "kh" && kh(ARGS[2])
end
