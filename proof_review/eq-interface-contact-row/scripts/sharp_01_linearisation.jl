# Sharpness 01: the contact row is a first-order linearisation; can the exact condition be imposed?
#
# Label:  eq:interface:contact-row
# Claim:  04_interface.tex:503-513: "the linearised condition (4.3) on the fitted normal is (4.20) ... it is the
#         unnormalised fitted normal eta - (a1 + 2 a2 tau*) t inserted in (4.3)".
# Fact:   with N = eta - f' t (f' = 2 a2 tau* + a1) the exact condition (N/|N|).n_w = cos(theta) reads
#         N.n_w = cos(theta) sqrt(1+f'^2); the row replaces sqrt(1+f'^2) by 1 (exact to first order in f'),
#         so the enforced normal has N^.n_w = cos(theta)/sqrt(1+f'^2), an angle error ~ (1/2) f'^2 cot(theta).
# Method: on the sloped-wall fixtures of code_01 (random wall slopes, angles 35..135 degrees, radii 3.5..5 cells,
#         unit cells, :lhf_ghf), for every fit that uses the contact row (the package's point selection), re-solve
#         the weighted least squares with the exact condition imposed by fixed-point iteration
#         rhs_k = cos(theta) sqrt(1 + f'_{k-1}^2) - eta.n_w  (row scaled by 1e3 as in the code); report the exact angle
#         error and the curvature error |kappa - 1/R| (clipped as in (4.17)) for k = 0 (the code) .. 4.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-contact-row/scripts/sharp_01_linearisation.jl
#         (from the repository root)

import TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :triple_field) || include(joinpath(@__DIR__, "code_helpers.jl"))

@testset "exact contact condition by fixed-point iteration on the row's right-hand side" begin
    rng = MersenneTwister(33)
    iterations = 4
    angle_err = [Float64[] for _ in 0:iterations]
    kappa_err = [Float64[] for _ in 0:iterations]
    slopes = Float64[]
    angle_by_slope = Dict(:small => Float64[], :large => Float64[])
    diverged = 0
    for trial in 1:60
        theta = deg2rad(35 + 100 * rand(rng))
        slope = (rand(rng) < 0.5 ? 1 : -1) * (0.05 + 0.5 * rand(rng))
        radius = 3.5 + 1.5rand(rng)
        field, geometry, _ = triple_field(; slope, y0 = 3.0 + 0.4 * rand(rng), x0 = 8.0, radius, theta, xc = 9.5 + 3 * rand(rng))
        values = AMR._fill_height_values!(field)[1]
        for j in 1:8, i in 1:16
            ref = AMR.CellRef(1, i, j)
            c = AMR._get(field.conditional_fraction, ref); alpha = AMR._get(geometry.solid.fraction, ref)
            (0 < c < 1 && 0 < alpha < 1) || continue
            normal, origin, h, contact, points, _ = ghf_setup(field, values, ref)
            contact === nothing && continue
            ts, nw, cs = contact
            t = (-normal[2], normal[1]); proj = t[1] * nw[1] + t[2] * nw[2]
            eta_nw = normal[1] * nw[1] + normal[2] * nw[2]
            fit = ls_independent(points, normal, origin, h, contact; alpha = 1.0e3)
            fit === nothing && continue
            abs(proj) > 1.0e-3 || continue                         # row degenerate when the seed normal is parallel to n_w
            fprev = 2fit.a2 * ts + fit.a1
            push!(slopes, abs(fprev))
            records = Tuple{Float64, Float64}[]
            for k in 0:iterations
                if k > 0
                    fit = ls_independent(points, normal, origin, h, contact; alpha = 1.0e3, row_rhs = cs * sqrt(1 + fprev^2) - eta_nw)
                    fit === nothing && break
                    fprev = 2fit.a2 * ts + fit.a1
                end
                enforced = (eta_nw - fprev * proj) / sqrt(1 + fprev^2)
                push!(records, (abs(acos(clamp(enforced, -1, 1)) - acos(cs)), abs(clamp(fit.kappa, -0.5, 0.5) - 1 / radius)))
            end
            length(records) == iterations + 1 || continue
            for k in 0:iterations
                push!(angle_err[k + 1], records[k + 1][1]); push!(kappa_err[k + 1], records[k + 1][2])
            end
            push!(abs(2fit.a2 * ts + fit.a1) < 0.3 ? angle_by_slope[:small] : angle_by_slope[:large], records[1][1])
            records[end][1] > records[1][1] && (diverged += 1)
        end
    end
    q(v, p) = sort(v)[clamp(ceil(Int, p * length(v)), 1, length(v))]
    n = length(angle_err[1])
    println("  $n fits with a contact row; |f'| at the triple point (code): median $(round(q(slopes, 0.5), sigdigits = 3)), 90% $(round(q(slopes, 0.9), sigdigits = 3))")
    println("  iteration k (0 = code):            ", join((@sprintf("%10d", k) for k in 0:iterations), " "))
    for (title, tab) in (("exact angle error [rad]", angle_err), ("|kappa - 1/R| (clipped)", kappa_err))
        for (label, p) in (("median", 0.5), ("90%", 0.9), ("max", 1.0))
            @printf("  %-24s %-8s  %s\n", title, label, join((@sprintf("%10.2e", q(tab[k + 1], p)) for k in 0:iterations), " "))
        end
    end
    println("  fits whose exact angle error is larger after $iterations iterations than for the code's row: $diverged of $n")
    # the angle error of the code's row is the linearisation error: it scales like f'^2 cot(theta)/2 and is removed by the iteration
    @test q(angle_err[iterations + 1], 0.5) < 1.0e-3 * q(angle_err[1], 0.5)
    @test q(angle_err[iterations + 1], 0.9) < q(angle_err[1], 0.9)
    # ... but the exact enforcement does not improve the curvature on these fixtures (the discretisation error of the
    # fit dominates): the linearisation is benign for the curvature
    @test q(kappa_err[iterations + 1], 0.5) > 0.8 * q(kappa_err[1], 0.5)
    println("  median exact angle error: |f'| < 0.3: $(q(angle_by_slope[:small], 0.5)) rad ($(length(angle_by_slope[:small])) fits); |f'| >= 0.3: $(q(angle_by_slope[:large], 0.5)) rad ($(length(angle_by_slope[:large])) fits)")
    @test q(angle_by_slope[:large], 0.5) > 5 * q(angle_by_slope[:small], 0.5)
end
