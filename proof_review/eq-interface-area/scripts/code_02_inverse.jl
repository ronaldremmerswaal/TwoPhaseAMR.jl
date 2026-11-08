# Code check 02: the plane constant of (4.9) (54 bisection steps) against the closed-form inverse of
# Scardovelli and Zaleski and a 256-bit reference, over the whole range of fractions.
#
# Label:  eq:interface:area (eq. 4.9, plane-constant inverse)
# Claim:  04_interface.tex:157-168, 181-190: "the plane constant is the root of (4.9), found by 54
#         bisection steps on [-r, r], r = (1/2) sum_k |n_k|"; the bisection root differs from the
#         closed-form inverse of Scardovelli and Zaleski by at most 1e-14 for 1e-2 <= C <= 1-1e-2
#         ("near pure cells the plane constant is ill-conditioned, da/dC ~ C^(-1/2)").
# Code:   src/interface/interface_field.jl:229-260 (_plane_fraction, _plane_intercept).
# Method: truth = closed form in 256-bit BigFloat (validated against exact polygon clipping at the
#         returned value); errors of the bisection and of the Float64 closed form per fraction C,
#         the amplification dz/dC, and the area residual |F(a(C)) - C|.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-area/scripts/code_02_inverse.jl
#         (from the repository root)

import TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :Geo) || include(joinpath(@__DIR__, "..", "..", "_shared", "techref_geometry.jl"))
using .Geo

setprecision(BigFloat, 256)

function test_normals()
    out = Tuple{Float64, Float64}[]
    for scale in (1.0, 1.7), k in 0:63
        angle = 2pi * (k + 0.37) / 64
        c, s = cos(angle), sin(angle)
        push!(out, scale .* (c, s) ./ (abs(c) + abs(s)))
    end
    append!(out, [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (1.0, 1.0e-9), (1.0, 1.0e-3), (-0.3, 0.7)])
    return out
end

const FRACTIONS = vcat(
    [10.0^(-e) for e in (14, 12, 10, 8, 6, 4)], [0.01, 0.1, 0.5, 0.9, 0.99],
    [1 - 10.0^(-e) for e in (4, 6, 8, 10, 12, 14)],
)

"""One row per fraction: (C, e_bis, e_sz, area residual, max e_bis/(eps dz/dC)); errors in units of s = |n|_1."""
function inverse_table()
    rows = NTuple{5, Float64}[]
    for c in FRACTIONS
        e_bis = e_sz = e_res = ratio = 0.0
        for (n1, n2) in test_normals()
            s = abs(n1) + abs(n2)
            truth = sz_plane_constant(BigFloat(c), BigFloat(n1), BigFloat(n2))
            a_bis = AMR._plane_intercept(c, n1, n2)
            a_sz = sz_plane_constant(c, n1, n2)
            e1 = Float64(abs(BigFloat(a_bis) - truth)) / s
            e2 = Float64(abs(BigFloat(a_sz) - truth)) / s
            e_bis = max(e_bis, e1)
            e_sz = max(e_sz, e2)
            e_res = max(e_res, abs(AMR._plane_fraction(n1, n2, a_bis) - c))
            # an absolute error eps in F moves the root by eps*dz/dC (dz/dC cut off at both ends of F)
            wl, wg = minmax(abs(n1), abs(n2))
            if wl > 0
                v1 = wl / (2wg)
                dz = c <= v1 ? sqrt(wl * wg / (2c)) : (c <= 1 - v1 ? wg : sqrt(wl * wg / (2 * (1 - c))))
                ratio = max(ratio, e1 * s / (eps() * dz))
            end
        end
        push!(rows, (c, e_bis, e_sz, e_res, ratio))
    end
    return rows
end

@testset "the BigFloat closed form is the root of the exact polygon area" begin
    rng = MersenneTwister(1)
    for _ in 1:200
        n1, n2 = randn(rng, 2)
        c = BigFloat(rand(rng))
        a = sz_plane_constant(c, BigFloat(n1), BigFloat(n2))
        area = plane_area_reference(BigFloat(n1), BigFloat(n2), a, BigFloat(-0.5), BigFloat(0.5), BigFloat(-0.5), BigFloat(0.5))
        @test abs(area - c) <= BigFloat(10)^(-60)
    end
end

const ROWS = inverse_table()

@testset "bisection vs closed form vs truth, per fraction" begin
    println("  C            max|a_bis-a*|/s   max|a_SZ-a*|/s   max |F(a_bis)-C|   max [e_bis / (eps dz/dC)]")
    for (c, e_bis, e_sz, e_res, ratio) in ROWS
        @printf("  %-12.4e %-17.3e %-16.3e %-18.3e %.3g\n", c, e_bis, e_sz, e_res, ratio)
        @test e_res <= 4eps()
    end
    # the Float64 closed form is accurate to a few ulp of s over the whole range of C
    @test maximum(r[3] for r in ROWS) <= 4eps()
    # the statement of the source: bisection and closed form agree to 1e-14 for 1e-2 <= C <= 1-1e-2
    @test maximum(r[2] for r in ROWS if 0.01 <= r[1] <= 0.99) <= 1.0e-14
    # the root error of the bisection is an eps-sized error of F times dz/dC with a small constant
    @test maximum(r[5] for r in ROWS) <= 8
end

@testset "closed-form inverse is accurate at both ends; the bisection loses accuracy only near full cells" begin
    near_empty = maximum(r[2] for r in ROWS if r[1] <= 1.0e-4)
    near_full = maximum(r[2] for r in ROWS if r[1] >= 1 - 1.0e-4 - 1.0e-12)
    println("  worst bisection error (units of s): C <= 1e-4: $(near_empty);  C >= 1-1e-4: $(near_full)")
    @test near_empty <= 4eps()
    @test near_full > 100 * near_empty
end
