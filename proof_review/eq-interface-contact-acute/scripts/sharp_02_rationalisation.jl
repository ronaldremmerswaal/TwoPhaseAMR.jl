# Sharpness 02: does the rationalised form (4.19) of a2 avoid a cancellation of the thesis' formula?
#
# Label:  eq:interface:contact-acute
# Claim:  04_interface.tex:493-499 and code comment height_functions.jl:795 "Rationalization avoids
#         cancellation for a nearly planar interface": a2 = (h1 + h2 - sqrt D)/w^2 (thesis) is
#         rewritten as ((h2-h1)^2 - w^2 T^2)/(w^2 (h1 + h2 + sqrt D)), D = 4 h1 h2 + w^2 T^2.
# Method: nearly straight interfaces: h1 = x1 T, h2 = (x1 + w) T + delta w (x1 = distance of the
#         first column from the contact point, w = cell width from 1e-3 to 1), delta from 1e-2
#         down to 1e-14; the exact a2 of the Float64 inputs in 512-bit arithmetic; relative errors
#         of both Float64 forms, and the absolute error of the direct form in units of
#         eps (h1+h2)/w^2.  Plain Julia (no packages).
# Run:    julia --project=@claude-symbolics proof_review/eq-interface-contact-acute/scripts/sharp_02_rationalisation.jl

using Test, Random, Printf

direct(h1, h2, w, T) = (h1 + h2 - sqrt(4h1 * h2 + w^2 * T^2)) / w^2
rationalised(h1, h2, w, T) = ((h2 - h1)^2 - w^2 * T^2) / (w^2 * (h1 + h2 + sqrt(4h1 * h2 + w^2 * T^2)))

function exact_a2(h1, h2, w, T)
    return setprecision(BigFloat, 512) do
        p, q, ww, TT = BigFloat(h1), BigFloat(h2), BigFloat(w), BigFloat(T)
        (p + q - sqrt(4p * q + ww^2 * TT^2)) / ww^2
    end
end

@testset "rationalisation: accuracy of a2 near a straight interface" begin
    rng = MersenneTwister(2026)
    println("  delta     median rel.err: direct   rationalised    max abs.err(direct) / (eps (h1+h2)/w^2)")
    med_direct = Dict{Float64, Float64}(); med_rat = Dict{Float64, Float64}()
    worst_noise = 0.0
    for delta in (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12, 1.0e-14)
        errs_d = Float64[]; errs_r = Float64[]; noise = 0.0
        for _ in 1:400
            w = 10.0^(-3 * rand(rng)); T = 0.2 + 2rand(rng)
            x1 = w * (0.3 + 1.2rand(rng))
            h1 = x1 * T
            h2 = (x1 + w) * T + delta * w
            a_true = exact_a2(h1, h2, w, T)
            setprecision(BigFloat, 512) do
                push!(errs_d, Float64(abs(BigFloat(direct(h1, h2, w, T)) - a_true) / abs(a_true)))
                push!(errs_r, Float64(abs(BigFloat(rationalised(h1, h2, w, T)) - a_true) / abs(a_true)))
                noise = max(noise, Float64(abs(BigFloat(direct(h1, h2, w, T)) - a_true)) / (eps() * (h1 + h2) / w^2))
            end
        end
        sort!(errs_d); sort!(errs_r)
        med_direct[delta] = errs_d[200]; med_rat[delta] = errs_r[200]
        worst_noise = max(worst_noise, noise)
        @printf("  %-8.0e  %-23.2e %-15.2e %.2f\n", delta, errs_d[200], errs_r[200], noise)
    end
    # Result.  The direct form has an absolute error of one ulp of (h1+h2)/w^2 whatever delta is (the
    # cancellation (h1+h2) - sqrt D), so its relative error grows like eps/delta.  The rationalised form moves the
    # cancellation to (h2-h1)^2 - w^2 T^2, i.e. to the data (a2 is a difference of size delta of quantities of size
    # h): its relative error also grows like eps/delta, but with a constant ~10 times smaller (the factor
    # ((h1+h2)/(w T))^2 of the two absolute errors).  The rationalisation is therefore a modest improvement, not a
    # removal of the cancellation, and irrelevant in absolute terms: |delta kappa| <= 2 eps (h1+h2)/w^2 ~ 1e-13 for w = 1e-3.
    @test worst_noise <= 3
    @test med_direct[1.0e-10] > 1.0e-7 && med_direct[1.0e-14] > 1.0e-3
    @test med_rat[1.0e-2] < 1.0e-14 && med_rat[1.0e-8] < 1.0e-8 && med_rat[1.0e-10] < 1.0e-6
    @test all(med_rat[d] <= med_direct[d] for d in keys(med_rat) if d <= 1.0e-6)
    @test all(5 < med_direct[d] / med_rat[d] < 50 for d in keys(med_rat))
end
