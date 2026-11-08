# Code check 01: the contact row (4.20) and its weight against the package.
#
# Label:  eq:interface:contact-row
# Claim:  04_interface.tex:500-523: the triple point is the intersection of the interface and solid lines of the
#         cell, tau* = t.(x_t - o)/h, n_w the solid normal into the fluid; the row
#         -2 tau* (t.n_w) a2 - (t.n_w) a1 = cos(theta) - eta.n_w is "added to (4.15) with weight 1e3".
# Code:   src/interface/height_functions.jl:466-500 (_ghf_fit: the row is multiplied by 1e3),
#         502-544 (_embedded_ghf_constraint), 546-600 (_ghf_geometry).
# Method: (A) the constraint built by the package (tau*, n_w, cos theta) equals an independent evaluation from the
#             stored interface and solid planes (anisotropic cells, random walls and angles);
#         (B) the package fit with the contact row equals an independent least-squares solve whose row is scaled by
#             1e3 (weight 1e6 in sum_j w_j r_j^2) and differs from the one scaled by sqrt(1e3) (weight 1e3);
#         (C) enforcement: residual of the row and of the exact condition (N/|N|).n_w = cos(theta) at the triple
#             point for scalings 1, sqrt(1e3), 1e3 (code), 1e6, and the equality-constrained limit; curvature errors.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-contact-row/scripts/code_01_contact_row_vs_package.jl
#         (from the repository root)

import TwoPhaseAMR
using Test, Random, Printf, LinearAlgebra
const AMR = TwoPhaseAMR

isdefined(Main, :triple_field) || include(joinpath(@__DIR__, "code_helpers.jl"))

@testset "(A) the constraint (tau*, n_w, cos theta) equals the independent triple point" begin
    rng = MersenneTwister(31)
    checked = 0; worst = 0.0
    for trial in 1:40
        theta = deg2rad(35 + 100 * rand(rng))
        slope = (rand(rng) < 0.5 ? 1 : -1) * (0.05 + 0.5 * rand(rng))
        hx, hy = trial % 2 == 0 ? (1.0, 1.0) : (0.7 + 0.8 * rand(rng), 0.7 + 0.8 * rand(rng))
        R = (3.5 + 1.5rand(rng)) * min(hx, hy)
        field, geometry, info = triple_field(; slope, y0 = 3.0 * hy + 0.4 * rand(rng) * hy, x0 = 8hx, radius = R, theta,
            xc = (9.5 + 3 * rand(rng)) * hx, hx, hy)
        values = AMR._fill_height_values!(field)[1]
        for j in 1:8, i in 1:16
            ref = AMR.CellRef(1, i, j)
            c = AMR._get(field.conditional_fraction, ref); alpha = AMR._get(geometry.solid.fraction, ref)
            (0 < c < 1 && 0 < alpha < 1) || continue
            normal, origin, h, contact, _, _ = ghf_setup(field, values, ref)
            # independent triple point in physical coordinates relative to the cell centre: solve the two line equations
            mx, my, a = (AMR._get(field.normal_x, ref), AMR._get(field.normal_y, ref), AMR._get(field.intercept, ref))
            sx, sy, sa = (AMR._get(geometry.solid.normal_x, ref), AMR._get(geometry.solid.normal_y, ref), AMR._get(geometry.solid.intercept, ref))
            M = [mx / hx my / hy; sx / hx sy / hy]
            if abs(det(M)) <= 256eps() * hypot(mx / hx, my / hy) * hypot(sx / hx, sy / hy)
                @test contact === nothing
                continue
            end
            xt = M \ [a, sa]
            inside = abs(xt[1]) <= hx * (0.5 + 256eps()) && abs(xt[2]) <= hy * (0.5 + 256eps())
            if !inside
                @test contact === nothing
                continue
            end
            @test contact !== nothing
            t = (-normal[2], normal[1])
            ts = (t[1] * (xt[1] - origin[1]) + t[2] * (xt[2] - origin[2])) / h
            nwv = [sx / hx, sy / hy] ./ hypot(sx / hx, sy / hy)
            worst = max(worst, abs(ts - contact[1]), abs(nwv[1] - contact[2][1]), abs(nwv[2] - contact[2][2]), abs(cos(theta) - contact[3]))
            @test isapprox(nwv[1] * info.nw[1] + nwv[2] * info.nw[2], 1.0; atol = 0.1)    # the stored solid normal is close to the exact wall normal
            checked += 1
        end
    end
    println("  (A) $checked triple cells with a contact row; max deviation of (tau*, n_w, cos theta): $worst")
    @test checked >= 40 && worst <= 1.0e-12
end

@testset "(B) package fit = least squares with the row scaled by 1e3 (weight 1e6), not by sqrt(1e3) (weight 1e3)" begin
    rng = MersenneTwister(32)
    cells = no_free = 0
    stages = Dict{Int, Int}()
    dev_code = dev_weight1e3 = dev_unconstrained = 0.0
    for trial in 1:60
        theta = deg2rad(35 + 100 * rand(rng))
        slope = (rand(rng) < 0.5 ? 1 : -1) * (0.05 + 0.5 * rand(rng))
        field, geometry, _ = triple_field(; slope, y0 = 3.0 + 0.4 * rand(rng), x0 = 8.0, radius = 3.5 + 1.5rand(rng), theta, xc = 9.5 + 3 * rand(rng))
        values = AMR._fill_height_values!(field)[1]
        for j in 1:8, i in 1:16
            ref = AMR.CellRef(1, i, j)
            c = AMR._get(field.conditional_fraction, ref); alpha = AMR._get(geometry.solid.fraction, ref)
            (0 < c < 1 && 0 < alpha < 1) || continue
            normal, origin, h, contact, points, stage = ghf_setup(field, values, ref)
            contact === nothing && continue
            code = AMR._ghf_fit(points, normal, origin, h; contact)
            code === nothing && continue
            stages[stage] = get(stages, stage, 0) + 1
            # the package's own constrained geometry (same seed, same previous-pass planes) is this fit up to orientation
            ghf = AMR._ghf_geometry(field, values, 1, i, j, field.previous_normal_x, field.previous_normal_y, field.previous_intercept; enforce_contact = true)
            @test ghf !== nothing && isapprox(abs(ghf[3]), abs(code[3]); rtol = 1.0e-12, atol = 1.0e-14)
            mine = ls_independent(points, normal, origin, h, contact; alpha = 1.0e3)
            w3 = ls_independent(points, normal, origin, h, contact; alpha = sqrt(1.0e3))
            free = ls_independent(points, normal, origin, h, contact; alpha = nothing)
            # the package returns (nx, ny, kappa)
            @test isapprox(code[3], mine.kappa; rtol = 1.0e-9, atol = 1.0e-12)
            @test isapprox(code[1], mine.normal[1]; atol = 1.0e-10) && isapprox(code[2], mine.normal[2]; atol = 1.0e-10)
            dev_code = max(dev_code, abs(code[3] - mine.kappa))
            dev_weight1e3 = max(dev_weight1e3, abs(code[3] - w3.kappa))
            # without the row the system often has fewer than three rows (cut neighbours give no height points)
            free === nothing ? (no_free += 1) : (dev_unconstrained = max(dev_unconstrained, abs(code[3] - free.kappa)))
            cells += 1
        end
    end
    println("  (B) $cells fits: max |kappa_code - kappa(row x 1e3)| = $dev_code;  |kappa_code - kappa(row x sqrt(1e3))| = $dev_weight1e3 ;  |kappa_code - kappa(no row)| = $dev_unconstrained ($no_free of $cells fits have fewer than three rows without the contact row); stages used (0 = height points, r = ring radius): $(sort(collect(stages)))")
    @test cells >= 40
    @test dev_code <= 1.0e-10
    @test dev_weight1e3 > 1.0e-7        # a weight of 1e3 in (4.15) would give a (slightly) different curvature: the stated weight is not the implemented one
    @test dev_weight1e3 < 1.0e-3        # ... but the difference is small: the row is enforced to high accuracy either way
end

@testset "(C) enforcement of the contact angle by the row: residuals, angle errors, curvature errors" begin
    rng = MersenneTwister(33)
    scalings = (1.0, sqrt(1.0e3), 1.0e3, 1.0e6)
    keys_all = (scalings..., :limit, :free)
    angle_err = Dict{Any, Vector{Float64}}(a => Float64[] for a in keys_all)
    kappa_err = Dict{Any, Vector{Float64}}(a => Float64[] for a in keys_all)
    row_res = Dict{Any, Vector{Float64}}(a => Float64[] for a in scalings)
    slopes = Float64[]                                         # |f'| at the triple point for the code's scaling
    diff_limit = Float64[]                                     # |angle(code scaling) - angle(equality-constrained limit)|
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
            rhs = cs - normal[1] * nw[1] - normal[2] * nw[2]
            function record(key, fit)
                fit === nothing && return
                fp = 2fit.a2 * ts + fit.a1
                enforced = (normal[1] * nw[1] + normal[2] * nw[2] - fp * proj) / sqrt(1 + fp^2)      # N^.n_w
                push!(angle_err[key], abs(acos(clamp(enforced, -1, 1)) - acos(cs)))
                push!(kappa_err[key], abs(clamp(fit.kappa, -0.5, 0.5) - 1 / radius))                 # clipped as in (4.17), h = 1
                key isa Float64 && push!(row_res[key], abs(-2ts * proj * fit.a2 - proj * fit.a1 - rhs))
                key == 1.0e3 && push!(slopes, abs(fp))
            end
            for a in scalings
                record(a, ls_independent(points, normal, origin, h, contact; alpha = a))
            end
            record(:free, ls_independent(points, normal, origin, h, contact; alpha = nothing))
            # equality-constrained limit: minimise the weighted residual subject to the row exactly (KKT system)
            rows = [sqrt(w) .* ((((x - origin[1]) * t[1] + (y - origin[2]) * t[2]) / h)^2, ((x - origin[1]) * t[1] + (y - origin[2]) * t[2]) / h, 1.0) for (x, y, w) in points]
            A = [r[k] for r in rows, k in 1:3]
            b = [sqrt(w) * ((x - origin[1]) * normal[1] + (y - origin[2]) * normal[2]) / h for (x, y, w) in points]
            C = [-2ts * proj -proj 0.0]
            sol = try ([2A' * A C'; C zeros(1, 1)] \ [2A' * b; rhs]) catch; nothing end
            if sol !== nothing && all(isfinite, sol)
                a2, a1, _ = sol
                record(:limit, (; a2, a1, kappa = -2a2 / (h * sqrt(1 + a1^2)^3)))
                length(angle_err[:limit]) > 0 && length(angle_err[1.0e3]) > 0 && push!(diff_limit, abs(angle_err[1.0e3][end] - angle_err[:limit][end]))
            end
        end
    end
    q(v, p) = isempty(v) ? NaN : sort(v)[clamp(ceil(Int, p * length(v)), 1, length(v))]
    println("  fits: $(length(slopes)); |f'| at the triple point (code scaling): median $(round(q(slopes, 0.5), sigdigits = 3)), ",
        "90% $(round(q(slopes, 0.9), sigdigits = 3)), max $(round(maximum(slopes), sigdigits = 3))")
    order = Any[1.0, sqrt(1.0e3), 1.0e3, 1.0e6, :limit, :free]
    names = ["1", "sqrt(1e3)", "1e3 (code)", "1e6", "eq.-constr.", "no row"]
    println("  row scaling                              ", join((@sprintf("%11s", n) for n in names), " "))
    for (title, tab, ks) in (
            ("|angle enforced - theta| [rad]", angle_err, order), ("|kappa - 1/R|, clipped (4.17)", kappa_err, order),
            ("|row residual|", row_res, order[1:4]),
        )
        for (label, p) in (("median", 0.5), ("90%", 0.9), ("max", 1.0))
            @printf("  %-30s %-6s %s\n", title, label, join((@sprintf("%11.2e", q(tab[k], p)) for k in ks), " "))
        end
    end
    println("  max |angle(1e3) - angle(equality-constrained)| = ", maximum(diff_limit), " over ", length(diff_limit), " fits")
    # (i) the row is enforced better and better as its scaling grows (residual ~ 1/alpha^2)
    @test maximum(row_res[1.0e3]) < 1.0e-3 * maximum(row_res[1.0])
    @test maximum(row_res[1.0e6]) < maximum(row_res[1.0e3])
    # (ii) the code's scaling enforces the row to ~1e-8 and is equivalent to the hard constraint: the angle error that
    #      remains is the linearisation error (1/2) f'^2 cot(theta) of the row (S3), not a lack of weight
    @test maximum(row_res[1.0e3]) < 1.0e-6
    @test maximum(diff_limit) < 1.0e-6
end
