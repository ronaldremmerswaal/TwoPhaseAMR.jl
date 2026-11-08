# Sharpness 01: consequence of evaluating the contact parabola at the wall cell (x = -w) instead
# of the first height column (x = 0).
#
# Label:  eq:interface:contact-acute / dev:interface:contact-acute
# Claim:  04_interface.tex:539-546: the thesis evaluates kappa = 2 A/(1+B^2)^(3/2) (slope B at the
#         first column), the code the slope B - 2 A w at the wall cell, "one cell closer to the
#         contact point".
# Method: sessile drops (R = 6 base cells, domain 16 x 8) at acute and obtuse angles, refined by
#         m = 1, 2, 4, 8, 16 (h = 1/m); 8 positions of the contact point inside its cell, both
#         contact points.  For every contact cell the independent evaluation of (4.19) with the
#         slope at x_e in {0, -1/2, -1, -3/2, -2} cell widths (x_e = 0: thesis with the sign of
#         (4.2); x_e = -1: code).  Reported: max |kappa - 1/R| over all contact cells and shifts,
#         observed orders, and the minimiser of the error over a fine sweep of x_e.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-contact-acute/scripts/sharp_01_evaluation_point.jl
#         (from the repository root)

import TwoPhaseAMR
using Test, Printf
const AMR = TwoPhaseAMR

const RADIUS = 6.0
const SHIFTS = 8
const XE = (0.0, -0.5, -1.0, -1.5, -2.0)

function sessile_scaled(theta, cx, m)
    mesh = AMR.BlockTreeMesh(;
        base_mesh = (16m, 8m), block_size = (16m, 8m), extent = (16.0, 8.0), periodic = (false, false),
    )
    cy = -RADIUS * cos(theta)
    return AMR.InterfaceField(
        mesh, (x, y) -> hypot(x - cx, y - cy) - RADIUS; contact_angle = (nothing, nothing, theta, nothing),
    )
end

function wall_column(V, ii, s)
    (1 <= ii <= size(V, 1)) || return nothing
    total = 0.0; previous = 1.0
    for k in 1:4
        v = s > 0 ? V[ii, k] : 1 - V[ii, k]
        v <= previous + 64eps() || return nothing
        total += v; previous = v
    end
    previous <= 64eps() || return nothing
    return total
end

# curvature of (4.19) for heights (cells), unit width, slope evaluated at x_e (cells from the first column)
function kappa_at(W0, W1, T, s, xe)
    a2 = ((W1 - W0)^2 - T^2) / (W0 + W1 + sqrt(4W0 * W1 + T^2))
    a1 = (W1 - W0) - a2
    return -s * 2a2 / (1 + (a1 + 2a2 * xe)^2)^1.5
end

function study(theta, m, xs = XE)
    T = abs(tan(theta)); s = cos(theta) > 0 ? 1 : -1
    errors = Dict(xe => 0.0 for xe in xs)
    consistency = 0.0
    cells = 0
    h = 1 / m
    for k in 0:(SHIFTS - 1)
        # place the right contact point x_r = cx + R sin(theta) at the fraction (k+1/2)/SHIFTS of a cell
        xr = (floor(Int, 11m) + (k + 0.5) / SHIFTS) * h
        cx = xr - RADIUS * sin(theta)
        field = sessile_scaled(theta, cx, m)
        V = [AMR._get(field.conditional_fraction, AMR.CellRef(1, i, j)) for i in 1:(16m), j in 1:6]
        for i in 1:(16m)
            0 < V[i, 1] < 1 || continue
            AMR._plic_touches_wall(field, AMR.CellRef(1, i, 1), 3) || continue
            side = (i - 0.5) * h > cx ? 1 : -1
            d = -s * side
            W0 = wall_column(V, i + d, s); W1 = wall_column(V, i + 2d, s)
            (W0 === nothing || W1 === nothing || !(W1 > W0 > 0)) && continue
            cells += 1
            for xe in xs
                kappa = kappa_at(W0, W1, T, s, xe) * m            # cells -> physical units
                errors[xe] = max(errors[xe], abs(kappa - 1 / RADIUS))
            end
            consistency = max(consistency, abs(AMR._get(field.curvature, AMR.CellRef(1, i, 1)) - kappa_at(W0, W1, T, s, -1.0) * m))
        end
    end
    return errors, consistency, cells
end

@testset "evaluation point of the contact parabola: errors and observed orders" begin
    println("  theta  m   cells  | max|kappa-1/R| for x_e/w = ", join((@sprintf("%5.1f", xe) for xe in XE), "  "), " | order(x_e=0)  order(x_e=-1)")
    summary = Dict()
    for degrees in (10, 20, 30, 150, 160, 170)
        theta = deg2rad(degrees)
        previous = nothing
        for m in (1, 2, 4, 8, 16)
            errors, consistency, cells = study(theta, m)
            @test consistency <= 1.0e-8                              # the independent x_e = -1 value is the code's curvature
            orders = previous === nothing ? ("  -  ", "  -  ") :
                (@sprintf("%5.2f", log2(previous[0.0] / errors[0.0])), @sprintf("%5.2f", log2(previous[-1.0] / errors[-1.0])))
            @printf("  %3d    %2d  %5d  | %s | %s  %s\n", degrees, m, cells,
                join((@sprintf("%9.2e", errors[xe]) for xe in XE), " "), orders[1], orders[2])
            summary[(degrees, m)] = errors
            previous = errors
        end
    end
    # first order is the best the thesis claims for the method; require clearly positive observed rates
    for degrees in (10, 20, 30, 150, 160, 170)
        e0 = [summary[(degrees, m)][0.0] for m in (1, 2, 4, 8, 16)]
        e1 = [summary[(degrees, m)][-1.0] for m in (1, 2, 4, 8, 16)]
        @test e0[end] < e0[1] && e1[end] < e1[1]
        @test log2(e0[2] / e0[end]) / 3 > 0.5            # mean observed order between m = 2 and m = 16
        @test log2(e1[2] / e1[end]) / 3 > 0.5
    end
end

@testset "which evaluation point is best? sweep of x_e at m = 16" begin
    xs = collect(-2.0:0.05:0.5)
    println("  theta   best x_e/w   error at best   error at x_e = 0 (thesis)   error at x_e = -1 (code)")
    for degrees in (10, 20, 30, 150, 160, 170)
        errors, _, _ = study(deg2rad(degrees), 16, xs)
        best = argmin(xe -> errors[xe], xs)
        @printf("  %3d     %6.2f       %9.2e      %9.2e                   %9.2e\n", degrees, best, errors[best], errors[0.0], errors[-1.0])
        @test errors[best] <= min(errors[0.0], errors[-1.0])
        @test -2.0 < best < 0.0
    end
end
