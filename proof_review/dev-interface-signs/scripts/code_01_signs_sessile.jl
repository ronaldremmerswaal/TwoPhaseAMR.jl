# Code check 01: the sign conventions of the contact-curvature formulas against the package and
# against counterfactuals (the printed thesis signs).
#
# Label:  dev:interface:signs
# Claim:  04_interface.tex:526-537: "The code follows the definitions; its tests confirm kappa -> 1/R > 0
#         for sessile drops at five contact angles. The printed [thesis] signs appear to be typographical
#         errors."; (4.3) eta.n_w = cos(theta); (4.18) right-angle ghost value; (4.19) acute branch.
# Code:   src/interface/height_functions.jl:733-804 (_contact_curvature).
# Method: (A) geometry of the exact circle (no package): eta.n_w = cos(theta) at both contact points, the
#             wall slope of the height function equals -cos(theta)/(s sin(theta)), div eta = +1/R;
#         (B) sessile drops at theta = 20, 55, 70, 90, 125, 160 degrees and resolutions m = 1..8 (R = 6m cells):
#             the stored contact curvature of every contact cell equals the independent evaluation of (4.18)/(4.19)
#             and converges to +1/R;
#         (C) counterfactuals with the printed thesis signs: overall sign of the curvature flipped (4.92),
#             ghost value with the opposite slope sign (thesis: dH/dx = +1/tan(theta)), and the acute branch
#             with +2A/(...): none of them converges to +1/R (the last one is right for obtuse angles only).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/dev-interface-signs/scripts/code_01_signs_sessile.jl
#         (from the repository root)

import TwoPhaseAMR
using Test, Printf
const AMR = TwoPhaseAMR

# --- (A) exact circle ------------------------------------------------------------------------
@testset "(A) exact sessile circle: (4.3), wall slope of the height function, kappa = +1/R" begin
    R = 6.0
    for degrees in (20, 55, 70, 90, 125, 160)
        theta = deg2rad(degrees)
        cx, cy = 8.0, -R * cos(theta)                         # the wall is y = 0, fluid above (n_w = +e_y)
        for (side, xc) in ((+1, cx + R * sin(theta)), (-1, cx - R * sin(theta)))
            eta = ((xc - cx) / R, (0 - cy) / R)               # outward liquid normal = into the gas
            @test isapprox(eta[2], cos(theta); atol = 1.0e-14)               # (4.3): eta.n_w = cos(theta)
            s = sign(eta[1])                                   # principal direction of the horizontal height function
            @test s == side
            # horizontal height function x = H(y) of the contact side; its derivative along the inward direction y
            H(y) = cx + side * sqrt(R^2 - (y - cy)^2)
            dH = (H(1.0e-6) - H(-1.0e-6)) / 2.0e-6
            @test isapprox(dH, -cos(theta) / (s * sin(theta)); atol = 1.0e-7)                # the doc (4.18)
            theta == pi / 2 || @test !isapprox(dH, cos(theta) / (s * sin(theta)); atol = 1.0e-3)   # not the thesis' printed sign
            # kappa = div eta = +1/R for the outward normal eta = (x - cx, y - cy)/|(x - cx, y - cy)| (central differences)
            etafield(x, y) = ((x - cx, y - cy) ./ hypot(x - cx, y - cy))
            d = 1.0e-5
            divergence = (etafield(xc + d, 0.0)[1] - etafield(xc - d, 0.0)[1]) / 2d + (etafield(xc, d)[2] - etafield(xc, -d)[2]) / 2d
            @test isapprox(divergence, 1 / R; rtol = 1.0e-8)
        end
    end
end

# --- (B), (C) pipeline and counterfactuals ---------------------------------------------------------

function sessile(theta, R, cx, nx, ny)
    mesh = AMR.BlockTreeMesh(;
        base_mesh = (nx, ny), block_size = (nx, ny), extent = (Float64(nx), Float64(ny)), periodic = (false, false),
    )
    cy = -R * cos(theta)
    return AMR.InterfaceField(mesh, (x, y) -> hypot(x - cx, y - cy) - R; contact_angle = (nothing, nothing, theta, nothing))
end

function column_height(V, i, j, axis, s; radius = 3)
    tol = 64eps(); total = 0.0; first = nothing; previous = 1.0
    for k in -radius:radius
        ii, jj = axis == 1 ? (i + k, j) : (i, j + k)
        available = checkbounds(Bool, V, ii, jj) && isfinite(V[ii, jj])
        if !available
            isnothing(first) && continue
            break
        end
        v = s > 0 ? V[ii, jj] : 1 - V[ii, jj]
        if isnothing(first)
            abs(v - 1) <= tol || return nothing
            first = k
        else
            v > previous + tol && return nothing
            tol < v < 1 - tol && abs(v - previous) <= tol && return nothing
        end
        total += v; previous = v
    end
    (isnothing(first) || abs(previous) > tol) && return nothing
    return total + first - 0.5
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

# right-angle branch (4.18), unit cells; ghost_sign = -1 / overall = -1 give the printed thesis signs
function right_branch(V, i, theta, s; ghost_sign = 1, overall = 1)
    H0 = column_height(V, i, 1, 1, s); H1 = column_height(V, i, 2, 1, s)
    (H0 === nothing || H1 === nothing) && return nothing
    u = -cos(theta) / (s * sin(theta))
    Hg = H0 - ghost_sign * u
    slope = (H1 - Hg) / 2
    return -overall * s * (H1 - 2H0 + Hg) / (1 + slope^2)^1.5
end

# defect of (4.3) for the normal built from the CENTRED slope (H1 - H_ghost)/2 used inside kappa: |eta.n_w - cos(theta)|
function right_branch_centred_defect(V, i, theta, s)
    H0 = column_height(V, i, 1, 1, s); H1 = column_height(V, i, 2, 1, s)
    (H0 === nothing || H1 === nothing) && return nothing
    u = -cos(theta) / (s * sin(theta))
    slope = (H1 - (H0 - u)) / 2                            # centred slope along the inward normal
    return abs(-s * slope / sqrt(1 + slope^2) - cos(theta))
end

# acute/obtuse branch (4.19), unit cells; thesis = true uses +2A/(...) for both orientations
function acute_branch(V, i, theta, s, side; thesis = false)
    d = -s * side
    W0 = wall_column(V, i + d, s); W1 = wall_column(V, i + 2d, s)
    (W0 === nothing || W1 === nothing || !(W1 > W0 > 0)) && return nothing
    T = abs(tan(theta))
    a2 = ((W1 - W0)^2 - T^2) / (W0 + W1 + sqrt(4W0 * W1 + T^2))
    a1 = (W1 - W0) - a2
    return (thesis ? 1 : -s) * 2a2 / (1 + (a1 - 2a2)^2)^1.5
end

function study(degrees, m)
    theta = deg2rad(degrees)
    R = 6.0m; cx = 8.2m
    field = sessile(theta, R, cx, 16m, 8m)
    V = [AMR._get(field.conditional_fraction, AMR.CellRef(1, i, j)) for i in 1:(16m), j in 1:6]
    right = abs(theta - pi / 2) <= pi / 4
    errs = Dict(:doc => 0.0, :flipped_sign => 0.0, :flipped_ghost => 0.0, :thesis_acute => 0.0, :centred_defect => 0.0)
    consistency = 0.0; cells = 0; negative = 0
    for i in 1:(16m)
        0 < V[i, 1] < 1 || continue
        AMR._plic_touches_wall(field, AMR.CellRef(1, i, 1), 3) || continue
        side = i - 0.5 > cx ? 1 : -1
        s = right ? side : (cos(theta) > 0 ? 1 : -1)               # principal direction (right branch) / wedge phase (acute)
        if right
            k = right_branch(V, i, theta, side)
            k === nothing && continue
            kflip = right_branch(V, i, theta, side; overall = -1)
            kghost = right_branch(V, i, theta, side; ghost_sign = -1)
            errs[:flipped_sign] = max(errs[:flipped_sign], abs(kflip * R - 1))
            errs[:flipped_ghost] = max(errs[:flipped_ghost], abs(kghost * R - 1))
            errs[:centred_defect] = max(errs[:centred_defect], right_branch_centred_defect(V, i, theta, side))
        else
            k = acute_branch(V, i, theta, s, side)
            k === nothing && continue
            kthesis = acute_branch(V, i, theta, s, side; thesis = true)
            errs[:flipped_sign] = max(errs[:flipped_sign], abs(-k * R - 1))
            errs[:thesis_acute] = max(errs[:thesis_acute], abs(kthesis * R - 1))
        end
        stored = AMR._get(field.curvature, AMR.CellRef(1, i, 1))
        consistency = max(consistency, abs(stored - clamp(k, -0.5, 0.5)))         # clip (4.17) with h = 1
        errs[:doc] = max(errs[:doc], abs(k * R - 1))
        negative += k < 0
        cells += 1
    end
    return errs, consistency, cells, negative
end

@testset "(B)/(C) sessile drops: the code's signs converge to +1/R, the printed thesis signs do not" begin
    println("  theta  branch   m   cells | rel.err |kappa R - 1|: doc    flipped sign   flipped ghost   thesis acute (+2A) | defect of (4.3) for the centred-slope normal")
    table = Dict()
    for degrees in (20, 55, 70, 90, 125, 160)
        for m in (1, 2, 4, 8)
            errs, consistency, cells, negative = study(degrees, m)
            @test consistency <= 1.0e-10 * max(1, m)                    # the stored curvature is the independent value
            @test negative == 0                                         # kappa > 0: convex liquid
            branch = abs(deg2rad(degrees) - pi / 2) <= pi / 4 ? "right " : "acute "
            @printf(
                "  %3d    %s  %2d   %3d   | %-12.3e %-14.3e %-15.3e %-18.3e | %.3e\n", degrees, branch, m, cells,
                errs[:doc], errs[:flipped_sign], errs[:flipped_ghost], errs[:thesis_acute], errs[:centred_defect]
            )
            table[(degrees, m)] = errs
        end
        e = [table[(degrees, m)][:doc] for m in (1, 2, 4, 8)]
        @test e[end] < e[1] / 2 && e[end] < 0.05                                  # converges to +1/R
        if abs(deg2rad(degrees) - pi / 2) <= pi / 4
            @test table[(degrees, 8)][:flipped_sign] > 1.9                        # -kappa -> -1/R: error 2
            degrees == 90 || @test table[(degrees, 8)][:flipped_ghost] > 0.5       # opposite ghost slope: no convergence
            # the normal inside kappa (centred slope) satisfies (4.3) only to O(h): the defect decreases ~ 1/m
            degrees == 90 || @test table[(degrees, 8)][:centred_defect] < table[(degrees, 1)][:centred_defect] / 3
        else
            degrees < 90 && @test table[(degrees, 8)][:thesis_acute] > 1.9        # printed +2A is -1/R for the liquid wedge
            degrees > 90 && @test isapprox(table[(degrees, 8)][:thesis_acute], table[(degrees, 8)][:doc]; rtol = 1.0e-9)  # right for obtuse
        end
    end
end
