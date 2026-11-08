# Step 01: symbolic moment tensor of EVERY vertex-patch configuration (Symbolics.jl).
#
# Label:  eq:setting:patch-metric
# Claim:  03_discrete_setting.tex:304-305 -- Y_v' N_v is symmetric on every patch of square
#         cells (2:1-graded quadtree). Also the consequence R_v Y_v = N_v (261-266).
# Source: Y_v rows |f| d_f' with d_f = centre(plus) - centre(minus) (src/operators/gradient.jl:79,
#         _local_inverse_metric 139-184), N_v rows (1/2) e_axis(f)'.
# Configurations (the complete list, see step03 for the exhaustive proof):
#         * 16 corner patterns: four cells with a corner at v, sides in {h, 2h} (NE,NW,SW,SE);
#         * 4 hanging vertices: coarse cell on W/E/S/N, two fine cells opposite.
#         The cells are lattice cells (model.jl); the physical cell widths are the SYMBOLS hx, hy.
# Checks (all symbolic in hx, hy, r):
#   (a) T = Y'N computed from the faces equals the closed form (1/4)[hx hy F11, hx^2 F12; hy^2 F21, hx hy F22];
#   (b) T12 - T21 = (r^2-1)/4 hy^2 (sNE + sSW - sNW - sSE) with hx = r hy (sigma = 1 for a coarse cell):
#       identically ZERO for hx = hy; hanging: T12 = T21 = 0 for every aspect ratio;
#   (c) square cells, h symbolic: R Y - N = 0 for the matrix of eq:setting:patch-metric,
#       R = N sym(T)^{-1} N' + stab * (I - Y (Y'Y)^{-1} Y'), stab = tr(first term)/(#F - 2).
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-setting-patch-metric/scripts/step01_configurations_symbolic.jl

using Symbolics
using Test
using LinearAlgebra: I
isdefined(Main, :PatchModel) || include(joinpath(@__DIR__, "model.jl"))
using .PatchModel

@variables hx hy r h

"""Is the symbolic expression identically zero? (expand/simplify, then cancel fractions)"""
function is_zero(x)
    y = Symbolics.expand(Num(x))
    for f in (Symbolics.simplify, Symbolics.simplify_fractions)
        v = Symbolics.value(f(y))
        v isa Number && return iszero(v)
    end
    return false
end

"""Symbolic Y (nf x 2) and N (nf x 2) of the patch of `cells` at v = (0,0), cell widths (wx, wy)."""
function symbolic_YN(cells, wx, wy)
    wx, wy = Num(wx), Num(wy)                              # keep every coefficient exact (no Float64)
    faces = faces_at_vertex(cells, 0, 0)
    nf = length(faces)
    Y = Matrix{Num}(undef, nf, 2)
    N = Matrix{Num}(undef, nf, 2)
    for (row, f) in enumerate(faces)
        area = f.len * (f.axis == 1 ? wy : wx)             # tangential extent of the segment
        d = (f.d2[1] * wx / 2, f.d2[2] * wy / 2)           # centre(plus) - centre(minus)
        for k in 1:2
            Y[row, k] = area * d[k]
            N[row, k] = f.axis == k ? Num(1) / 2 : Num(0)
        end
    end
    return faces, Y, N
end

# element-wise arithmetic only: Matrix{Num} / Num would become a symbolic ARRAY expression
function inv2(A)
    d = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    return [A[2, 2] / d -A[1, 2] / d; -A[2, 1] / d A[1, 1] / d]
end

"""R of eq:setting:patch-metric for symbolic Y, N."""
function patch_metric(Y, N)
    nf = size(Y, 1)
    T = transpose(Y) * N
    S = (T .+ transpose(T)) ./ 2
    first_term = N * inv2(S) * transpose(N)
    stab = sum(first_term[i, i] for i in 1:nf) / (nf - 2)
    P = Matrix{Num}(I, nf, nf) .- Y * inv2(transpose(Y) * Y) * transpose(Y)
    return first_term .+ stab .* P
end

# names of the 16 corner patterns: sides (NE, NW, SW, SE) in units of the fine side
corner_patterns() = [(a, b, c, d) for a in (1, 2), b in (1, 2), c in (1, 2), d in (1, 2)]

@testset "step01: symbolic patch tensors" begin
    @testset "corner patterns (16)" begin
        for pattern in corner_patterns()
            sNE, sNW, sSW, sSE = pattern
            cells = corner_cells(sNE, sNW, sSW, sSE)
            faces, Y, N = symbolic_YN(cells, hx, hy)
            @test length(faces) == 4
            T = transpose(Y) * N
            Tform = physical_moment(moment_integers(faces), hx, hy)
            @test all(is_zero(T[i, j] - Tform[i, j]) for i in 1:2, j in 1:2)            # (a)
            sigma = (sNE - 1, sNW - 1, sSW - 1, sSE - 1)
            asym = T[1, 2] - T[2, 1]
            formula = (r^2 - 1) / 4 * hy^2 * (sigma[1] + sigma[3] - sigma[2] - sigma[4])
            @test is_zero(Symbolics.substitute(asym, Dict(hx => r * hy)) - formula)       # (b)
            @test is_zero(Symbolics.substitute(asym, Dict(hx => h, hy => h)))             # square: 0
        end
    end

    @testset "hanging vertices (4)" begin
        for side in (:W, :E, :S, :N)
            faces, Y, N = symbolic_YN(hanging_cells(side), hx, hy)
            @test length(faces) == 3
            T = transpose(Y) * N
            Tform = physical_moment(moment_integers(faces), hx, hy)
            @test all(is_zero(T[i, j] - Tform[i, j]) for i in 1:2, j in 1:2)
            @test is_zero(T[1, 2]) && is_zero(T[2, 1])        # symmetric for every aspect ratio
        end
    end

    @testset "R Y = N for square cells (h symbolic), all 20 configurations" begin
        configurations = Pair{Any, Any}[]
        for pattern in corner_patterns()
            push!(configurations, pattern => corner_cells(pattern...))
        end
        for side in (:W, :E, :S, :N)
            push!(configurations, side => hanging_cells(side))
        end
        for (name, cells) in configurations
            faces, Y, N = symbolic_YN(cells, h, h)
            R = patch_metric(Y, N)
            residual = R * Y - N
            @test all(is_zero(residual[i, j]) for i in axes(residual, 1), j in 1:2)
        end
    end

    @testset "R Y = N for stretched cells where Y'N is symmetric (hanging, CCFF, FFFF)" begin
        # hy = 1, hx = r symbolic (the metric scales homogeneously)
        for cells in (hanging_cells(:W), hanging_cells(:N), corner_cells(2, 2, 1, 1), corner_cells(2, 1, 1, 2), corner_cells(1, 1, 1, 1))
            faces, Y, N = symbolic_YN(cells, r, 1)
            R = patch_metric(Y, N)
            residual = R * Y - N
            @test all(is_zero(residual[i, j]) for i in axes(residual, 1), j in 1:2)
        end
    end
end
