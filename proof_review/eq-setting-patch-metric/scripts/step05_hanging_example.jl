# Step 05 (side check): the worked hanging-vertex example of eq:setting:patch-example.
#
# Label:  eq:setting:patch-metric / eq:setting:patch-example
# Source: 03_discrete_setting.tex:273-288 (T = diag(3/2, 1/2), stab = 5/6, K = [4/9 -1/9 5/18; ...])
#         and docs/techref/figures/generate.jl:397 (expected K).
# Method: exact rational arithmetic (Rational{Int}); R = N S^{-1} N' + stab (I - Y (Y'Y)^{-1} Y').
#         Also the two-point value 1/(2 dualvol(f1)) = 1/3 quoted in the text.
# Run:    julia --heap-size-hint=3G -t 1 proof_review/eq-setting-patch-metric/scripts/step05_hanging_example.jl

using Test
using LinearAlgebra: I, tr, inv
isdefined(Main, :PatchModel) || include(joinpath(@__DIR__, "model.jl"))
using .PatchModel

@testset "step05: hanging-vertex example" begin
    # fine cell width one, coarse cell on the west: f1 (lower), f2 (upper), g (between the fine cells)
    cells = hanging_cells(:W)
    faces = faces_at_vertex(cells, 0, 0)
    # order (f1, f2, g) as in the text: f1 = W/SE, f2 = W/NE, g = SE/NE
    f1 = only(f for f in faces if f.axis == 1 && f.d2[2] < 0)
    f2 = only(f for f in faces if f.axis == 1 && f.d2[2] > 0)
    g = only(f for f in faces if f.axis == 2)
    ordered = [f1, f2, g]
    Y = Rational{Int}[f.len * f.d2[k] // 2 for f in ordered, k in 1:2]       # unit lengths and widths
    N = Rational{Int}[f.axis == k ? 1 // 2 : 0 // 1 for f in ordered, k in 1:2]
    @test Y == [3 // 2 -1 // 2; 3 // 2 1 // 2; 0 // 1 1 // 1]                            # displacements of the text
    T = transpose(Y) * N
    @test T == [3 // 2 0 // 1; 0 // 1 1 // 2]                                        # diag(3/2, 1/2)
    S = (T + transpose(T)) / 2
    first_term = N * inv(S) * transpose(N)
    stab = tr(first_term) / (3 - 2)
    @test stab == 5 // 6
    P = Matrix{Rational{Int}}(I, 3, 3) - Y * inv(transpose(Y) * Y) * transpose(Y)
    R = first_term + stab * P
    @test R == [4 // 9 -1 // 9 5 // 18; -1 // 9 4 // 9 -5 // 18; 5 // 18 -5 // 18 7 // 9]
    @test R * Y == N
    # two-point value of the text: dualvol(f1) = |f1| (h_coarse + h_fine)/2 = 3/2
    @test 1 // (2 * (1 * (2 + 1) // 2)) == 1 // 3
end
