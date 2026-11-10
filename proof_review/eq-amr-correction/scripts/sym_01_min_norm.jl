# Symbolic check 01: the four-cell flux system of eq:amr:correction.
#
# Label: eq:amr:correction
# Source: docs/techref/chapters/12_amr_criteria_transfer.tex:527-552
# Checks: (S1) the children's outflow changes of the chapter (F12+F13, -F12+F24, F34-F13, -F34-F24) solve A F = delta for the chapter's
#         F12, F34, F13, F24 when delta4 = -delta1-delta2-delta3;  (S2) the null space of A is spanned by (1,-1,-1,1) (a circulation
#         around the centre vertex), rank A = 3;  (S3) the chapter's F is orthogonal to the null vector, hence the minimum Euclidean-norm
#         solution F = A^+ delta;  (S4) in the velocity variable F/|f| the same F is minimal only when the four internal faces have equal
#         area (square children); for rectangular children it is minimal for the integrated flux, not for the velocity increments.
# Run:    julia --project=@claude-symbolics proof_review/eq-amr-correction/scripts/sym_01_min_norm.jl

using Symbolics, LinearAlgebra, Test

@variables d1 d2 d3
d4 = -d1 - d2 - d3
d = [d1, d2, d3, d4]
# columns: faces 12, 34, 13, 24; rows: children 1..4 (outflow change of child r when the face flux F_f increases, f oriented +x / +y)
A = [1 0 1 0; -1 0 0 1; 0 1 -1 0; 0 -1 0 -1]
F12 = (2d1 - d2 + d3) / 4
F34 = d1 + d3 - F12
F13 = d1 - F12
F24 = d2 + F12
F = [F12, F34, F13, F24]

@testset "eq:amr:correction (symbolic)" begin
    residual = Symbolics.simplify.(Symbolics.expand.(A * F - d))
    @test all(isequal(0), residual)                                  # S1
    n = [1, -1, -1, 1]
    @test A * n == zeros(Int, 4)                                     # S2
    @test rank(Float64.(A)) == 3
    @test isequal(Symbolics.simplify(Symbolics.expand(dot(F, n))), 0)  # S3: F orthogonal to the null space
    # S3b: the same F from the Moore-Penrose pseudo-inverse with exact rationals
    P = pinv(Float64.(A))
    for _ in 1:5
        v = randn(3); dd = [v; -sum(v)]
        Fnum = [Symbolics.value(substitute(f, Dict(d1 => v[1], d2 => v[2], d3 => v[3]))) for f in F]
        @test maximum(abs, Fnum .- P * dd) < 1.0e-13
    end
    # S4: velocity-variable minimisation with face areas a12,a34 (normal 1) and a13,a24 (normal 2): min sum (F/a)^2
    # => the minimiser is F = W A' (A W A')^+ d with W = diag(a^2).  For equal areas it is the chapter's F, otherwise not.
    for (a, equal) in (([1.0, 1.0, 1.0, 1.0], true), ([1.0, 1.0, 2.0, 2.0], false))
        W = Diagonal(a .^ 2)
        v = randn(3); dd = [v; -sum(v)]
        Fw = W * A' * pinv(A * W * A') * dd
        Fnum = [Symbolics.value(substitute(f, Dict(d1 => v[1], d2 => v[2], d3 => v[3]))) for f in F]
        @test (maximum(abs, Fw .- Fnum) < 1.0e-12) == equal
    end
end
