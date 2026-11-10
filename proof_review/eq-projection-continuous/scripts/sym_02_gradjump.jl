# Symbolic check 02: the continuous velocity update and the gradient-jump condition (eq:projection:cont-update, eq:projection:cont-gradjump).
#
# Labels: eq:projection:cont-update, eq:projection:cont-gradjump  (thesis (7.8), (7.9); pt2_poisson.tex:14-18)
# Source: docs/techref/chapters/10_projection.tex:20-40
# Claim: u^{π,n+1} = u^{π,*} - (dt/ρ^π) ∇p^π  together with [[u^{n+1}·η]] = 0  (gas minus liquid) is equivalent to
#        [[ (1/ρ) η·∇p ]] = [[u*·η]] / dt.
# Method: Symbolics; the normal components are scalars (η·u^π, η·∇p^π); equivalence of the two equalities (difference is dt^{-1} times the constraint).
# Run:    julia --project=@claude-symbolics proof_review/eq-projection-continuous/scripts/sym_02_gradjump.jl

using Symbolics, Test
@variables dt ρl ρg ul ug gl gg          # ul = η·u*^l, ug = η·u*^g, gl = η·∇p^l, gg = η·∇p^g
unl = ul - dt * gl / ρl
ung = ug - dt * gg / ρg
continuity = ung - unl                                  # [[u^{n+1}·η]]
gradjump = (gg / ρg - gl / ρl) - (ug - ul) / dt        # [[ (1/ρ) ∂_η p ]] - [[u*·η]]/dt
@testset "gradjump" begin
    @test isequal(Symbolics.simplify(Symbolics.expand(continuity + dt * gradjump)), 0)    # continuity = -dt * gradjump
    # sign: gas minus liquid in both; the opposite convention [[·]] = liquid - gas gives the same equation (both sides change sign)
    @test isequal(Symbolics.simplify(Symbolics.expand((unl - ung) - (-continuity))), 0)
end
println("sym_02_gradjump: OK")
