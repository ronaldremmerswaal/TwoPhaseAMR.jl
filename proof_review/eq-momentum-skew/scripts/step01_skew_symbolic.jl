# Step 01 (symbolic): skew-symmetry of the centred staggered convection operator and its energy consequences.
#
# Label:  eq:momentum:skew  (with eq:momentum:operator, eq:momentum:dual-divergence, eq:momentum:euler-energy)
# Source: docs/techref/chapters/06_momentum_transport.tex:315-330 (operator, boundary dual faces), 486-512 (skew identity)
# Checks (Symbolics.jl), on an arbitrary dual graph (5 unknowns, 8 dual faces with arbitrary orientation, a pair joined by two
# dual faces, arbitrary rates and volumes):
#   S1  <psi, C q>_V + <q, C psi>_V = sum_f V_f q_f psi_f (div U)_f   and   2 dC_ff/dq_f = (div U)_f      (eq:momentum:skew),
#       for the centred state (q_tail + q_head)/2 and (div U)_f = V_f^{-1} sum_g sigma_{f,g} U_g.
#   S2  The arithmetic mean is the only two-point state with this property: for a state alpha q_tail + (1 - alpha) q_head the
#       off-diagonal symmetric part of V C of a pair of unknowns is sum_g U_g (1 - 2 alpha_g) (signed by the orientation), so a
#       single dual face of non-zero rate forces alpha = 1/2.
#   S3  Boundary dual faces (walls): with the normal-component state (q_f + q_b)/2 of eq:momentum:boundary and boundary values
#       q_b = 0 the sum  <psi,Cq> + <q,C psi>  equals  sum_f V_f q_f psi_f [ (div U)_f + V_f^{-1} sum_{b in B(f)} sigma_b U_b ]:
#       the right-hand side of eq:momentum:skew needs the boundary dual faces in its divergence (they are not in (div U)_f of
#       eq:momentum:dual-divergence); with q_b != 0 the operator is affine and an extra term sum_b sigma_b U_b (q_b psi_f +
#       psi_b q_f)/2 appears.
#   S4  Forward Euler: E1 - E0 = -dt <q, C q> + (dt^2/2) <C q, C q>, and <q, C q> = (1/2) sum V q^2 div U, so E1 - E0 =
#       (dt^2/2) |C q|_V^2 >= 0 for div U = 0 (eq:momentum:euler-energy).
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-momentum-skew/scripts/step01_skew_symbolic.jl

using Symbolics, Test

expanded(e) = Symbolics.simplify(Symbolics.expand(e))
iszero_sym(e) = isequal(expanded(e), 0)

const N = 5
# dual faces (tail, head); (1, 2) appears twice (two axes joining the same pair), orientations are arbitrary
const EDGES = [(1, 2), (2, 3), (4, 3), (4, 5), (5, 1), (1, 3), (2, 5), (1, 2)]
@variables V[1:N] q[1:N] psi[1:N]
@variables U[1:length(EDGES)] alpha[1:length(EDGES)]
@variables Ub[1:N] qb[1:N] psib[1:N]            # boundary dual face of every unknown (side orientation +1), boundary values

sigma(i, g) = EDGES[g][1] == i ? 1 : (EDGES[g][2] == i ? -1 : 0)

"""V_f (C q)_f of the interior dual faces with face state `state(g, q)`."""
volume_C(i, state, x) = sum(sigma(i, g) * U[g] * state(g, x) for g in eachindex(EDGES))
centred(g, x) = (x[EDGES[g][1]] + x[EDGES[g][2]]) / 2
weighted(g, x) = alpha[g] * x[EDGES[g][1]] + (1 - alpha[g]) * x[EDGES[g][2]]

inner(a, b) = sum(a[i] * b[i] * V[i] for i in 1:N)                     # < , >_V
divU(i) = sum(sigma(i, g) * U[g] for g in eachindex(EDGES)) / V[i]

@testset "eq:momentum:skew symbolic" begin
    @testset "S1 skew-symmetry up to half the dual divergence on the diagonal" begin
        Cq(i, x) = volume_C(i, centred, x) / V[i]
        lhs = sum(psi[i] * V[i] * Cq(i, q) for i in 1:N) + sum(q[i] * V[i] * Cq(i, psi) for i in 1:N)
        rhs = sum(V[i] * q[i] * psi[i] * divU(i) for i in 1:N)
        @test iszero_sym(lhs - rhs)
        for i in 1:N
            @test iszero_sym(2 * Symbolics.derivative(Cq(i, q), q[i]) - divU(i))
        end
        # off-diagonal part of V C is skew
        VC(i, j) = Symbolics.derivative(volume_C(i, centred, q), q[j])
        for i in 1:N, j in 1:N
            i == j && continue
            @test iszero_sym(VC(i, j) + VC(j, i))
        end
    end
    @testset "S2 the arithmetic mean is necessary (two-point states, nonzero rate)" begin
        VCw(i, j) = Symbolics.derivative(volume_C(i, weighted, q), q[j])
        # a pair joined by one dual face only: (2,3) is edge 2; symmetric off-diagonal part = U (1 - 2 alpha)
        @test iszero_sym(VCw(2, 3) + VCw(3, 2) - U[2] * (1 - 2 * alpha[2]))
        @test iszero_sym(VCw(4, 3) + VCw(3, 4) - U[3] * (1 - 2 * alpha[3]))        # tail 4, head 3: the symmetric sum is orientation-free
        # alpha = 1/2 removes it
        @test iszero_sym(substitute(VCw(2, 3) + VCw(3, 2), Dict(alpha[2] => 1 // 2)))
        # the pair (1,2) is joined by two faces: only the combination U1 (1 - 2 alpha1) + U8 (1 - 2 alpha8) matters
        @test iszero_sym(VCw(1, 2) + VCw(2, 1) - (U[1] * (1 - 2 * alpha[1]) + U[8] * (1 - 2 * alpha[8])))
    end
    @testset "S3 boundary dual faces: the divergence in the identity must include them" begin
        # one boundary dual face per unknown, flux Ub[i], orientation +1; state (q_i + qb_i)/2
        boundary_term(i, x, xb) = Ub[i] * (x[i] + xb[i]) / 2
        VCq(i, x, xb) = volume_C(i, centred, x) + boundary_term(i, x, xb)
        lhs = sum(psi[i] * VCq(i, q, qb) for i in 1:N) + sum(q[i] * VCq(i, psi, psib) for i in 1:N)
        zero_b = merge(Dict(qb[i] => 0 for i in 1:N), Dict(psib[i] => 0 for i in 1:N))
        closed(i) = divU(i) + Ub[i] / V[i]
        Vdiv(i) = sum(sigma(i, g) * U[g] for g in eachindex(EDGES))                   # V_f (div U)_f
        rhs0 = sum(q[i] * psi[i] * (Vdiv(i) + Ub[i]) for i in 1:N)                    # = sum V q psi closed(i)
        @test iszero_sym(substitute(lhs, zero_b) - rhs0)
        # the printed right-hand side (interior dual faces only) misses exactly the boundary part
        printed = sum(q[i] * psi[i] * Vdiv(i) for i in 1:N)                           # = sum V q psi div U
        @test iszero_sym(substitute(lhs, zero_b) - printed - sum(q[i] * psi[i] * Ub[i] for i in 1:N))
        # non-zero boundary values: the extra affine term
        extra = sum(Ub[i] * (qb[i] * psi[i] + psib[i] * q[i]) / 2 for i in 1:N)
        @test iszero_sym(lhs - rhs0 - extra)
        # the diagonal: 2 dC_ff/dq_f = div U + Ub/V (boundary values zero or not: independent of qb)
        for i in 1:N
            @test iszero_sym(2 * Symbolics.derivative(VCq(i, q, qb) / V[i], q[i]) - closed(i))
        end
    end
    @testset "S4 forward Euler energy growth" begin
        @variables dt
        Cq(i, x) = volume_C(i, centred, x) / V[i]
        q1 = [q[i] - dt * Cq(i, q) for i in 1:N]
        E1 = inner(q1, q1) / 2
        E0 = inner(q, q) / 2
        qCq = sum(q[i] * V[i] * Cq(i, q) for i in 1:N)
        CqCq = sum(V[i] * Cq(i, q)^2 for i in 1:N)
        @test iszero_sym((E1 - E0) - (-dt * qCq + dt^2 / 2 * CqCq))
        @test iszero_sym(qCq - sum(V[i] * q[i]^2 * divU(i) for i in 1:N) / 2)
        # a divergence-free rate vector: a circulation 1 -> 2 -> 3 -> 1 along edges 1 = (1,2), 2 = (2,3) and 6 = (1,3) reversed
        @variables c
        div_free = Dict{Any, Any}(U[g] => 0 for g in eachindex(EDGES))
        div_free[U[1]] = c
        div_free[U[2]] = c
        div_free[U[6]] = -c
        for i in 1:N
            @test iszero_sym(substitute(divU(i) * V[i], div_free))
        end
        E1d = substitute(E1 - E0, div_free)
        growth = substitute(dt^2 / 2 * CqCq, div_free)
        @test iszero_sym(E1d - growth)
    end
end
