# Step 01 (symbolic): term-wise upwinding of partial volumes (eq:momentum:ctu) versus net upwinding, and what bounds the update.
#
# Label:  eq:momentum:ctu  (with eq:momentum:weights, eq:momentum:one-velocity, eq:momentum:mass-identity)
# Source: docs/techref/chapters/06_momentum_transport.tex:652-690 (modified flux interpolant) and the thesis (6.49)-(6.51), Lemmas 6.2, 6.3
# Checks (Symbolics.jl):
#   S1  One dual face with partial volumes p_i (positive part P = sum_{p_i>0} p_i, negative part N = sum_{p_i<0} |p_i|): the term-wise flux is
#           P q_tail - N q_head,
#       the net-volume upwind flux is (P - N) q_tail if P >= N and (P - N) q_head otherwise, and
#           CTU flux - net upwind flux = -min(P, N) (q_head - q_tail):
#       the two differ exactly when partial volumes of opposite signs cross the same dual face, by a diffusive exchange of
#       min(P, N) between the two unknowns.  Both streams are upwinded from their own side; nothing bypasses a control volume.
#   S2  Only INFLOW partial volumes contribute to the fluctuation form:
#           (1/V) sum_g sigma_{f,g} sum_i p_{g,i} (q_f - q_{u(g,i)}) = (1/V) sum_{inflow i} |p_i| (q_{nb(i)} - q_f)
#       (u = tail for p >= 0, head otherwise), so with the mass identity V rho^new = V rho^old + I_f - O_f + V R
#           V rho^new q^new = (V rho^old - O_f) q_f + sum_{in} |p_i| q_nb + V R q^n,
#       a convex combination iff O_f <= V rho^old and R >= 0 ("bounded outflow"), O_f / I_f the total outflow / inflow partial volumes.
#   S3  Closed forms of O_f and I_f from the printed rules, arbitrary hanging side partitions and every sign pattern of the fluxes of the
#       sweep axis (Phi_h = s_h x_h, x_h >= 0), with out_c / in_c the volumes leaving / entering the cell c through its faces of the sweep axis:
#           O_f = sum_c (varpi_{c,f}/2) out_c + E_f,   I_f = sum_c (varpi_{c,f}/2) in_c + E_f,
#           E_f = (2 - varpi_{c^-,f} - varpi_{c^+,f})/2 |Phi_f|   (component along the sweep),      E_f = 0   (component across the sweep).
#       On a conforming patch (varpi = 1) the outflow of a control volume is half the outflow of each of its two cells; at a hanging
#       segment an extra E_f leaves (1/4 |Phi_f| for a fine segment of a coarse side) and the same extra enters.
#   S4  Own-value coefficient gamma_f = (V rho^old - O_f) / (V rho^new) of a draining unknown.  With only the own flux of f non-zero, flowing
#       c^- -> c^+, all phase of c^- inside its donating rectangle and c^+ empty:  gamma_f = -(2 - varpi^- - varpi^+) / varpi^+,
#       i.e. 0 on a conforming patch, -1/2 for fluid leaving a coarse cell through a fine segment (varpi^- = 1/2, varpi^+ = 1) and -1 for
#       fluid entering the coarse cell (varpi^- = 1, varpi^+ = 1/2): the update is then NOT a convex combination.
# Run:    julia --heap-size-hint=3G -t 1 --project=@claude-symbolics proof_review/eq-momentum-ctu/scripts/step01_ctu_symbolic.jl

using Symbolics, Test, Random
isdefined(Main, :MomentumSymbolic) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_symbolic.jl"))
using .MomentumSymbolic

@testset "eq:momentum:ctu symbolic" begin
    @testset "S1 CTU flux = net upwind flux - min(P, N) (q_head - q_tail)" begin
        @variables P N d qt qh
        # P >= N: P = N + d; the net volume is d >= 0 and the net upwind flux d q_tail
        @test iszero_sym(((N + d) * qt - N * qh) - d * qt + N * (qh - qt))              # = -min(P, N)(qh - qt), min = N
        # P < N: N = P + d; the net volume is -d < 0 and the net upwind flux -d q_head
        @test iszero_sym((P * qt - (P + d) * qh) - (-d * qh) + P * (qh - qt))           # min = P
        # with explicit lists of partial volumes: P and N are the sums of the positive and negative parts
        @variables x1 x2 x3 x4
        positive, negative = [x1, x2], [x3, x4]
        Psum, Nsum = sum(positive), sum(negative)
        ctu = sum(x * qt for x in positive) + sum(-x * qh for x in negative)
        @test iszero_sym(ctu - (Psum * qt - Nsum * qh))
    end
    @testset "S2 only inflow partial volumes enter the fluctuation form" begin
        rng = MersenneTwister(3)
        @variables V qf q1 q2 q3
        nb = [q1, q2, q3]
        sigma = [+1, +1, -1]                                   # f is the tail of two dual faces and the head of one
        for trial in 1:30
            magnitudes = [Symbolics.variable(Symbol("a", trial, "_", g, "_", i)) for g in 1:3, i in 1:2]
            signs = rand(rng, [-1, 1], 3, 2)                   # sign of each signed partial volume p_{g,i} (orientation tail -> head)
            lhs = 0
            rhs = 0
            for g in 1:3, i in 1:2
                p = signs[g, i] * magnitudes[g, i]
                # upwind unknown of the partial volume: tail if p >= 0 else head; f is the tail iff sigma = +1
                u = signs[g, i] > 0 ? (sigma[g] == 1 ? qf : nb[g]) : (sigma[g] == 1 ? nb[g] : qf)
                lhs += sigma[g] * p * (qf - u) / V
                inflow = sigma[g] * signs[g, i] < 0
                inflow && (rhs += magnitudes[g, i] * (nb[g] - qf) / V)
            end
            @test iszero_sym(lhs - rhs)
        end
        # conservative form given the mass identity V rho_new = V rho_old + I - O + V R; S_in = sum_in |p_i| q_nb(i)
        @variables rho_old M_new I O R qn S_in                                 # M_new = V rho_new
        T = M_new * qf + S_in - I * qf + V * R * (qn - qf)                   # = V rho_new * q_new of the fluctuation form
        conservative = (V * rho_old - O) * qf + S_in + V * R * qn
        @test iszero_sym(substitute(T, Dict(M_new => V * rho_old + I - O + V * R)) - conservative)
    end
    @testset "S3 closed forms of the outflow and inflow partial volumes (random hanging topologies and sign patterns)" begin
        rng = MersenneTwister(11)
        for trial in 1:12
            patch = build_patch(rng; constrained = true)
            (; orient, trans_of) = patch
            events = events_at_f(patch)
            x = Dict{Symbol, Any}(name => Symbolics.variable(Symbol("x_", name, "_", trial)) for name in keys(all_fluxes(patch)))
            for pattern in 1:6
                s = Dict{Symbol, Int}(name => rand(rng, [-1, 1]) for name in keys(x))
                for kind in (:ax, :tr)
                    O = 0
                    I = 0
                    for (k, sg, flux, magnitude, sign) in events
                        k == kind || continue
                        (sg * sign * s[flux] > 0 ? (O += magnitude * x[flux]) : (I += magnitude * x[flux]))
                    end
                    varpi(c) = frac_of(patch, c, orient[(c, :f)], :f)
                    faces_of(c) = kind == :ax ? [(fh, orient[(c, fh)]) for fh in axial_faces(patch, c)] : trans_of[c]
                    out_c(c) = sum(o * s[h] > 0 ? x[h] : 0 for (h, o) in faces_of(c); init = 0)
                    in_c(c) = sum(o * s[h] < 0 ? x[h] : 0 for (h, o) in faces_of(c); init = 0)
                    E = kind == :ax ? (2 - varpi(:cm) - varpi(:cp)) / 2 * x[:f] : 0
                    @test iszero_sym(O - (sum(varpi(c) / 2 * out_c(c) for c in (:cm, :cp)) + E))
                    @test iszero_sym(I - (sum(varpi(c) / 2 * in_c(c) for c in (:cm, :cp)) + E))
                end
            end
        end
    end
    @testset "S4 the own-value coefficient of a draining unknown at a hanging segment" begin
        for seed in 1:8
            patch = build_patch(MersenneTwister(100 + seed); constrained = true)
            (; orient) = patch
            events = events_at_f(patch)
            @variables xf
            varpi(c) = frac_of(patch, c, orient[(c, :f)], :f)
            # only the own flux is non-zero, flowing c^- -> c^+ (s = +1): sigma * sign * s > 0 means outflow
            O = 0
            I = 0
            for (k, sg, flux, magnitude, sign) in events
                k == :ax && flux == :f || continue
                (sg * sign > 0 ? (O += magnitude * xf) : (I += magnitude * xf))
            end
            Vold = varpi(:cm) * xf / 2                      # all phase of c^- (= x_f) inside its strip, c^+ empty
            Vnew = Vold - O + I
            @test iszero_sym(Vnew - varpi(:cp) * xf / 2)    # the received phase sits in the strip of c^+
            # gamma = (Vold - O)/Vnew = -(2 - varpi^- - varpi^+)/varpi^+  <=>  (Vold - O) varpi^+ + (2 - varpi^- - varpi^+) Vnew = 0
            @test iszero_sym((Vold - O) * varpi(:cp) + (2 - varpi(:cm) - varpi(:cp)) * Vnew)
        end
        # the two hanging configurations of a 2:1 mesh
        for (vm, vp, expected) in ((1 // 2, 1, -1 // 2), (1, 1 // 2, -1), (1, 1, 0))
            @test (vm + vp - 2) / vp == expected
        end
    end
end
