# Symbolic check 01: the algebra of the force map -- conforming reductions (eq:forces:conforming), the relation between the one- and the
# two-velocity map, and the entry-wise form of the map.
#
# Label:  eq:forces:map, eq:forces:conforming, eq:forces:two-velocity
# Source: docs/techref/chapters/08_forces.tex:137-168 (map, conforming), 99-136 (coefficients), 520-540 (two velocities)
# Notation: s = sqrt(gamma); rho = r^2 (r > 0); a = sa^2 (sa > 0); W^2 = V/Vo (W = w); K = 1/V on a conforming patch.
#   S1  no solid:  R J = s K s J = J / (rho V);
#   S2  solid, one-velocity (gamma = a^2/rho):  R J = (s/a) W K W s J = a J / (rho Vo);
#   S3  solid, two-velocity (gamma = a/rho):    R J = J / (rho Vo);  hence  R^1 J = a R^tv J  (diagonal rows);
#   S4  general symmetric M = W K W (3 x 3, six symbols):  R^1_ij = sqrt(a_i a_j) R^tv_ij  and R^1_ij = M_ij a_j / (sqrt(rho_i) sqrt(rho_j)),
#       i.e.  R^1 = A^{1/2} R^tv A^{1/2}  (rows and columns of closed faces, a = 0, vanish in both).
# Run:    julia --project=@claude-symbolics proof_review/eq-forces-map/scripts/step01_map_symbolic.jl

using Symbolics, Test

zero_expr(e) = isequal(Symbolics.simplify(Symbolics.expand(e)), 0)

@variables V Vo J w r sa
a = sa^2
rho = r^2
# S1
K = 1 / V
s1 = (1 / r) * K * (1 / r) * J
# S2
s2 = (a / r) / a * w * K * w * (a / r) * J
# S3
s3 = (sa / r) / a * w * K * w * (sa / r) * J
Vo_expr = V / w^2          # W^2 = V / Vo
@testset "conforming reductions" begin
    @test zero_expr(s1 - J / (rho * V))
    @test zero_expr(Symbolics.substitute(s2 - a * J / (rho * Vo_expr), Dict()))
    @test zero_expr(s3 - J / (rho * Vo_expr))
    @test zero_expr(s2 - a * s3)
    @test !zero_expr(s2 - s3)                         # the checker is not vacuous: the two maps differ by the factor a
end

# S4: entry-wise form with a symmetric 3 x 3 M
@variables m11 m12 m13 m22 m23 m33 r1 r2 r3 q1 q2 q3
M = [m11 m12 m13; m12 m22 m23; m13 m23 m33]
rr = [r1, r2, r3]; qq = [q1, q2, q3]            # q_i = sqrt(a_i)
R1 = [((qq[i]^2 / rr[i]) / qq[i]^2) * M[i, j] * (qq[j]^2 / rr[j]) for i in 1:3, j in 1:3]       # (sqrt(gamma_i)/a_i) M_ij sqrt(gamma_j), gamma = a^2/rho
Rtv = [((qq[i] / rr[i]) / qq[i]^2) * M[i, j] * (qq[j] / rr[j]) for i in 1:3, j in 1:3]          # gamma = a/rho
@testset "entry-wise relation of the two maps" begin
    for i in 1:3, j in 1:3
        @test zero_expr(R1[i, j] - qq[i] * qq[j] * Rtv[i, j])
        @test zero_expr(R1[i, j] - M[i, j] * qq[j]^2 / (rr[i] * rr[j]))
    end
end
