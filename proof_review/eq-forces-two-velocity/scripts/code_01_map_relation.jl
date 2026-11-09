# Code check 01: the one-velocity force map R^1 (coefficient a^2/rho) against the two-velocity pressure map R^tv (coefficient a/rho):
# the algebraic condition for the two-velocity projections to return a hydrostatic state, eq:forces:two-velocity.
#
# Label:  eq:forces:two-velocity  (eq:forces:coefficient, issue #76)
# Source: docs/techref/chapters/08_forces.tex:520-574; src/two_phase/projection/common.jl (_projection_pressure_operator,
#         _projection_gradient_core!), src/two_phase/two_velocity/stepping.jl (_phase_force_substep!)
# With M = W K W (cut-cell metric) and rho = staggered density:  R^1 = rho^{-1/2} M diag(a) rho^{-1/2},
#                                                                R^tv = (a rho)^{-1/2} M a^{1/2} rho^{-1/2}   (rows with a = 0 vanish).
# Checks (hydrostatic planar interface, vertical g, densities (1000, 1); J = the gravity jumps of the code; p* = the cell function with
# J = D p* on faces with a > 0):
#   M1  no solid: the two coefficients, hence the two maps, coincide (bitwise);
#   M2  with a solid and a diagonal row of K:  (R^1 J)_f = a_f (R^tv J)_f  exactly; at a coupled row (level transition) the ratio is not a_f;
#   M3  E = R^1 J - R^tv(D p*) (the part of the gravity increment that the common pressure gradient of the two-velocity projection
#       cannot remove): zero when a_f = 1 on every face carrying a jump and on every face coupled to such a face by K; nonzero otherwise;
#   F   (after the fix of #76) the operator the code uses for the phase forces equals the two-velocity pressure operator;
#   M4  the necessary and sufficient condition (see audit): E = 0.  Tabulated for the six fixtures.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-two-velocity/scripts/code_01_map_relation.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

g = (0.0, -1.0)
table = []
for kind in (:uniform, :amr), solid in (:none, :gas, :liquid)
    field = FA.plane_field(kind; gravity = g, solid)
    mesh = field.mesh
    op1 = FA.one_velocity_operator(field); optv = FA.two_velocity_operator(field)
    gravity!(FaceVelocity(mesh), field, op1, EllipticWorkspace(op1); density = FA.DENSITY, gravity = g)
    J = copy(op1.gradient_operator.pressure_jump)
    a = FA.apertures(field)
    isolid = !isnothing(field.geometry)
    rho = AMR.mixture_density(field, FA.DENSITY)
    D = FA.jump_matrix(mesh)
    support = findall(>(0), a)
    pstar = Matrix(D[support, :]) \ J[support]
    K1 = FA.metric_matrix(op1); Ktv = FA.metric_matrix(optv)
    gamma1 = isolid ? a .^ 2 ./ rho : 1 ./ rho
    gammatv = isolid ? a ./ rho : 1 ./ rho
    # M1
    coefficient_equal = !isolid ? (op1.face_coefficient == optv.face_coefficient) : nothing
    R1J = FA.force_map(K1, gamma1, a, J, isolid)
    RtvJ = FA.force_map(Ktv, gammatv, a, J, isolid)
    # M2: ratio at diagonal rows
    rowcount = [length(nzrange(K1, i)) for i in 1:size(K1, 1)]
    diagrows = [i for i in findall(==(1), rowcount) if a[i] > 0 && abs(RtvJ[i]) > 0]
    ratio_dev = isempty(diagrows) ? 0.0 : maximum(abs(R1J[i] - a[i] * RtvJ[i]) / abs(R1J[i]) for i in diagrows)
    coupled = [i for i in findall(>(1), rowcount) if a[i] > 0 && abs(RtvJ[i]) > 0 && a[i] < 1]
    ratio_dev_coupled = isempty(coupled) ? NaN : maximum(abs(R1J[i] - a[i] * RtvJ[i]) / abs(R1J[i]) for i in coupled)
    # M3/M4: E and the sufficient condition
    E = R1J .- FA.force_map(Ktv, gammatv, a, D * pstar, isolid)
    scale = maximum(abs, R1J)
    Enorm = maximum(abs, E) / scale
    jumpfaces = findall(!iszero, J)
    reach = Set(jumpfaces)
    for f in jumpfaces, k in nzrange(K1, f)
        push!(reach, rowvals(K1)[k])
    end
    min_a = minimum(a[i] for i in reach if a[i] > 0)       # closed faces (a = 0) have zero rows and do not matter
    push!(table, (; kind, solid, coefficient_equal, ratio_dev, ratio_dev_coupled, Enorm, min_a, nreach = length(reach)))
    println(
        rpad("$kind $solid", 14), " coefficients equal: ", coefficient_equal, "  M2 diagonal rows: |R1J - a RtvJ| = ", round(ratio_dev, sigdigits = 2),
        "  coupled cut rows: ", round(ratio_dev_coupled, sigdigits = 2), "  M3 |E|/|R1J| = ", round(Enorm, sigdigits = 3), "  min a>0 over jump faces and their K-neighbours = ", round(min_a, sigdigits = 4)
    )
end
# F: the operator the code uses for the phase forces after the fix (#76) IS the two-velocity pressure operator (same coefficient and metric)
fix_ok = true
for kind in (:uniform, :amr), solid in (:gas, :liquid)
    field = FA.plane_field(kind; gravity = g, solid)
    cs = CMDGFMProjectionSolver(field; density = FA.DENSITY)
    code_op = AMR._phase_force_pressure_operator!(cs, CMDGFMProjectionWorkspace(cs))
    ref_op = FA.two_velocity_operator(field)
    global fix_ok &= code_op.face_coefficient == ref_op.face_coefficient && FA.metric_matrix(code_op) == FA.metric_matrix(ref_op)
    ms = MimeticProjectionSolver(field; density = FA.DENSITY)
    mim_op = AMR._phase_force_pressure_operator!(ms, MimeticProjectionWorkspace(ms))
    global fix_ok &= mim_op.face_coefficient == ref_op.face_coefficient
end
println("F code's force operator (CMDGFM and mimetic) equals the two-velocity pressure operator: ", fix_ok)
@testset "R^1 versus R^tv" begin
    @test fix_ok
    for r in table
        r.solid == :none && @test r.coefficient_equal === true
        @test r.ratio_dev <= 1.0e-12
        # M3: E = 0  <=>  min a = 1 (on the reach of the jumps), in both directions over the six fixtures
        @test (r.Enorm <= 1.0e-10) == (r.min_a >= 1 - 1.0e-9)
    end
    @test any(r -> r.Enorm > 1.0e-3, table)         # the mismatch exists (composite + solid)
end
