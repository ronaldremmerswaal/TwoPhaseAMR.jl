# Code check 02: the energy (Pythagoras) identity of the mimetic projection (eq:projection:mimetic-energy), its conditions, and the code statistic.
#
# Label:  eq:projection:mimetic-energy
# Source: docs/techref/chapters/10_projection.tex:553-583
# Checks:
#   E1  E(u*) = E(u^{n+1}) + E(u^{n+1} - u*) with E(u) = |ū|²_M + Σ_{faces with both phases} m^r V^o [u]² (independent dense evaluation, 4 fixtures x 2 variants);
#       the code statistic `energy_identity_error` = 2|<u^{n+1}, u^{n+1}-u*>_E| (independent value);
#   E2  the conditions: zero boundary flux per component (walled box: yes) and apgfm[u] = 0 on the faces outside F_mix that carry both phases:
#       with the default floor sqrt(eps) the leftover is 1e-13-ish; with a coarse floor 0.1 the identity FAILS: E(u*) - E(u1) - E(Δ) =
#       -2 dt Σ_{f carrying both phases, f not in F_mix} apgfm_f [u^{n+1}]_f (J p)_f   (derived from the proof; verified to 1e-9 relative) -- the condition is sharp;
#   E3  the CMDGFM does NOT satisfy the identity (text: "no energy identity is claimed").
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-kkt/scripts/code_02_mimetic_energy.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit

"""Common velocity ū (zero where a phase is absent only through the zero-weight rule) and jump on faces with both phases (NaN-safe)."""
function common_and_jump(d, v)
    n = length(d.m)
    pr(w, x) = (iszero(w) || isnan(x)) ? 0.0 : w * x           # absent (NaN) velocities carry no weight; covered faces have both NaN
    ū = [(pr(d.ml[f], v[1].interior[f]) + pr(d.mg[f], v[2].interior[f])) / d.m[f] for f in 1:n]
    both = [d.ml[f] > 0 && d.mg[f] > 0 && !isnan(v[1].interior[f]) && !isnan(v[2].interior[f]) for f in 1:n]
    jmp = [both[f] ? v[2].interior[f] - v[1].interior[f] : 0.0 for f in 1:n]
    return ū, jmp, both
end

function energy_pieces(field, d, op, γ, M, v, v2 = nothing)
    ū, jmp, both = common_and_jump(d, v)
    Vo = PA.open_dual_volumes(field)
    if v2 === nothing
        return ū' * M * ū + sum(d.mr[f] * Vo[f] * jmp[f]^2 for f in findall(both)), ū, jmp
    end
    ū2, jmp2, _ = common_and_jump(d, v2)
    return ū' * M * ū2 + sum(d.mr[f] * Vo[f] * jmp[f] * jmp2[f] for f in findall(both)), ū, jmp
end

function kinetic_metric(field, op, γ)
    K = PA.metric_dense(op); P = findall(>(0), γ); a = PA.aperture(field)
    M = zeros(length(γ), length(γ))
    M[P, P] = (a[P] ./ sqrt.(γ[P])) .* inv(Symmetric(K[P, P])) .* (a[P] ./ sqrt.(γ[P]))'
    return M
end

function run_case(field, variant; seed = 3)
    mesh = field.mesh; dt = PA.DT
    solver = MimeticProjectionSolver(field; density = PA.DENSITY, variant)
    work = MimeticProjectionWorkspace(solver)
    d = PA.face_data(field)
    a = PA.aperture(field)
    γ = isnothing(field.geometry) ? 1 ./ d.m : a ./ d.m
    v0 = PA.two_predictor(field, MersenneTwister(seed)); vstar = PA.copy_velocity(v0)
    velocity = PA.copy_velocity(v0)
    pressure = CellField(mesh)
    stats = project!(velocity, pressure, solver, dt, work; tolerance = 1.0e-10, max_iterations = 20000)
    @assert stats.converged
    op = work.cache.pressure.operator
    M = kinetic_metric(field, op, γ)
    Estar = energy_pieces(field, d, op, γ, M, vstar)[1]
    E1 = energy_pieces(field, d, op, γ, M, velocity)[1]
    Δ = (copy_with(velocity[1], vstar[1]), copy_with(velocity[2], vstar[2]))
    Edelta = energy_pieces(field, d, op, γ, M, Δ)[1]
    cross = energy_pieces(field, d, op, γ, M, velocity, Δ)[1]            # <u^{n+1}, u^{n+1} - u*>
    return (; d, d_stats = stats, Estar, E1, Edelta, cross, velocity, vstar, pressure, work, op, γ, M)
end
copy_with(a::FaceVelocity, b::FaceVelocity) = (w = FaceVelocity(a.mesh); w.interior .= a.interior .- b.interior; w.boundary .= a.boundary .- b.boundary; w)

@testset "mimetic energy identity" begin
    for (name, field) in PA.fixtures(), variant in (:joint_smoothed, :joint_face)
        r = run_case(field, variant)
        res = abs(r.Estar - r.E1 - r.Edelta) / r.Estar
        codeerr = r.d_stats.energy_identity_error
        println(
            rpad(name, 18), rpad(variant, 15), " E1: |E(u*)-E(u1)-E(Δ)|/E(u*) = ", round(res, sigdigits = 3), "  removed ", round(100 * r.Edelta / r.Estar, digits = 2), " %;  2|<u1,Δ>|/E* = ",
            round(2abs(r.cross) / r.Estar, sigdigits = 3), " code statistic/E* = ", round(codeerr / r.Estar, sigdigits = 3)
        )
        @test res <= 1.0e-9
        @test abs(2abs(r.cross) - codeerr) <= 1.0e-6 * r.Estar + 1.0e-12
        @test abs((r.Estar - r.E1 - r.Edelta) + 2r.cross) <= 1.0e-9 * r.Estar     # algebra: E(u*) = E(u1) - 2<u1,Δ> + E(Δ)
    end
    # E2: coarse fraction floor breaks the identity by exactly the predicted amount
    field = PA.fixture(:uniform; solid = false, fraction_floor = 0.1)
    r = run_case(field, :joint_smoothed)
    d = r.d; mesh = field.mesh
    J = PA.jump_dense(field)
    pv = FA.cell_vector(ref -> AMR._get(r.pressure, ref), mesh)
    ū1, jmp1, both = common_and_jump(d, r.velocity)
    notF = [f for f in 1:length(d.m) if both[f] && !(f in d.mixed)]
    predicted = -2PA.DT * sum(d.apgfm[f] * jmp1[f] * (J * pv)[f] for f in notF)
    actual = (r.Estar - r.E1 - r.Edelta)
    println("E2 floor 0.1: faces carrying both phases outside F_mix: ", length(notF), "; E(u*)-E(u1)-E(Δ) = ", actual, " (", round(abs(actual) / r.Estar, sigdigits = 3), " of E*); predicted -2dtΣ apgfm[u](Jp) = ", predicted)
    @test abs(actual) > 1.0e-6 * r.Estar
    @test abs(actual - predicted) <= 1.0e-6 * abs(predicted) + 1.0e-9 * r.Estar
    # E3: CMDGFM
    field = PA.fixture(:uniform; solid = false)
    solver = CMDGFMProjectionSolver(field; density = PA.DENSITY); work = CMDGFMProjectionWorkspace(solver)
    d = PA.face_data(field); γ = 1 ./ d.m
    op = AMR._projection_pressure_operator(solver)
    M = kinetic_metric(field, op, γ)
    v0 = PA.two_predictor(field, MersenneTwister(3)); vstar = PA.copy_velocity(v0); velocity = PA.copy_velocity(v0)
    st = project!(velocity, CellField(field.mesh), solver, PA.DT, work; tolerance = 1.0e-10, max_iterations = 5000)
    Estar = energy_pieces(field, d, op, γ, M, vstar)[1]; E1 = energy_pieces(field, d, op, γ, M, velocity)[1]
    Edelta = energy_pieces(field, d, op, γ, M, (copy_with(velocity[1], vstar[1]), copy_with(velocity[2], vstar[2])))[1]
    println("E3 CMDGFM: |E(u*)-E(u1)-E(Δ)|/E* = ", abs(Estar - E1 - Edelta) / Estar)
    @test abs(Estar - E1 - Edelta) / Estar > 1.0e-6
end
