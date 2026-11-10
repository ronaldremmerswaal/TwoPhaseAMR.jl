# Code check 04: the orthogonality / kinetic-energy statements of the one-velocity projection (eq:projection:one-energy).
#
# Label:  eq:projection:one-energy
# Source: docs/techref/chapters/10_projection.tex:123-156
# Checks:
#   E1  v' M_γ R J p = v' A J p for random v and p, M_γ = A γ^{-1/2} (K^o_{++})^{-1} γ^{-1/2} A on the faces γ > 0 (dense, independent K^o from the code);
#   E2  the Pythagoras identity |v*|²_M = |v^{n+1}|²_M + dt² |R J p|²_M after project! (4 fixtures, density ratio 1000);
#   E3  where K^o is diagonal: M_γ = diag(ρ̄_f V_f) (no solid), diag(ρ̄_f V^o_f) (one-velocity, solid, γ = a²/ρ̄) and diag(a_f ρ̄_f V^o_f) for the
#       two-velocity coefficient γ = a/ρ̄ (uniform mesh; composite mesh: only off the transition);
#   E4  nonzero boundary values (Inflow/Outflow channel on a composite mesh): |v*|²_M = |v^{n+1}|²_M + dt²|RJp|²_M + 2 dt B,
#       B = Σ_{f in bnd} s_f |f| a_f v*_f p_{c(f)}  (the printed sentence "the left-hand side gains" is tested with both signs).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-one-system/scripts/code_04_energy.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit

"""Dense M_γ on the faces with γ > 0 (zero elsewhere)."""
function kinetic_metric(field, op, γ)
    K = PA.metric_dense(op)
    P = findall(>(0), γ)
    a = PA.aperture(field)
    Kpp = K[P, P]
    M = zeros(length(γ), length(γ))
    M[P, P] = (a[P] ./ sqrt.(γ[P])) .* inv(Symmetric(Kpp)) .* (a[P] ./ sqrt.(γ[P]))'
    return M, P
end

rng = MersenneTwister(20261114)
dt = PA.DT
@testset "one-velocity energy identity" begin
    for (name, field) in PA.fixtures()
        mesh = field.mesh
        op = FA.one_velocity_operator(field)
        γ = op.face_coefficient
        M, P = kinetic_metric(field, op, γ)
        R = PA.force_map_dense(field, op, γ)
        J = PA.jump_dense(field)
        A = Diagonal(PA.aperture(field))
        # E1
        v = randn(rng, length(γ)); p = randn(rng, size(J, 2))
        e1 = abs(v' * M * R * (J * p) - v' * A * (J * p)) / abs(v' * A * (J * p))
        # E2
        v0 = PA.one_predictor(field, rng); vs = copy(v0.interior)
        pr, st, _ = PA.run_projection!(field, op, v0; tolerance = 1.0e-12)
        v1 = v0.interior
        Δ = v1 .- vs
        nstar = vs' * M * vs; nend = v1' * M * v1; ndel = Δ' * M * Δ
        e2 = abs(nstar - nend - ndel) / nstar
        # also: the pressure correction is dt R J p
        pv = FA.cell_vector(ref -> AMR._get(pr, ref), mesh)
        e2b = maximum(abs.(Δ .+ dt .* (R * (J * pv)))) / maximum(abs.(Δ))
        println(rpad(name, 18), " E1=", round(e1, sigdigits = 3), " E2=", round(e2, sigdigits = 3), " removed=", round(100 * ndel / nstar, digits = 2), " %  (Δ vs -dt R J p: ", round(e2b, sigdigits = 3), ")")
        @test e1 <= 1.0e-12
        @test e2 <= 1.0e-8
        @test e2b <= 1.0e-12
        # E3 diagonal statements
        d = PA.face_data(field)
        Vo = PA.open_dual_volumes(field)
        a = PA.aperture(field)
        offdiag(Mx) = Mx - Diagonal(diag(Mx))
        faces_ok = [f for f in P if all(abs.(PA.metric_dense(op)[f, :] .- (1:length(γ) .== f) .* PA.metric_dense(op)[f, f]) .<= 1.0e-14 * abs(PA.metric_dense(op)[f, f]))]   # rows of K^o with a single entry
        expected = isnothing(field.geometry) ? d.m[faces_ok] .* Vo[faces_ok] : d.m[faces_ok] .* Vo[faces_ok]
        @test maximum(abs.(diag(M)[faces_ok] .- expected) ./ expected) <= 1.0e-12
        println("   E3 one-velocity: ", length(faces_ok), " of ", length(P), " faces have a single-entry row of K^o; diag(M) = ρ̄ V", isnothing(field.geometry) ? "" : "^o", " there")
        if !isnothing(field.geometry)
            γtv = a ./ d.m
            Mt, Pt = kinetic_metric(field, op, γtv)       # same K^o, two-velocity coefficient
            ok = [f for f in Pt if all(abs.(PA.metric_dense(op)[f, :] .- (1:length(γ) .== f) .* PA.metric_dense(op)[f, f]) .<= 1.0e-14 * abs(PA.metric_dense(op)[f, f]))]
            @test maximum(abs.(diag(Mt)[ok] .- (a .* d.m .* Vo)[ok]) ./ (a .* d.m .* Vo)[ok]) <= 1.0e-12
            println("   E3 two-velocity: diag(M_tv) = a ρ̄ V^o on ", length(ok), " faces")
        end
    end
    # E4: Inflow / Outflow channel on a composite mesh
    chan = BlockTreeMesh(;
        base_mesh = (16, 8), block_size = (4, 4), extent = (2.0, 1.0), periodic = (false, false), max_level = 1,
        refine = (key, o, e) -> key.level == 0 && key.i == 3
    )
    prof(y) = 4y * (1 - y)
    bc = VelocityBoundaryConditions(; left = Inflow((x, y, t, axis) -> axis == 1 ? prof(y) : 0.0), right = Outflow(), bottom = NoSlip(), top = NoSlip())
    vel = FaceVelocity(chan, (x, y, t, axis) -> axis == 1 ? prof(y) + 0.3sinpi(2x) * sinpi(y) : 0.2sinpi(x) * sinpi(2y))
    enforce_velocity_boundary!(vel, bc; time = 0.0)
    vstar = copy(vel.interior); bstar = copy(vel.boundary)
    cop = PoissonOperator(chan); csolver = EllipticSolver(cop; preconditioner = :tree_geometric)
    cp = CellField(chan)
    st = project!(vel, cp, csolver, dt, ProjectionWorkspace(csolver); velocity_boundary = bc, tolerance = 1.0e-13, max_cycles = 500)
    cfield = (; mesh = chan, geometry = nothing)
    cJ = Matrix(FA.jump_matrix(chan)); cK = Matrix(FA.metric_matrix(cop)); cγ = cop.face_coefficient
    cM = inv(Symmetric(cK))                                       # γ = 1: M = K^{-1}
    @test all(cγ .== 1.0)
    pv = FA.cell_vector(ref -> AMR._get(cp, ref), chan)
    Gp = cK * (cJ * pv)                                            # R J p with γ = 1, a = 1
    vpre = vstar .- dt .* Gp                                       # v^{n+1} before the boundary reset
    nstar = vstar' * cM * vstar; nend = vpre' * cM * vpre; ndel = dt^2 * Gp' * cM * Gp
    nx, ny = chan.block_size
    B = sum(face.side * face.area * bstar[i] * pv[AMR._cell_index(face.cell, nx, ny)] for (i, face) in pairs(chan.boundary_faces))
    plus = (nstar - nend - ndel - 2dt * B) / nstar       # tests  |v*|² = |v|² + dt²|G|² + 2 dt B
    minus = (nstar - nend - ndel + 2dt * B) / nstar      # tests  |v*|² = |v|² + dt²|G|² - 2 dt B
    println("E4 channel: 2 dt B / |v*|² = ", 2dt * B / nstar, "; residual with +2dtB on the right: ", plus, "; with -2dtB: ", minus)
    @test abs(plus) <= 1.0e-8
    @test abs(minus) > 100 * abs(plus)
end
