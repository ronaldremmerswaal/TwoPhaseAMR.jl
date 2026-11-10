# Code check 03: compatibility, removed mean and gauge of the one-velocity projection (eq:projection:one-gauge) and the sentence on
# disconnected open domains.
#
# Label:  eq:projection:one-gauge
# Source: docs/techref/chapters/10_projection.tex:66-96; src/solvers/multigrid/elliptic_solver.jl (solve!), cycles.jl
# Checks:
#   G1  after project!, Σ_c |c^o| p_c = 0 (round-off relative to Σ|c^o||p_c|), on all four fixtures;
#   G2  solve! on a right-hand side with nonzero mean: `removed_rhs_mean` = Σ_c b_c / Σ_c |c^o| with b_c = |c^o| rhs_c (integrated), and the
#       solution solves S p = b - b̄ |c^o|;
#   G3  compatibility = :error throws iff |Σ_c b_c| > 256 eps max(1, Σ_c |b_c|) (bracketing the threshold by factors 0.5 and 2);
#   G4  Σ_c b_c = -(1/dt) Σ_{f in bnd} s_f |f| a_f v*_f (discrete divergence theorem) for random boundary values, and it is zero (b̄ ~ 1e-16)
#       for Inflow + Outflow sides;
#   G5  disconnected open domain (band solid): S has exactly 2 null vectors (independent eigenvalue count = number of components by BFS);
#       separate volume means are fixed: two projections with different initial component constants yield the same pressure and velocity;
#   G6  the default tree-geometric preconditioner converges on the split domain.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-one-system/scripts/code_03_gauge.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit

rng = MersenneTwister(20261113)
dt = PA.DT

@testset "gauge, compatibility, disconnected domains" begin
    # G1
    for (name, field) in PA.fixtures()
        op = FA.one_velocity_operator(field)
        v = PA.one_predictor(field, rng)
        p, stats, _ = PA.run_projection!(field, op, v)
        vol = PA.open_volumes(field)
        pv = FA.cell_vector(ref -> AMR._get(p, ref), field.mesh)
        rel = abs(dot(vol, pv)) / dot(vol, abs.(pv))
        println(rpad(name, 18), " G1: |Σ |c^o| p| / Σ |c^o||p| = ", rel)
        @test rel <= 1.0e-14
    end
    # G2, G3 on the uniform mesh without solid with a prescribed strong right-hand side
    field = PA.fixture(:uniform; solid = false)
    mesh = field.mesh
    op = FA.one_velocity_operator(field)
    solver = EllipticSolver(op; preconditioner = :tree_geometric)
    vol = PA.open_volumes(field)
    n = length(vol)
    strong = randn(rng, n); strong .-= dot(vol, strong) / sum(vol)       # zero mean
    shift = 0.37
    rhs = FA.cell_field_of(strong .+ shift, mesh)
    sol = CellField(mesh)
    stats = solve!(sol, solver, rhs; rtol = 0.0, atol = 1.0e-9, max_cycles = 500)
    @test abs(stats.removed_rhs_mean - shift) <= 1.0e-14
    bint = vol .* (strong .+ shift)
    @test abs(stats.removed_rhs_mean - sum(bint) / sum(vol)) <= 1.0e-14
    J = PA.jump_dense(field); K = PA.metric_dense(op); γ = op.face_coefficient
    S = J' * (sqrt.(γ) .* K .* sqrt.(γ)') * J
    sv = FA.cell_vector(ref -> AMR._get(sol, ref), mesh)
    resid = (bint .- stats.removed_rhs_mean .* vol) .- S * sv
    @test maximum(abs.(resid ./ vol)) <= 2.0e-9
    # G3
    scale = sum(abs, bint)
    thr = 256 * eps() * max(1.0, scale)
    for (factor, expect_throw) in ((0.5, false), (2.0, true))
        b0 = vol .* strong
        excess = factor * thr
        bint2 = b0 .+ excess .* vol ./ sum(vol)                      # integrated sum = excess exactly
        sc = sum(abs, bint2); thr2 = 256 * eps() * max(1.0, sc)
        # scale so that excess = factor * thr2 (thr2 ~ thr for this size)
        bint2 = b0 .+ (factor * thr2) .* vol ./ sum(vol)
        rhs2 = FA.cell_field_of(bint2 ./ vol, mesh)
        thrown = try
            solve!(CellField(mesh), solver, rhs2; rtol = 0.0, atol = 1.0e-6, max_cycles = 500, compatibility = :error); false
        catch e
            e isa ArgumentError || rethrow()
            true
        end
        println("G3 factor ", factor, "  |Σ b| / thr = ", abs(sum(bint2)) / thr2, "  threw=", thrown)
        @test thrown == expect_throw
    end
    # G4: discrete divergence theorem with random boundary values
    v = PA.one_predictor(field, rng)
    v.boundary .= randn(rng, length(v.boundary))
    flux = PA.open_flux(field, v)
    bflux = sum(b.side * b.area * b.aperture * v.boundary[b.index] for b in PA.boundary_faces(field))
    @test abs(sum(flux) - bflux) <= 1.0e-12 * max(1.0, abs(bflux))
    # Inflow + Outflow sides: zero total up to rounding (b̄ is the removed mean)
    chan = BlockTreeMesh(;
        base_mesh = (16, 8), block_size = (4, 4), extent = (2.0, 1.0), periodic = (false, false), max_level = 1,
        refine = (key, o, e) -> key.level == 0 && key.i == 3
    )
    prof(y) = 4y * (1 - y)
    bc = VelocityBoundaryConditions(; left = Inflow((x, y, t, axis) -> axis == 1 ? prof(y) : 0.0), right = Outflow(), bottom = NoSlip(), top = NoSlip())
    vel = FaceVelocity(chan, (x, y, t, axis) -> axis == 1 ? prof(y) + 0.3sinpi(2x) * sinpi(y) : 0.2sinpi(x) * sinpi(2y))
    cop = PoissonOperator(chan)
    csolver = EllipticSolver(cop; preconditioner = :tree_geometric)
    cp = CellField(chan)
    cstats = project!(vel, cp, csolver, dt, ProjectionWorkspace(csolver); velocity_boundary = bc, tolerance = 1.0e-12, max_cycles = 500)
    println("G4 Inflow/Outflow channel: removed_rhs_mean = ", cstats.removed_rhs_mean)
    @test abs(cstats.removed_rhs_mean) <= 1.0e-13
    # G5: band solid, :ilu
    band = PA.band_fixture(:uniform)
    bop = FA.one_velocity_operator(band)
    bJ = PA.jump_dense(band); bK = PA.metric_dense(bop); bγ = bop.face_coefficient
    bS = bJ' * (sqrt.(bγ) .* bK .* sqrt.(bγ)') * bJ
    bvol = PA.open_volumes(band); oc = findall(>(0), bvol)
    ev = eigvals(Symmetric(bS[oc, oc]))
    nnull = count(<=(1.0e-10 * maximum(ev)), ev)
    # components by BFS on open cells joined by faces with a > 0
    a = PA.aperture(band); nx, ny = band.mesh.block_size
    adj = Dict(c => Int[] for c in oc)
    for (i, face) in pairs(band.mesh.faces)
        a[i] > 0 || continue
        m = AMR._cell_index(face.minus, nx, ny); q = AMR._cell_index(face.plus, nx, ny)
        (bvol[m] > 0 && bvol[q] > 0) || continue
        push!(adj[m], q); push!(adj[q], m)
    end
    comp = Dict{Int, Int}(); ncomp = 0
    for c in oc
        haskey(comp, c) && continue
        ncomp += 1; stack = [c]; comp[c] = ncomp
        while !isempty(stack)
            x = pop!(stack)
            for y in adj[x]
                haskey(comp, y) || (comp[y] = ncomp; push!(stack, y))
            end
        end
    end
    println("G5 band: components by BFS = ", ncomp, ", null vectors of S = ", nnull)
    @test ncomp == 2 && nnull == 2
    runs = []
    for shiftc in ((0.0, 0.0), (3.0e4, -1.0e4))
        v = PA.one_predictor(band, MersenneTwister(5))
        p0 = FA.cell_field_of([get(comp, c, 0) == 1 ? shiftc[1] : get(comp, c, 0) == 2 ? shiftc[2] : 0.0 for c in 1:length(bvol)], band.mesh)
        p, st, _ = PA.run_projection!(band, bop, v; pressure = p0, preconditioner = :ilu, tolerance = 1.0e-12)
        @test st.converged
        push!(runs, (v, FA.cell_vector(ref -> AMR._get(p, ref), band.mesh), st))
    end
    dv = maximum(abs.(runs[1][1].interior .- runs[2][1].interior)) / maximum(abs.(runs[1][1].interior))
    dp = runs[1][2] .- runs[2][2]
    per = [dp[[c for c in oc if comp[c] == k]] for k in 1:2]
    spread = [maximum(abs.(x .- sum(x) / length(x))) for x in per]
    means = [sum(x) / length(x) for x in per]
    globalmean = dot(bvol[oc], dp[oc]) / sum(bvol[oc])
    println(
        "G5 velocity difference (relative) = ", dv, "; pressure difference per component: constants ", means, " spread within component ", spread, "; global mean ", globalmean,
        "; global volume means of the two pressures ", [dot(bvol, r[2]) / sum(bvol) for r in runs]
    )
    @test dv <= 1.0e-8
    @test all(spread .<= 1.0e-6 * maximum(abs, runs[1][2]))
    @test all(abs.(means) .<= 1.0e-6 * max(1.0, maximum(abs, runs[1][2])))
    @test abs(globalmean) <= 1.0e-8 * maximum(abs, runs[1][2])
    # G6: the default preconditioner handles both components
    v = PA.one_predictor(band, MersenneTwister(5))
    _, st, _ = PA.run_projection!(band, bop, v; preconditioner = :tree_geometric)
    println("G6 tree_geometric on the band solid: converged=", st.converged)
    @test st.converged
end
