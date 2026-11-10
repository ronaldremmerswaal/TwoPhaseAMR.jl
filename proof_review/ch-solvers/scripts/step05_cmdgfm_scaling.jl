# Audit ch:solvers -- claim 5: the congruence-free two-sided scaling of the CMDGFM system, A^ = Dr A Dc, b^ = Dr b, x = Dc x^,
# P^ = Dc^-1 P Dr^-1, gives A^ P^ = Dr (A P) Dr^-1, similar to A P ("the scaling changes the norm in which BiCGSTAB measures its
# residual, not the preconditioned spectrum").
# (a) algebra: spec(A^ P^) = spec(A P) (similarity) -- verified on random matrices; code scalings read from
#     _cmdgfm_krylov_scaling! and _ScaledProjectionPreconditioner (same Dr, Dc, P^ as the text).
# (b) what is NOT invariant: BiCGSTAB is not invariant under a non-uniform Dr. Krylov.bicgstab! on the scaled system
#     reproduces, step for step, BiCGSTAB on the unscaled right-preconditioned system with the inner product
#     <u, v> = u' Dr^2 v (shadow residual r~0 = Dr r0, and the omega-step minimising |Dr r|_2); and it differs from the
#     Euclidean BiCGSTAB on the unscaled system.
# Source: docs/techref/chapters/11_solvers.tex:765-790; code src/two_phase/projection/cmdgfm_projection.jl:99-196, 1432-1500.
# Run: julia --project=docs/techref/figures proof_review/ch-solvers/scripts/step05_cmdgfm_scaling.jl
isdefined(Main, :SOLV_DROP) || include(joinpath(@__DIR__, "_prelude.jl"))

# Reference right-preconditioned BiCGSTAB (van der Vorst 1992) in the inner product <u, v> = u' W v, x0 = 0.
# Returns the iterate after k iterations (the unpreconditioned variable x; right preconditioning: x = P y).
function bicgstab_weighted(A, P, b, W, k)
    ip(u, v) = dot(u, W * v)
    n = length(b)
    x = zeros(n); r = copy(b); rt = copy(r)
    rho_old = 1.0; alpha = 1.0; omega = 1.0
    v = zeros(n); p = zeros(n)
    for _ in 1:k
        rho = ip(rt, r)
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        phat = P * p
        v = A * phat
        alpha = rho / ip(rt, v)
        s = r - alpha * v
        shat = P * s
        t = A * shat
        omega = ip(t, s) / ip(t, t)           # minimises |s - omega t|_W
        x = x + alpha * phat + omega * shat
        r = s - omega * t
        rho_old = rho
    end
    return x
end

struct DenseP{M}
    M::M
end
LinearAlgebra.ldiv!(y, p::DenseP, x) = (y .= p.M * x; y)
Base.size(p::DenseP) = size(p.M)
Base.eltype(::DenseP) = Float64

@testset "CMDGFM scaling" begin
    println("== code: scaling constants and the scaled preconditioner ==")
    mesh = solv_uniform(16)
    field = solv_drop_field(mesh)
    # the scalings of _cmdgfm_krylov_scaling!: rows (dt/L)^2 | dt^2/L, columns rho0 (L/dt)^2 | L/dt^2
    scaling = AMR._ProjectionKrylovScaling(Float64)
    assembled = (; pressure_indices = 1:7, mixed_faces = 1:3, pressure_operator = (; mesh))
    AMR._cmdgfm_krylov_scaling!(scaling, assembled, 1000.0, 0.01)
    L, dt, rho0 = 1.0, 0.01, 1000.0
    @test scaling.rows[1] ≈ (dt / L)^2 && scaling.rows[8] ≈ dt^2 / L
    @test scaling.columns[1] ≈ rho0 * (L / dt)^2 && scaling.columns[8] ≈ L / dt^2
    # P^ = Dc^-1 P Dr^-1 as applied by _ScaledProjectionPreconditioner
    n = 10
    scaling = AMR._ProjectionKrylovScaling(Float64)
    AMR._cmdgfm_krylov_scaling!(scaling, (; pressure_indices = 1:7, mixed_faces = 1:3, pressure_operator = (; mesh)), 1000.0, 0.01)
    rng = MersenneTwister(1)
    Pm = randn(rng, n, n)
    pc = AMR._ScaledProjectionPreconditioner(DenseP(Pm), scaling)
    u = randn(rng, n); y = zeros(n); ldiv!(y, pc, u)
    @test y ≈ Diagonal(1 ./ scaling.columns) * Pm * Diagonal(1 ./ scaling.rows) * u

    println("== (a) similarity of the scaled and unscaled preconditioned operators ==")
    worst = 0.0
    for trial in 1:50
        n = 24
        A = randn(rng, n, n) + 6I; Pm = inv(A + 0.3 * randn(rng, n, n))
        Dr = Diagonal(10.0 .^ (4 .* rand(rng, n) .- 2)); Dc = Diagonal(10.0 .^ (4 .* rand(rng, n) .- 2))
        Ah = Dr * A * Dc; Ph = inv(Dc) * Pm * inv(Dr)
        ev1 = sort(eigvals(A * Pm); by = z -> (real(z), imag(z))); ev2 = sort(eigvals(Ah * Ph); by = z -> (real(z), imag(z)))
        worst = max(worst, maximum(abs, ev1 - ev2) / maximum(abs, ev1))
        @test Ah * Ph ≈ Dr * (A * Pm) * inv(Dr)
    end
    @printf("  spec(Dr A P Dr^-1) vs spec(A P): max relative difference %.1e (similarity; condition of the diagonal similarity limits it)\n", worst)
    @test worst < 1.0e-6

    println("== (b) BiCGSTAB is invariant only for the Dr^2-weighted inner product ==")
    n = 40
    A = randn(rng, n, n) + 8I
    Pm = inv(A + 0.8 * randn(rng, n, n))                  # an imperfect right preconditioner
    b = randn(rng, n)
    # block scaling like the CMDGFM one: two distinct row/column scales on a (30, 10) split
    rows = vcat(fill(1.0e-4, 30), fill(1.0e-6, 10)); cols = vcat(fill(1.0e5, 30), fill(1.0e4, 10))
    Dr = Diagonal(rows); Dc = Diagonal(cols)
    Ah = Dr * A * Dc; Ph = inv(Dc) * Pm * inv(Dr); bh = Dr * b
    for k in (1, 3, 6)
        ws = Krylov.BicgstabWorkspace(Ah, bh)
        Krylov.bicgstab!(ws, Ah, bh; N = Ph, ldiv = false, atol = 0.0, rtol = 0.0, itmax = k)
        @test Krylov.iteration_count(ws) == k
        x_scaled = Dc * Krylov.solution(ws)
        x_weighted = bicgstab_weighted(A, Pm, b, Diagonal(rows .^ 2), k)
        x_euclid = bicgstab_weighted(A, Pm, b, Diagonal(ones(n)), k)
        d_w = norm(x_scaled - x_weighted) / norm(x_weighted)
        d_e = norm(x_scaled - x_euclid) / norm(x_euclid)
        @printf("  k=%d: |x_scaled - x_(Dr^2-weighted)|/|x| = %.1e   |x_scaled - x_(Euclidean)|/|x| = %.1e\n", k, d_w, d_e)
        @test d_w < 1.0e-8
        @test d_e > 1.0e-3
    end
end
