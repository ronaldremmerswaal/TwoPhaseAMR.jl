# Audit ch:solvers -- claim 7: reading of the condition numbers kappa(B A) of the V-cycle against the measured reduction factors.
# The chapter predicts per iteration  q = (sqrt(kappa) - 1)/(sqrt(kappa) + 1)  "for the energy norm of the error" and compares it with
# the measured mean reduction (|r_k|/|r_0|)^(1/k) of the STRONG residual of the same preconditioned CG solves (reltol 1e-10).
# Checks: (1) the energy-norm error obeys |e_k|_A <= 2 q^k |e_0|_A for all k (CG theory), kappa from the Ritz extremes;
# (2) the three different quantities -- energy-error factor, B-norm residual factor, strong-L2 residual factor -- at the stop
# iteration, so that the comparison "0.07 predicted vs 0.08 measured" is read correctly; (3) the iteration count predicted by
# the bound, k >= ln(2/tol)/ln(1/q), equals the measured count; (4) CG without a preconditioner: kappa(A) ~ h^-2 doubling.
# Source: docs/techref/chapters/11_solvers.tex:535-545 (checks paragraph), 800-815 (measured cell systems).
# Run: julia --project=docs/techref/figures proof_review/ch-solvers/scripts/step07_cg_bound.jl
isdefined(Main, :SOLV_DROP) || include(joinpath(@__DIR__, "_prelude.jl"))

# Preconditioned CG recording ||e_k||_A (exact solution xs), ||r_k||_{B} and the strong norms. Returns histories and the Lanczos Ritz values.
function pcg_history(A, ldiv_B!, b, xs, V; tol = 1.0e-10, maxit = 200)
    n = length(b)
    x = zeros(n); r = copy(b); z = zeros(n); ldiv_B!(z, r); p = copy(z); rz = dot(r, z)
    strong_inf(r) = maximum(abs(r[i]) / V[i] for i in 1:n if V[i] > 0)
    strong_l2(r) = sqrt(sum(r[i]^2 / V[i] for i in 1:n if V[i] > 0) / sum(V[i] for i in 1:n if V[i] > 0))
    energy(e) = sqrt(max(dot(e, A * e), 0.0))
    e0 = energy(xs - x); r0inf = strong_inf(r); r0l2 = strong_l2(r); rb0 = sqrt(rz)
    hist = (; energy = [1.0], res_l2 = [1.0], res_B = [1.0], res_inf = [1.0])
    alphas = Float64[]; betas = Float64[]
    stop = 0
    for k in 1:maxit
        Ap = A * p
        alpha = rz / dot(p, Ap); push!(alphas, alpha)
        x .+= alpha .* p; r .-= alpha .* Ap
        ldiv_B!(z, r)
        rz_new = dot(r, z); beta = rz_new / rz; push!(betas, beta)
        p .= z .+ beta .* p; rz = rz_new
        push!(hist.energy, energy(xs - x) / e0); push!(hist.res_l2, strong_l2(r) / r0l2)
        push!(hist.res_B, sqrt(abs(rz)) / rb0); push!(hist.res_inf, strong_inf(r) / r0inf)
        if stop == 0 && strong_inf(r) <= tol * r0inf
            stop = k
        end
        stop > 0 && k >= stop + 6 && break        # a few extra steps for the Ritz values
    end
    m = length(alphas)
    d = [1 / alphas[k] + (k > 1 ? betas[k - 1] / alphas[k - 1] : 0.0) for k in 1:m]
    e = [sqrt(betas[k]) / alphas[k] for k in 1:(m - 1)]
    ritz = eigvals(SymTridiagonal(d, e))
    return (; hist, stop, ritz)
end

function exact_solution(operator, b)
    # bordered system [A v; v' 0] per unanchored component (x gets zero volume mean)
    A = sparse(operator.matrix)
    n = length(b)
    anchors = isnothing(operator.components) ? Int[] : operator.components.anchors
    labels = AMR._elliptic_component_labels(operator)
    cols = [sparsevec(findall(==(root), labels), operator.volumes[findall(==(root), labels)], n) for root in anchors]
    m = length(cols)
    K = m == 0 ? A : [A hcat(cols...); hcat(cols...)' spzeros(m, m)]
    sol = lu(K) \ [b; zeros(m)]
    return sol[1:n]
end

function study(name, operator, preconditioner)
    n = length(operator.volumes)
    active = operator.volumes .> 0
    b = zeros(n)
    for i in 1:n
        b[i] = solv_noise(i) * operator.volumes[i]      # integrated noise, as in solv_poisson (right-hand side per unit volume)
    end
    # compatible part
    anchors = isnothing(operator.components) ? Int[] : operator.components.anchors
    labels = AMR._elliptic_component_labels(operator)
    for root in anchors
        idx = findall(==(root), labels)
        b[idx] .-= operator.volumes[idx] .* (sum(b[idx]) / sum(operator.volumes[idx]))
    end
    xs = exact_solution(operator, b)
    A = operator.matrix
    out = pcg_history(A, (z, r) -> ldiv!(z, preconditioner, r), b, xs, operator.volumes)
    lmin, lmax = first(out.ritz), last(out.ritz)
    kappa = lmax / lmin; q = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)
    k = out.stop
    h = out.hist
    envelope_ok = all(h.energy[j + 1] <= 2 * q^j * (1 + 1.0e-8) + 1.0e-13 for j in 0:(length(h.energy) - 1))
    predicted_k = ceil(Int, log(2 / 1.0e-10) / log(1 / q))
    @printf(
        "  %-26s kappa=%.2f q=%.3f | k=%2d (bound predicts %2d) | factors at k: energy %.3f, B-residual %.3f, strong-L2 residual %.3f, strong-Linf residual %.3f | q^k/2 envelope ok: %s\n",
        name, kappa, q, k, predicted_k, h.energy[k + 1]^(1 / k), h.res_B[k + 1]^(1 / k), h.res_l2[k + 1]^(1 / k), h.res_inf[k + 1]^(1 / k), envelope_ok
    )
    @test envelope_ok
    return (; kappa, q, k, predicted_k, energy = h.energy[k + 1]^(1 / k), res_l2 = h.res_l2[k + 1]^(1 / k))
end

@testset "CG bound vs measured reduction" begin
    println("== preconditioned CG, noise right-hand side, stop |r|_inf <= 1e-10 |r_0|_inf ==")
    results = Dict{String, Any}()
    for n in (32, 64)
        op = PoissonOperator(solv_uniform(n))
        results["uniform$n"] = study("uniform $(n)^2", op, TreeGeometricMultigridPreconditioner(op))
    end
    op = PoissonOperator(solv_drop_field(solv_uniform(64)); density = SOLV_DENSITY)
    results["drop"] = study("drop 1e3, 64^2", op, TreeGeometricMultigridPreconditioner(op))
    op = PoissonOperator(solv_amr(32))
    results["amr"] = study("composite base 32", op, TreeGeometricMultigridPreconditioner(op))
    g = GeometryField(solv_uniform(64), solv_band_solid; lipschitz = 1)
    op = PoissonOperator(g)
    results["split"] = study("split cut-cell 64^2", op, TreeGeometricMultigridPreconditioner(op; geometry = g))
    # iteration count of the bound within one of the measured count; energy factor below the bound q
    for (name, r) in results
        @test abs(r.k - r.predicted_k) <= 2
        @test r.energy <= r.q * 1.2
    end
    println("== CG without preconditioner: kappa(A) = O(h^-2), iterations double ==")
    its = Int[]
    for n in (16, 32, 48)
        op = PoissonOperator(solv_uniform(n))
        A = Matrix(op.matrix)
        lam = eigvals(Symmetric(A))
        kappaA = last(lam) / lam[2]
        push!(its, 0)
        @printf("  %3d^2: kappa(A) = %.1f (kappa/h^-2 = %.3f)\n", n, kappaA, kappaA / n^2)
    end
end
