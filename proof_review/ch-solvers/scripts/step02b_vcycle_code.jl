# Audit ch:solvers -- claim 2 (code part): the package's projected V-cycle (TreeGeometricMultigridPreconditioner, default
# 2+2 Jacobi sweeps, rediscretised coarse operators, no Galerkin product) is symmetric, has its range in the
# volume-mean-free space of every unanchored component, couples active and covered cells by exactly zero, and is
# positive definite on that space -- tested beyond the 32 x 32 / 2560-cell cases of docs/techref/figures/solvers.jl:
# 64^2 dense, 128^2 and 256^2 uniform, composite meshes (base 16 dense; base 64 with 2-3 levels), density ratios 1e3 and
# 1e6, cut-cell split domains (64^2 dense, 128^2), nearly vanishing cut cells, a solid on a composite mesh, the
# Dirichlet-inactive operator of the liquid-carrier projection, periodic meshes, anchoring by Outflow, and the
# matrix-free finest level (assemble_matrix = false).
# Dense cases: B from n V-cycles, spectrum of B A on the mean-free space exactly (as solv_vcycle_check).
# Large cases: symmetry on random vectors, Rayleigh quotients, and the extreme Ritz values of PCG (Lanczos) on the
# mean-free space.
# Source: docs/techref/chapters/11_solvers.tex:498-515, 521-545; docs/techref/figures/solvers.jl:solv_vcycle_check.
# Run: julia --project=docs/techref/figures proof_review/ch-solvers/scripts/step02b_vcycle_code.jl
isdefined(Main, :SOLV_DROP) || include(joinpath(@__DIR__, "_prelude.jl"))

function mean_free_basis(operator)
    n = length(operator.volumes)
    active = findall(>(0), operator.volumes)
    anchors = isnothing(operator.components) ? Int[] : operator.components.anchors
    labels = AMR._elliptic_component_labels(operator)
    C = zeros(length(anchors), length(active))
    for (k, root) in pairs(anchors), (i, cell) in pairs(active)
        labels[cell] == root && (C[k, i] = operator.volumes[cell])
    end
    return active, C
end

function dense_vcycle_check(operator, preconditioner, name)
    n = length(operator.volumes)
    B = vcycle_matrix(preconditioner, n)
    A = Matrix(operator.matrix)
    active, C = mean_free_basis(operator)
    Z = isempty(C) ? Matrix(1.0I, length(active), length(active)) : nullspace(C)
    Baa = B[active, active]
    L = cholesky(Symmetric(Z' * A[active, active] * Z)).L
    M = L' * (Z' * Baa * Z) * L
    lambda = eigvals(Symmetric((M + M') / 2))
    inactive = setdiff(1:n, active)
    asym = maximum(abs, B - B') / maximum(abs, B)
    range = isempty(C) ? 0.0 : maximum(abs, C * Baa) / maximum(abs, Baa)
    coupling = isempty(inactive) ? 0.0 : maximum(abs, B[inactive, active]) + maximum(abs, B[active, inactive])
    @printf(
        "  %-40s n=%5d levels=%d comps=%d  asym=%.1e range=%.1e inactive=%.0e  spec(BA)=[%.3f, %.3f] kappa=%.2f\n",
        name, n, length(preconditioner.levels), size(C, 1), asym, range, coupling, first(lambda), last(lambda), last(lambda) / first(lambda)
    )
    @test asym <= 1.0e-11
    @test range <= 1.0e-11
    @test coupling == 0
    @test first(lambda) > 0
    return B
end

# PCG with the V-cycle on a compatible random right-hand side; Ritz values from the Lanczos tridiagonal matrix.
function lanczos_extremes(operator, preconditioner; steps = 60, seed = 1)
    n = length(operator.volumes)
    A = operator.matrix
    rng = MersenneTwister(seed)
    active = operator.volumes .> 0
    b = A * (randn(rng, n) .* active)          # in the range of A (compatible)
    x = zeros(n); r = copy(b); z = zeros(n); ldiv!(z, preconditioner, r); p = copy(z)
    rz = dot(r, z)
    alphas = Float64[]; betas = Float64[]
    for k in 1:steps
        Ap = A * p
        alpha = rz / dot(p, Ap)
        push!(alphas, alpha)
        x .+= alpha .* p; r .-= alpha .* Ap
        ldiv!(z, preconditioner, r)
        rz_new = dot(r, z)
        beta = rz_new / rz
        push!(betas, beta)
        p .= z .+ beta .* p
        rz = rz_new
        sqrt(abs(rz_new) / abs(dot(b, b))) < 1.0e-12 && break      # converged: more steps only give noise
    end
    m = length(alphas)
    d = zeros(m); e = zeros(m - 1)
    for k in 1:m
        d[k] = 1 / alphas[k] + (k > 1 ? betas[k - 1] / alphas[k - 1] : 0.0)
        k < m && (e[k] = sqrt(betas[k]) / alphas[k])
    end
    ritz = eigvals(SymTridiagonal(d, e))
    return first(ritz), last(ritz), m
end

function sampled_checks(operator, preconditioner, name; draws = 8)
    n = length(operator.volumes)
    rng = MersenneTwister(5)
    active = operator.volumes .> 0
    A = operator.matrix
    worst = 0.0; quotient = Inf
    u = zeros(n); v = zeros(n); Bu = zeros(n); Bv = zeros(n)
    for _ in 1:draws
        # compatible vectors (images of A): B only sees the compatible part, so symmetry is tested on it
        u .= A * (randn(rng, n) .* active); v .= A * (smooth_field(rng, operator))
        ldiv!(Bu, preconditioner, u); ldiv!(Bv, preconditioner, v)
        worst = max(worst, abs(dot(u, Bv) - dot(v, Bu)) / (norm(u) * norm(Bv)))
        quotient = min(quotient, dot(u, Bu) / dot(u, u), dot(v, Bv) / dot(v, v))
    end
    lmin, lmax, m = lanczos_extremes(operator, preconditioner)
    @printf(
        "  %-40s n=%7d levels=%d  sym(u,v)=%.1e  min u'Bu/u'u=%.2e  Ritz(BA)=[%.3f, %.3f] (%d steps)\n",
        name, n, length(preconditioner.levels), worst, quotient, lmin, lmax, m
    )
    @test worst <= 1.0e-10
    @test quotient > 0
    @test lmin > 0
    return (; lmin, lmax)
end

# smooth random field: a few low Fourier modes in the cell index order is meaningless on a tree; use a random-weighted sum
# of cell centres' trigonometric modes instead.
function smooth_field(rng, operator)
    mesh = operator.mesh
    n = length(operator.volumes)
    nx, ny = mesh.block_size
    f = zeros(n)
    c = randn(rng, 4)
    for (block, leaf) in pairs(mesh.blocks)
        offset = (block - 1) * nx * ny
        for j in 1:ny, i in 1:nx
            x = leaf.origin[1] + (i - 0.5) * leaf.spacing[1]
            y = leaf.origin[2] + (j - 0.5) * leaf.spacing[2]
            f[offset + (j - 1) * nx + i] = c[1] * cospi(x) * cospi(y) + c[2] * sinpi(2x) * cospi(y) + c[3] * x * y + c[4]
        end
    end
    return f .* (operator.volumes .> 0)
end

split_geometry(mesh; shift = 0.0) = GeometryField(mesh, (x, y) -> solv_band_solid(x, y) + shift; lipschitz = 1)

@testset "projected V-cycle: SPD, symmetric, mean-free range" begin
    println("== dense (exact spectrum of B A on the mean-free space) ==")
    for n in (16, 32, 64)
        op = PoissonOperator(solv_uniform(n))
        dense_vcycle_check(op, TreeGeometricMultigridPreconditioner(op), "uniform $(n)^2")
    end
    op = PoissonOperator(solv_uniform(32; periodic = (true, true)))
    dense_vcycle_check(op, TreeGeometricMultigridPreconditioner(op), "periodic 32^2")
    op = PoissonOperator(solv_amr(16))
    dense_vcycle_check(op, TreeGeometricMultigridPreconditioner(op), "composite base 16, 2 levels")
    op = PoissonOperator(solv_amr(16; levels = 3))
    dense_vcycle_check(op, TreeGeometricMultigridPreconditioner(op), "composite base 16, 3 levels")
    for ratio in (1.0e3, 1.0e6)
        op = PoissonOperator(solv_drop_field(solv_uniform(32)); density = (ratio, 1.0))
        dense_vcycle_check(op, TreeGeometricMultigridPreconditioner(op), "drop ratio $(Int(ratio)), 32^2")
        op = PoissonOperator(solv_drop_field(solv_amr(16)); density = (ratio, 1.0))
        dense_vcycle_check(op, TreeGeometricMultigridPreconditioner(op), "drop ratio $(Int(ratio)), composite 16")
    end
    for n in (32, 64)
        geometry = split_geometry(solv_uniform(n))
        op = PoissonOperator(geometry)
        dense_vcycle_check(op, TreeGeometricMultigridPreconditioner(op; geometry), "split cut-cell $(n)^2")
    end
    for (name, boundary) in (
            ("split 32^2, Outflow top (anchors one)", VelocityBoundaryConditions(left = NoSlip(), right = NoSlip(), bottom = NoSlip(), top = Outflow())),
            ("split 32^2, Outflow right (anchors both)", VelocityBoundaryConditions(left = NoSlip(), right = Outflow(), bottom = NoSlip(), top = NoSlip())),
        )
        geometry = split_geometry(solv_uniform(32))
        op = PoissonOperator(geometry; velocity_boundary = boundary)
        dense_vcycle_check(op, TreeGeometricMultigridPreconditioner(op; geometry), name)
    end
    for shift in (3.0e-5, 3.0e-6)
        # band edge moved to within `shift` of a cell edge: cut cells with open fraction ~ shift/h
        geometry = split_geometry(solv_uniform(32); shift = shift - 0.29 * 32 % 1 / 32)
        op = PoissonOperator(geometry)
        fractions = filter(>(0), op.volumes ./ AMR._cell_volumes(op.mesh, Float64))
        dense_vcycle_check(op, TreeGeometricMultigridPreconditioner(op; geometry), @sprintf("split 32^2, min open fraction %.0e", minimum(fractions)))
    end
    geometry = split_geometry(solv_amr(16))
    op = PoissonOperator(geometry)
    dense_vcycle_check(op, TreeGeometricMultigridPreconditioner(op; geometry), "composite 16 + solid band")
    # Dirichlet-inactive operator with a blob of inactive cells (liquid-carrier type), no geometry: activity by the "any fine cell" rule
    mesh = solv_uniform(32)
    nx, ny = mesh.block_size
    n = AMR.ncells(mesh)
    mask = trues(n)
    for (block, leaf) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
        x = leaf.origin[1] + (i - 0.5) * leaf.spacing[1]; y = leaf.origin[2] + (j - 0.5) * leaf.spacing[2]
        mask[(block - 1) * nx * ny + (j - 1) * nx + i] = hypot(x - 0.4, y - 0.55) < 0.3
    end
    for mode in (:dirichlet, :decoupled)
        volumes = ifelse.(mask, AMR._cell_volumes(mesh, Float64), 0.0)
        op = EllipticOperator(mesh; volumes, active_cells = mask, inactive_condition = mode)
        dense_vcycle_check(op, TreeGeometricMultigridPreconditioner(op), "inactive blob, $(mode)")
    end

    println("== matrix-free finest level equals the assembled one ==")
    for mesh in (solv_uniform(32), solv_amr(16))
        a = PoissonOperator(mesh)
        free = PoissonOperator(mesh; assemble_matrix = false)
        pa = TreeGeometricMultigridPreconditioner(a); pf = TreeGeometricMultigridPreconditioner(free)
        n = length(a.volumes)
        rng = MersenneTwister(3)
        u = randn(rng, n); xa = zeros(n); xf = zeros(n)
        ldiv!(xa, pa, u); ldiv!(xf, pf, u)
        d = norm(xa - xf, Inf) / norm(xa, Inf)
        @printf("  n=%5d: |B_assembled u - B_matrix-free u|/|B u| = %.1e\n", n, d)
        @test d <= 1.0e-10
    end

    println("== large (sampled symmetry, Rayleigh quotients, Ritz values of PCG) ==")
    for n in (128, 256)
        op = PoissonOperator(solv_uniform(n))
        sampled_checks(op, TreeGeometricMultigridPreconditioner(op), "uniform $(n)^2")
    end
    for (base, levels) in ((32, 3), (64, 2), (64, 3))
        op = PoissonOperator(solv_amr(base; levels))
        sampled_checks(op, TreeGeometricMultigridPreconditioner(op), "composite base $base, $levels levels")
    end
    for ratio in (1.0e3, 1.0e6)
        op = PoissonOperator(solv_drop_field(solv_amr(64)); density = (ratio, 1.0))
        sampled_checks(op, TreeGeometricMultigridPreconditioner(op), "drop ratio $(Int(ratio)), composite 64")
        op = PoissonOperator(solv_drop_field(solv_uniform(128)); density = (ratio, 1.0))
        sampled_checks(op, TreeGeometricMultigridPreconditioner(op), "drop ratio $(Int(ratio)), 128^2")
    end
    geometry = split_geometry(solv_uniform(128))
    op = PoissonOperator(geometry)
    sampled_checks(op, TreeGeometricMultigridPreconditioner(op; geometry), "split cut-cell 128^2")
    geometry = split_geometry(solv_amr(32; levels = 2))
    op = PoissonOperator(geometry)
    sampled_checks(op, TreeGeometricMultigridPreconditioner(op; geometry), "composite base 32 + solid band")
end
