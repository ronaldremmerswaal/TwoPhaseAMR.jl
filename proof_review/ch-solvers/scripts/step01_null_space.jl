# Audit ch:solvers -- claim 1: eq:solvers:null.
#   ker A = span{1_C : C an unanchored component}  (components of the graph of active cells joined by interior faces with
#   a nonzero coefficient; anchored = a cell with boundary conductance > 0 or a nonzero face to an inactive cell), and
#   A x = b is solvable iff 1_C' b = 0 for every unanchored C.
# Source: docs/techref/chapters/11_solvers.tex:150-205 (eq:solvers:null, eq:solvers:energy), code
#   src/operators/elliptic.jl (EllipticOperator), src/solvers/multigrid/elliptic_solver.jl (_refresh_elliptic_components!).
# The prediction is computed by an independent union-find (`predicted_components` in _prelude.jl) and compared with the
# dense eigen-decomposition of the assembled matrix and with the code's own `components.anchors`.
# Run: julia --project=docs/techref/figures proof_review/ch-solvers/scripts/step01_null_space.jl
isdefined(Main, :SOLV_DROP) || include(joinpath(@__DIR__, "_prelude.jl"))

# Dimension of the kernel from the spectrum of the dense symmetric matrix (relative gap test) and agreement of the
# kernel with the span of the predicted indicators: A * 1_C = 0 and rank([1_C]) = dim ker.
function check_kernel(operator, name; expect_code = true)
    A = Matrix(operator.matrix)
    @assert norm(A - A', Inf) <= 1.0e-13 * norm(A, Inf) "$name: matrix not symmetric"
    lambda = eigvals(Symmetric(A))
    scale = last(lambda)
    @assert first(lambda) >= -1.0e-12 * scale "$name: not positive semidefinite ($(first(lambda)))"
    kernel_dim = count(<=(1.0e-11 * scale), lambda)
    gap = kernel_dim < length(lambda) ? lambda[kernel_dim + 1] / scale : Inf
    groups = predicted_components(operator)
    @test kernel_dim == length(groups)
    @test gap > 1.0e-9 || kernel_dim == length(lambda)        # the kernel is separated from the rest of the spectrum
    n = length(operator.volumes)
    worst = 0.0
    for g in groups
        indicator = zeros(n)
        indicator[g] .= 1
        worst = max(worst, norm(A * indicator, Inf) / scale)
    end
    @test worst <= 1.0e-12
    # solvability: the range is the orthogonal complement of the indicators
    rng = MersenneTwister(11)
    b = randn(rng, n)
    for g in groups
        b[g] .-= sum(b[g]) / length(g)           # compatible (A is symmetric: range = ker^perp)
    end
    x = pinv(A; rtol = 1.0e-10) * b
    @test norm(A * x - b, Inf) <= 1.0e-9 * norm(b, Inf)
    if !isempty(groups)
        b2 = copy(b)
        b2[groups[1]] .+= 1.0                      # break one compatibility condition
        x2 = pinv(A; rtol = 1.0e-10) * b2
        @test norm(A * x2 - b2, Inf) >= 0.5 * norm(b2[groups[1]], Inf) / 2
    end
    if expect_code && !isnothing(operator.components)
        AMR._refresh_elliptic_components!(operator.components, operator)
        @test sort(operator.components.anchors) == sort(first.(groups)) # labels = smallest index
        @test operator.singular == !isempty(groups)
    end
    @printf("  %-34s n=%5d  unanchored=%d  kernel=%d  gap=%.2e  max|A 1_C|/|A|=%.1e\n", name, n, length(groups), kernel_dim, gap, worst)
    return length(groups)
end

@testset "eq:solvers:null" begin
    println("== split cut-cell domain (decoupled), 32x32 ==")
    mesh = solv_uniform(32)
    for (name, boundary) in (
            ("closed", nothing),
            ("Outflow top", VelocityBoundaryConditions(left = NoSlip(), right = NoSlip(), bottom = NoSlip(), top = Outflow())),
            ("Outflow right", VelocityBoundaryConditions(left = NoSlip(), right = Outflow(), bottom = NoSlip(), top = NoSlip())),
            ("Outflow bottom+top", VelocityBoundaryConditions(left = NoSlip(), right = NoSlip(), bottom = Outflow(), top = Outflow())),
        )
        geometry = GeometryField(mesh, solv_band_solid; lipschitz = 1)
        operator = PoissonOperator(geometry; velocity_boundary = boundary)
        k = check_kernel(operator, "split 32 / " * name)
        name == "closed" && @test k == 2
        name == "Outflow top" && @test k == 1
        name == "Outflow right" && @test k == 0
        name == "Outflow bottom+top" && @test k == 0
    end

    println("== composite mesh with a solid (non-diagonal metric at level transitions, cut cells) ==")
    amr = solv_amr(8; levels = 2, block = 4)
    geometry = GeometryField(amr, solv_band_solid; lipschitz = 1)
    for (name, boundary) in (
            ("closed", nothing),
            ("Outflow top", VelocityBoundaryConditions(left = NoSlip(), right = NoSlip(), bottom = NoSlip(), top = Outflow())),
        )
        operator = PoissonOperator(geometry; velocity_boundary = boundary)
        @assert !(operator.gradient_operator.inverse_metric isa AMR.DiagonalInverseFaceMetric) "metric is diagonal: no level transition"
        check_kernel(operator, "composite+solid / " * name)
    end

    println("== composite mesh, no solid, periodic x: kernel = constants ==")
    pamr = solv_amr(8; levels = 2, block = 4, periodic = (true, false))
    check_kernel(PoissonOperator(pamr), "composite periodic")
    @test check_kernel(PoissonOperator(solv_amr(8; levels = 2, block = 4)), "composite closed") == 1

    println("== random active masks and zero face coefficients, :dirichlet / :decoupled, random open sides ==")
    rng = MersenneTwister(2026)
    total_unanchored = 0
    total_anchored_by_inactive = 0
    for trial in 1:60
        composite = isodd(trial)
        mesh = composite ? solv_amr(4; levels = 2, block = 4) : solv_uniform(8; block = 4)
        n = AMR.ncells(mesh)
        nf = length(mesh.faces)
        mask = rand(rng, n) .< 0.75
        coefficient = [rand(rng) < 0.3 ? 0.0 : 10.0^(4 * (rand(rng) - 0.5)) for _ in 1:nf]
        nb = length(mesh.boundary_faces)
        bcoef = [rand(rng) < 0.3 ? 0.0 : rand(rng) + 0.1 for _ in 1:nb]
        sides = Tuple(rand(rng) < 0.3 for _ in 1:4)
        mode = rand(rng, (:dirichlet, :decoupled))
        volumes = ifelse.(mask, 0.5 .+ rand(rng, n), 0.0)
        operator = EllipticOperator(
            mesh; face_coefficient = coefficient, boundary_face_coefficient = bcoef,
            velocity_boundary = sides, volumes, active_cells = mask, inactive_condition = mode,
        )
        total_unanchored += check_kernel(operator, "random $trial $(composite ? "amr" : "uniform") $mode"; expect_code = true)
    end
    @info "random trials done" total_unanchored
    @test total_unanchored > 20    # the random cases do contain unanchored components
end

# Sharpness of the "nonzero" criterion: the equivalence is an exact-arithmetic statement about the coefficient being
# zero; a coefficient eps > 0 joins the components and leaves a nonzero eigenvalue of order eps.
@testset "nearly decoupled components" begin
    println("== a small coefficient on the faces cutting a 4x4 mesh in two halves ==")
    mesh = solv_uniform(8; block = 4)
    nx = 4
    for eps in (1.0e-2, 1.0e-6, 1.0e-10)
        coefficient = Float64[]
        for face in mesh.faces
            a = AMR._cell_index(face.minus, 4, 4); b = AMR._cell_index(face.plus, 4, 4)
            xa = mesh.blocks[face.minus.block].origin[1] + (face.minus.i - 0.5) * mesh.blocks[face.minus.block].spacing[1]
            xb = mesh.blocks[face.plus.block].origin[1] + (face.plus.i - 0.5) * mesh.blocks[face.plus.block].spacing[1]
            push!(coefficient, (xa < 0.5) != (xb < 0.5) ? eps : 1.0)
        end
        operator = EllipticOperator(mesh; face_coefficient = coefficient)
        lambda = eigvals(Symmetric(Matrix(operator.matrix)))
        @printf("  eps = %.0e: lambda_1 = %.1e, lambda_2 = %.2e (components: %d)\n", eps, lambda[1], lambda[2], length(predicted_components(operator)))
        @test length(predicted_components(operator)) == 1
        @test lambda[2] > 0.1 * eps           # the second eigenvalue is of order eps, not zero
        @test lambda[2] < 10 * eps * 8 * 8
    end
end
