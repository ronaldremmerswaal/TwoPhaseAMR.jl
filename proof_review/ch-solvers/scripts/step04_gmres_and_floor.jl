# Audit ch:solvers -- claim 4: (a) the GMRES stopping bound eq:solvers:gmres (||r||_2 <= tau * min V  =>  max |r_c|/V_c <= tau) and
# the code that implements it (_gmres_absolute_tolerance); (b) the factorised round-off floor uses an entrywise larger
# |A| than the assembled one, so roundoff_floor(factorised) >= roundoff_floor(assembled) for every iterate, and the
# 128 eps scale is adequate for the factorised action's evaluation error (compared with an exact BigFloat evaluation).
# Source: docs/techref/chapters/11_solvers.tex:236-258 (eq:solvers:floor), 280-291 (eq:solvers:gmres); code
#   src/solvers/multigrid/elliptic_solver.jl:358-364 (_gmres_absolute_tolerance), src/solvers/multigrid/cycles.jl:80-160.
# Run: julia --project=docs/techref/figures proof_review/ch-solvers/scripts/step04_gmres_and_floor.jl
isdefined(Main, :SOLV_DROP) || include(joinpath(@__DIR__, "_prelude.jl"))

@testset "GMRES stopping bound" begin
    rng = MersenneTwister(1)
    # (a1) the implication for random residuals, volumes over 8 decades
    for trial in 1:2000
        n = rand(rng, 5:60)
        V = 10.0 .^ (8 .* rand(rng, n) .- 4)
        V[rand(rng, 1:n)] = 0.0                       # a covered cell does not enter the strong norm
        tau = 10.0^(-12 * rand(rng))
        r = randn(rng, n); r[V .== 0] .= 0
        r .*= tau * minimum(V[V .> 0]) / norm(r) * rand(rng)        # ||r||_2 <= tau * min V
        strong = maximum(abs(r[i]) / V[i] for i in 1:n if V[i] > 0)
        @test strong <= tau * (1 + 1.0e-14)
    end
    # (a2) the code's tolerance is target * min(1, min V): at most the text's  target * min V.
    f = AMR._gmres_absolute_tolerance
    @test f(1.0e-6, [0.25, 3.0]) == 1.0e-6 * 0.25                  # volumes < 1: equals the text
    @test f(1.0e-6, [4.0, 9.0]) == 1.0e-6 * 1.0                    # all volumes > 1: min(1, min V) = 1 < min V = 4
    @test f(1.0e-6, [0.0, 2.0]) == 1.0e-6 * 1.0                    # a covered cell (V = 0) is skipped
    println("  _gmres_absolute_tolerance: target*min(1, min_{V>0} V)  (text: target*min V; stricter when every volume exceeds 1)")
    # (a3) real GMRES / FGMRES solves reach the strong target, also with cut cells
    for (name, make) in (
            ("uniform 32^2", () -> (PoissonOperator(solv_uniform(32)), nothing)),
            ("split cut-cell 32^2", () -> (g = GeometryField(solv_uniform(32), solv_band_solid; lipschitz = 1); (PoissonOperator(g), g))),
        )
        for workspace in (Krylov.GmresWorkspace, Krylov.FgmresWorkspace)
            operator, geometry = make()
            options = isnothing(geometry) ? (;) : (; geometry)
            solver = EllipticSolver(operator; krylov = workspace, preconditioner = :tree_geometric, preconditioner_options = options)
            rhs = CellField(operator.mesh)
            for (index, ref) in pairs(operator.references)
                AMR._set!(rhs, solv_noise(index), ref)
            end
            solution = CellField(operator.mesh)
            stats = solve!(solution, solver, rhs; rtol = 1.0e-9, max_cycles = 500)
            @printf("  %-22s %-16s converged=%s iterations=%d final strong residual %.2e <= target %.2e\n", name, nameof(workspace), stats.converged, stats.cycles, stats.final_linf, stats.target_linf)
            @test stats.converged
            @test stats.final_linf <= stats.target_linf
        end
    end
end

@testset "round-off floor: factorised >= assembled" begin
    rng = MersenneTwister(4)
    cases = Any[]
    push!(cases, ("uniform 16^2", PoissonOperator(solv_uniform(16))))
    push!(cases, ("composite 16", PoissonOperator(solv_amr(16))))
    push!(cases, ("drop 1e3 composite", PoissonOperator(solv_drop_field(solv_amr(16)); density = (1000.0, 1.0))))
    g = GeometryField(solv_uniform(32), solv_band_solid; lipschitz = 1)
    push!(cases, ("split cut-cell 32^2", PoissonOperator(g)))
    g = GeometryField(solv_amr(16), solv_band_solid; lipschitz = 1)
    push!(cases, ("composite 16 + solid", PoissonOperator(g)))
    sides = VelocityBoundaryConditions(left = NoSlip(), right = Outflow(), bottom = NoSlip(), top = NoSlip())
    push!(cases, ("composite, Outflow right", PoissonOperator(solv_amr(16); velocity_boundary = sides)))
    push!(cases, ("reaction 3, composite", EllipticOperator(solv_amr(16); reaction = 3.0, diffusion = 0.7)))
    for (name, operator) in cases
        n = length(operator.volumes)
        action = AMR.FactorizedEllipticOperator(operator)
        tmp_a = zeros(n); tmp_f = zeros(n)
        min_ratio = Inf; max_ratio = 0.0
        for draw in 1:6
            x = randn(rng, n) .* (draw > 3 ? 10.0 .^ (6 .* rand(rng, n) .- 3) : 1.0)
            b = randn(rng, n)
            floor_a = AMR._roundoff_residual_floor!(operator.matrix, x, b, operator.volumes, tmp_a)
            floor_f = AMR._roundoff_residual_floor!(action, x, b, operator.volumes, tmp_f)
            # entrywise: (|A||x|)_assembled <= (|A_fact||x|)_factorised on every cell (the buffers hold these vectors)
            @test all(tmp_a .<= tmp_f .* (1 + 1.0e-12) .+ 1.0e-300)
            @test floor_f >= floor_a * (1 - 1.0e-14)
            ratio = floor_f / floor_a
            min_ratio = min(min_ratio, ratio); max_ratio = max(max_ratio, ratio)
        end
        @printf("  %-26s n=%5d  floor_factorised / floor_assembled in [%.3f, %.3f]\n", name, n, min_ratio, max_ratio)
    end
end

# The 128 eps scale against the actual evaluation error of the factorised action: evaluate A x exactly in BigFloat from the
# same Float64 factors (J, sqrt(c), R, constants) and compare with the Float64 action.
@testset "floor bounds the factorised evaluation error" begin
    rng = MersenneTwister(9)
    for (name, operator) in (
            ("composite 16", PoissonOperator(solv_amr(16))),
            ("drop 1e3 composite", PoissonOperator(solv_drop_field(solv_amr(16)); density = (1000.0, 1.0))),
            ("composite + solid", PoissonOperator(GeometryField(solv_amr(16), solv_band_solid; lipschitz = 1))),
        )
        n = length(operator.volumes)
        mesh = operator.mesh
        nx, ny = mesh.block_size
        action = AMR.FactorizedEllipticOperator(operator)
        x = randn(rng, n) .* (operator.volumes .> 0)
        y = zeros(n); mul!(y, action, x)
        # exact evaluation (BigFloat) of the same factorisation
        setprecision(BigFloat, 200) do
            xb = BigFloat.(x)
            nfaces = length(mesh.faces)
            fin = zeros(BigFloat, nfaces)
            for (i, face) in pairs(mesh.faces)
                a = AMR._cell_index(face.minus, nx, ny); b = AMR._cell_index(face.plus, nx, ny)
                xm = iszero(operator.constraints[a]) ? xb[a] : zero(BigFloat)
                xp = iszero(operator.constraints[b]) ? xb[b] : zero(BigFloat)
                fin[i] = BigFloat(operator.square_root_face_coefficient[i]) * BigFloat(face.area) * (xp - xm)
            end
            metric = operator.gradient_operator.inverse_metric
            fout = if metric isa AMR.DiagonalInverseFaceMetric
                BigFloat.(metric.diagonal) .* fin
            else
                BigFloat.(metric.matrix) * fin
            end
            exact = zeros(BigFloat, n)
            for c in 1:n
                exact[c] = iszero(operator.constraints[c]) ?
                    (BigFloat(operator.reaction) * BigFloat(operator.volumes[c]) + BigFloat(operator.diffusion) * BigFloat(operator.boundary_diagonal[c])) * xb[c] : xb[c]
            end
            for (i, face) in pairs(mesh.faces)
                a = AMR._cell_index(face.minus, nx, ny); b = AMR._cell_index(face.plus, nx, ny)
                flux = BigFloat(operator.diffusion) * BigFloat(face.area) * BigFloat(operator.square_root_face_coefficient[i]) * fout[i]
                iszero(operator.constraints[a]) && (exact[a] -= flux)
                iszero(operator.constraints[b]) && (exact[b] += flux)
            end
            err = [Float64(abs(BigFloat(y[c]) - exact[c])) / operator.volumes[c] for c in 1:n if operator.volumes[c] > 0]
            tmp = zeros(n)
            AMR._roundoff_residual_floor!(action, x, zeros(n), operator.volumes, tmp)
            scale = maximum(tmp[c] / operator.volumes[c] for c in 1:n if operator.volumes[c] > 0)
            worst = maximum(err) / (eps() * scale)
            @printf("  %-22s max strong evaluation error = %.2e eps * max(|A_fact||x|/V) -> ratio %.3f (floor allows 128)\n", name, maximum(err), worst)
            @test worst < 128
        end
    end
end

# Sharpness of eq:solvers:gmres: the implication is sufficient only and conservative by the ratio ||r||_2 / max|r_c| together with
# V_c / min V; with tiny cut cells min V is small and the Euclidean tolerance can be attained only after many extra iterations
# (or not at all before round-off). Compare GMRES and CG on the split domain with a cut cell of open fraction ~1e-3.
@testset "conservatism of the Euclidean GMRES stop" begin
    h = 1 / 32
    shift = 3.0e-5 - 0.29 * 32 % 1 / 32
    for (name, delta) in (("regular cut cells", 0.0), ("open fraction ~1e-3", shift))
        geometry = GeometryField(solv_uniform(32), (x, y) -> solv_band_solid(x, y) + delta; lipschitz = 1)
        iterations = Dict{Any, Any}()
        for workspace in (Krylov.CgWorkspace, Krylov.GmresWorkspace)
            operator = PoissonOperator(geometry)
            solver = EllipticSolver(operator; krylov = workspace, preconditioner_options = (; geometry))
            rhs = CellField(operator.mesh)
            for (index, ref) in pairs(operator.references)
                AMR._set!(rhs, solv_noise(index), ref)
            end
            solution = CellField(operator.mesh)
            stats = solve!(solution, solver, rhs; rtol = 1.0e-9, max_cycles = 500)
            iterations[workspace] = (stats.cycles, stats.converged, stats.final_linf, stats.target_linf)
        end
        fractions = filter(>(0), PoissonOperator(geometry).volumes ./ AMR._cell_volumes(geometry.mesh, Float64))
        @printf(
            "  %-22s min open fraction %.1e: CG %d its (converged %s, final %.1e of target %.1e); GMRES %d its (converged %s, final %.1e of target %.1e)\n",
            name, minimum(fractions), iterations[Krylov.CgWorkspace]..., iterations[Krylov.GmresWorkspace]...
        )
        @test iterations[Krylov.GmresWorkspace][2]
        @test iterations[Krylov.GmresWorkspace][1] >= iterations[Krylov.CgWorkspace][1]
    end
end
