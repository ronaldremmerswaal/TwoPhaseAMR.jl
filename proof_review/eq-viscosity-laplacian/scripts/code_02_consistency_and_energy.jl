# Code check 02: consistency order of the single-phase Laplacian by row class, solution-level order, and the energy effect of the
# non-symmetric band.
#
# Label:  eq:viscosity:laplacian (also :single-midpoint for the energy statement)
# Source: docs/techref/chapters/09_viscosity.tex:93-130, 372-380 (single-phase step has no energy identity);
#         src/operators/staggered_diffusion.jl
# Checks:
#   K1  truncation error |L u* + V^-1 b - Delta u*| for u* = sin(pi x/L) sin(pi y/L) (zero on walls) on the walled composite mesh
#       (base n = 16, 32, 64, right part refined) per class: core interior rows, band rows, rows with a wall connection;
#       observed order of the max error per class;
#   K2  solution-level order: (I - L)u = (I - Delta)u* with the code's L (wall data 0): max error of u_h - u* per level;
#   K3  the CN amplification in the V-norm  ||V^{1/2}(I - aL)^{-1}(I + aL)V^{-1/2}||_2 for a = dt nu/2 in 1e-4 ... 1e4 (walled and periodic
#       composite, both components): > 1 means the energy (1/2) u'V u of an exact solve can GROW (u^n = u^(1)); the size is the effect of the
#       non-symmetric band on the energy identity of eq:viscosity:energy (which does not apply to the single-phase operator);
#   K4  momentum: sum_f V_f (L u)_f on the periodic uniform mesh = 0, on the periodic composite mesh relative size for smooth/random u.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-laplacian/scripts/code_02_consistency_and_energy.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA

composite(n; periodic = (false, false), L = 8.0) = BlockTreeMesh(; base_mesh = (n, n), block_size = (4, 4), extent = (L, L), periodic, max_level = 1,
    refine = (key, o, e) -> key.level == 0 && key.i >= div(n, 8))

function classes(mesh, op)
    out = Dict{Symbol, Vector{Tuple{Int, Int}}}(:core => [], :band => [], :wall => [])
    for (component, axis) in ((op.x, 1), (op.y, 2))
        L = component.laplacian; V = component.volumes; S = component.stiffness
        base = -spdiagm(0 => inv.(V)) * S
        n = length(V)
        wall_dofs = Set(c.degree_of_freedom for c in component.wall_connections)
        for r in 1:n
            band = maximum(abs, Vector(L[r, :]) .- Vector(base[r, :])) > 1.0e-12
            push!(out[band ? :band : (r in wall_dofs ? :wall : :core)], (axis, r))
        end
    end
    return out
end

function truncation(n)
    L = 8.0
    mesh = composite(n)
    op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
    ustar(x, y) = sin(pi * x / L) * sin(pi * y / L)
    lap(x, y) = -2 * (pi / L)^2 * ustar(x, y)
    cls = classes(mesh, op)
    err = Dict(:core => 0.0, :band => 0.0, :wall => 0.0)
    for (component, axis) in ((op.x, 1), (op.y, 2))
        pts = [mesh.faces[g].center for g in component.local_to_global]
        u = [ustar(p...) for p in pts]
        applied = component.laplacian * u        # wall values of u* are zero: no source
        for (k, rows) in cls, (ax, r) in rows
            ax == axis || continue
            err[k] = max(err[k], abs(applied[r] - lap(pts[r]...)))
        end
    end
    return err, Dict(k => length(v) for (k, v) in cls)
end

function solution_error(n)
    L = 8.0
    mesh = composite(n)
    op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
    ustar(x, y) = sin(pi * x / L) * sin(pi * y / L)
    worst = 0.0
    for component in (op.x, op.y)
        pts = [mesh.faces[g].center for g in component.local_to_global]
        us = [ustar(p...) for p in pts]
        f = us .- (-2 * (pi / L)^2) .* us .* 1.0         # (I - Delta) u*  with Delta u* = -2 (pi/L)^2 u*
        A = sparse(I, length(us), length(us)) - component.laplacian
        u = A \ f
        worst = max(worst, maximum(abs, u .- us))
    end
    return worst
end

@testset "K1/K2: consistency by class and solution order" begin
    prev = nothing; prev_sol = nothing
    for n in (16, 32, 64)
        err, counts = truncation(n)
        sol = solution_error(n)
        orders = prev === nothing ? "" : join(["$k: " * string(round(log2(prev[k] / err[k]), digits = 2)) for k in (:core, :band, :wall)], ", ")
        println("n=", n, " rows ", counts, " truncation max error ", Dict(k => round(v, sigdigits = 3) for (k, v) in err), "  orders: ", orders,
            "  | solution error ", round(sol, sigdigits = 3), prev_sol === nothing ? "" : "  order " * string(round(log2(prev_sol / sol), digits = 2)))
        prev = err; prev_sol = sol
    end
end

@testset "K3: CN energy amplification of the single-phase operator" begin
    meshes = (("walled composite", composite(16)), ("periodic composite", composite(16; periodic = (true, true))))
    for (name, mesh) in meshes
        op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
        for (component, axis) in ((op.x, 1), (op.y, 2))
            V = component.volumes; L = Matrix(component.laplacian); n = length(V)
            Vh = Diagonal(sqrt.(V)); Vhi = Diagonal(1 ./ sqrt.(V))
            rho = maximum(abs, eigvals(L))                    # spectral radius of L
            amp(a) = opnorm(Vh * ((I - a * L) \ (I + a * L)) * Vhi)
            # diffusion number ND = a rho(L) = (dt nu/2) rho(L): ND = 1 is the explicit-Euler-like limit nu dt rho/2
            table = [(nd, amp(nd / rho)^2) for nd in (0.1, 1.0, 10.0, 100.0, 1.0e4, 1.0e8)]
            worst = maximum(amp(a) for a in 10.0 .^ (-4:0.25:8) ./ rho .* rho)
            println(rpad(name, 20), "axis ", axis, ": energy growth factor ||R||_V^2 vs ND = (dt nu/2) rho(L): ", [(nd, round(g, sigdigits = 4)) for (nd, g) in table])
            @test all(g -> g < 3.0, last.(table))                  # recorded size of the effect (periodic composite: up to ~2.7)
        end
    end
end

@testset "K4: momentum of the explicit rate on periodic meshes" begin
    rng = MersenneTwister(4)
    uniform = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, true))
    for (name, mesh) in (("uniform periodic", uniform), ("periodic composite", composite(16; periodic = (true, true))))
        op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
        for (component, axis) in ((op.x, 1), (op.y, 2))
            V = component.volumes
            u = randn(rng, length(V))
            rate = component.laplacian * u
            rel = abs(dot(V, rate)) / dot(V, abs.(rate))
            smooth = [sin(2pi * mesh.faces[g].center[1] / 8) * cos(2pi * mesh.faces[g].center[2] / 8 + 0.3) for g in component.local_to_global]
            rate_s = component.laplacian * smooth
            rel_s = abs(dot(V, rate_s)) / dot(V, abs.(rate_s))
            println(rpad(name, 20), "axis ", axis, ": |1'V L u| / 1'V|L u| = ", rel, " (random), ", rel_s, " (smooth mode)")
            name == "uniform periodic" && @test rel <= 1.0e-13 && rel_s <= 1.0e-13
        end
    end
end
