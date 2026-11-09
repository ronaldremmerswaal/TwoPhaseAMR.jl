# Code check 01: eq:viscosity:laplacian -- the single-phase Laplacian: conservative core, the corrected band B, its quadratic exactness
# and minimality, the five-point limit, and the non-symmetric band.
#
# Label:  eq:viscosity:laplacian (also :stiffness for the core)
# Source: docs/techref/chapters/09_viscosity.tex:93-130; src/operators/staggered_diffusion.jl (_corrected_laplacian,
#         _quadratic_laplacian_stencil, _expand_graph_neighbourhood!)
# Checks (components x, y; walled composite (docs fixture), periodic composite, 6 random composite meshes with levels 1-2):
#   L1  uniform periodic mesh: L equals the five-point Laplacian (independent face lookup);
#   L2  the set B (rows that differ from the core -V^-1 S) equals the chapter's set: endpoints of refinement-interface connections
#       without a wall connection, plus the unknowns within two graph rings reached without passing wall unknowns (independent BFS);
#   L3  every band row reproduces Delta p for p in {1,x,y,x^2,xy,y^2} (periodic meshes; with walls only rows away from wall values), uses
#       unknowns within three graph rings, and equals the KKT minimiser of sum a^2 (|r|^2/l^2 + 1/64), l = largest coordinate offset;
#   L4  the core rows OUTSIDE the band: quadratic exactness (they are the five-point rows of regular conforming patches) -- count
#       and size of the inconsistency at core rows that touch a refinement interface (only possible next to a wall);
#   L5  the band makes V L non-symmetric: number of asymmetric entries; spectrum: max Re(eig L) <= 0?; lambda_min of sym(-V L)
#       relative to the core stiffness (effect on the energy identity of the Crank--Nicolson step with u^n = u^(1));
#   L6  momentum: 1' V L u = 0 on the uniform periodic mesh, != 0 on a periodic composite mesh (band rows not conservative): size.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-laplacian/scripts/code_01_laplacian.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA

function component_data(mesh, axis)
    field = FaceField{axis}(mesh)
    schedule = AMR.StaggeredAdvectionWorkspace(field).schedule
    g2l, l2g = AMR._component_local_to_global(mesh, axis)
    conns = AMR._dual_diffusion_connections(mesh, schedule, l2g)
    adjacency = [Int[] for _ in l2g]
    for c in conns
        push!(adjacency[c.minus], c.plus); push!(adjacency[c.plus], c.minus)
    end
    foreach(unique!, adjacency)
    return conns, adjacency, l2g
end

function rings(adjacency, start, depth; excluded = Set{Int}())
    sel = Set(start); frontier = collect(start)
    for _ in 1:depth
        nxt = Int[]
        for i in frontier, j in adjacency[i]
            j in excluded && continue
            j in sel && continue
            push!(sel, j); push!(nxt, j)
        end
        frontier = nxt
    end
    return sel
end

@testset "L1: five-point Laplacian on the uniform periodic mesh" begin
    mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, true))
    op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
    for (component, axis) in ((op.x, 1), (op.y, 2))
        pts = [mesh.faces[g].center for g in component.local_to_global]
        index = Dict((round(Int, 4p[1]), round(Int, 4p[2])) => i for (i, p) in enumerate(pts))
        L = component.laplacian
        worst = 0.0
        for (i, p) in enumerate(pts)
            key = (round(Int, 4p[1]), round(Int, 4p[2]))
            nb = [index[(mod(key[1] + dx, 32), mod(key[2] + dy, 32))] for (dx, dy) in ((2, 0), (-2, 0), (0, 2), (0, -2))]
            expected = zeros(length(pts)); expected[nb] .= 1 / 0.25; expected[i] = -4 / 0.25
            worst = max(worst, maximum(abs, Vector(L[i, :]) .- expected))
        end
        @test worst <= 1.0e-12
    end
end

fixtures = Any[]
push!(fixtures, ("walled composite", BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false), max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= 2)))
push!(fixtures, ("periodic composite", BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, true), max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= 2)))
let rng = MersenneTwister(808)
    for (periodic, levels) in (((true, true), 1), ((true, true), 2), ((false, false), 1), ((false, false), 2), ((true, false), 1), ((true, false), 2))
        push!(fixtures, ("random periodic=$periodic L=$levels", MA.random_mesh(rng; periodic, levels, refine_probability = 0.5)))
    end
end

@testset "L2-L6 band, exactness, minimality, symmetry" begin
    rng = MersenneTwister(81)
    for (name, mesh) in fixtures
        op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
        info = Dict{Symbol, Any}()
        nband = 0; worst_q = 0.0; worst_kkt = 0.0; set_ok = true; worst_core = 0.0; ncore_bad = 0; nasym = 0; bad_rows = []
        for (component, axis) in ((op.x, 1), (op.y, 2))
            conns, adjacency, l2g = component_data(mesh, axis)
            L = component.laplacian; V = component.volumes; S = component.stiffness
            base = -spdiagm(0 => inv.(V)) * S
            n = length(V)
            diffrows = Set(r for r in 1:n if maximum(abs, Vector(L[r, :]) .- Vector(base[r, :])) > 1.0e-12)
            wall_dofs = Set(c.degree_of_freedom for c in component.wall_connections)
            seed = Set{Int}()
            for c in conns
                c.refinement_interface || continue
                c.minus in wall_dofs || push!(seed, c.minus)
                c.plus in wall_dofs || push!(seed, c.plus)
            end
            expected_B = rings(adjacency, seed, 2; excluded = wall_dofs)
            set_ok &= diffrows == expected_B
            nband += length(diffrows)
            pos(i) = mesh.faces[l2g[i]].center
            # quadratics at band rows
            for r in diffrows
                nbr = rings(adjacency, [r], 3)
                cols = [j for j in 1:n if L[r, j] != 0]
                @test all(j -> j in nbr, cols)
                x0 = pos(r)
                offs = [(wrapped(pos(j)[1] - x0[1], mesh.extent[1], mesh.periodic[1]), wrapped(pos(j)[2] - x0[2], mesh.extent[2], mesh.periodic[2])) for j in cols]
                a = [L[r, j] for j in cols]
                for (p, lap) in (((dx, dy) -> 1.0, 0.0), ((dx, dy) -> dx, 0.0), ((dx, dy) -> dy, 0.0), ((dx, dy) -> dx^2, 2.0), ((dx, dy) -> dx * dy, 0.0), ((dx, dy) -> dy^2, 2.0))
                    worst_q = max(worst_q, abs(sum(a[k] * p(offs[k]...) for k in eachindex(a)) - lap))
                end
                # KKT minimiser over the three-ring neighbourhood
                idx = sort!(collect(nbr))
                ofs = [(wrapped(pos(j)[1] - x0[1], mesh.extent[1], mesh.periodic[1]), wrapped(pos(j)[2] - x0[2], mesh.extent[2], mesh.periodic[2])) for j in idx]
                ell = max(maximum(abs(o[1]) for o in ofs), maximum(abs(o[2]) for o in ofs))
                C = zeros(6, length(idx)); D = zeros(length(idx))
                for (k, o) in enumerate(ofs)
                    dx, dy = o[1] / ell, o[2] / ell
                    C[:, k] = [1.0, dx, dy, dx^2 / 2, dx * dy, dy^2 / 2]
                    D[k] = dx^2 + dy^2 + 1 / 64
                end
                t = [0.0, 0, 0, 1 / ell^2, 0, 1 / ell^2]
                lam = (C .* (1 ./ D)') * C' \ t
                akkt = (1 ./ D) .* (C' * lam)
                coded = [L[r, j] for j in idx]
                worst_kkt = max(worst_kkt, maximum(abs, akkt .- coded) * ell^2)
            end
            # core rows outside the band: consistency with the Laplacian of a quadratic, wall data supplied by the connections
            #   rows without a wall connection of the NORMAL kind: P generic quadratic, tangential wall value P(wall point);
            #   rows with a normal wall connection (value 0): P = x_k (Lk - x_k), which vanishes at the normal walls.
            L_k = mesh.extent[axis]
            Pgen(x, y) = x^2 + 0.3x * y - 0.7y^2 + 0.2x - 0.1y + 1
            Pk(x, y) = axis == 1 ? x * (mesh.extent[1] - x) : y * (mesh.extent[2] - y)
            normal_rows = Set(c.degree_of_freedom for c in component.wall_connections if !c.tangential && AMR._include_wall_connection(NoSlip(), c))
            for (Pf, LAP, rowfilter) in ((Pgen, 2 - 1.4, r -> !(r in normal_rows)), (Pk, -2.0, r -> r in normal_rows))
                vals = [Pf(pos(i)...) for i in 1:n]
                applied = L * vals
                for c in component.wall_connections
                    AMR._include_wall_connection(NoSlip(), c) || continue
                    wallval = c.tangential ? Pf(c.point...) : 0.0
                    applied[c.degree_of_freedom] += c.conductance * wallval / V[c.degree_of_freedom]
                end
                for r in 1:n
                    (r in diffrows) && continue
                    rowfilter(r) || continue
                    x0 = pos(r)
                    ok_seam = true
                    for a in 1:2
                        mesh.periodic[a] || continue
                        (x0[a] < 3 * maximum(maximum(b.spacing) for b in mesh.blocks) || x0[a] > mesh.extent[a] - 3 * maximum(maximum(b.spacing) for b in mesh.blocks)) && (ok_seam = false)
                    end
                    ok_seam || continue
                    err = abs(applied[r] - LAP)
                    if err > 1.0e-9
                        ncore_bad += 1; worst_core = max(worst_core, err)
                        push!(bad_rows, (axis, r, pos(r), err, r in wall_dofs))
                    end
                end
            end
            # asymmetry
            scaled = spdiagm(0 => V) * L
            nasym += count(abs.(scaled .- transpose(scaled)) .> 1.0e-12 * maximum(abs, nonzeros(scaled)))
        end
        println(rpad(name, 34), "band rows ", nband, " (set ", set_ok ? "matches" : "DIFFERS", "), quadratic defect ", worst_q, ", KKT defect ", worst_kkt, ", asymmetric entries ", nasym, "; core rows with quadratic defect > 1e-9 outside band: ", ncore_bad, " (max ", worst_core, ")")
        isempty(bad_rows) || println("     first inconsistent core rows (axis, row, position, error, has wall connection): ", bad_rows[1:min(3, end)], "; without a wall connection: ", count(b -> !b[5], bad_rows))
        @test all(b -> b[5], bad_rows)          # a core row away from walls is quadratic exact; inconsistent core rows all touch a wall
        @test set_ok
        @test worst_q <= 1.0e-8
        @test worst_kkt <= 1.0e-8
    end
end

@testset "L5/L6: effect of the non-symmetric band" begin
    for (name, mesh) in fixtures[1:2]
        op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
        for (component, axis) in ((op.x, 1), (op.y, 2))
            V = component.volumes; L = Matrix(component.laplacian); S = Matrix(component.stiffness)
            A = -(Diagonal(V) * L)                      # integrated, should be S on core rows
            sym = (A + A') / 2
            ev_sym = eigvals(Symmetric(sym)); ev_core = eigvals(Symmetric(S))
            ev = eigvals(L)
            asym = norm(A - A') / norm(A)
            diff_core = norm(A - S) / norm(S)
            mom = maximum(abs, vec(sum(Diagonal(V) * L, dims = 1)))          # |1' V L| entries (column sums)
            println(
                rpad(name, 22), "axis ", axis, ": |A-A'|/|A| = ", round(asym, sigdigits = 3), ", |A-S|/|S| = ", round(diff_core, sigdigits = 3),
                ", lambda_min(sym A)/lambda_max(S) = ", round(ev_sym[1] / ev_core[end], sigdigits = 3), ", lambda_min(S)/lambda_max(S) = ", round(ev_core[1] / ev_core[end], sigdigits = 3),
                ", max Re eig(L) = ", round(maximum(real, ev), sigdigits = 3), ", max |col sum of V L| = ", round(mom, sigdigits = 3)
            )
            @test maximum(real, ev) <= 1.0e-9
        end
    end
end
