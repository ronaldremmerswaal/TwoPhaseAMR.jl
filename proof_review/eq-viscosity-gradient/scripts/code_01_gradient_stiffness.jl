# Code check 01: eq:viscosity:gradient (two-point / widened derivative across a connection) and eq:viscosity:stiffness (integrated
# stiffness, symmetric positive semidefinite).
#
# Label:  eq:viscosity:gradient, eq:viscosity:stiffness (the stiffness folder repeats this script's include)
# Source: docs/techref/chapters/09_viscosity.tex:62-92; src/operators/staggered_diffusion.jl (_gradient_stencil, _diffusion_stiffness)
# Checks (periodic and walled composite meshes, random composite meshes, components 1 and 2):
#   G1  two-point difference exactly when the tangential offset |(x+ - x-)_s| <= 256 eps extent_s; otherwise the widened stencil;
#   G2  the widened stencil equals the KKT minimiser (independent algebra) of  sum a_f^2 (|x_f - x-|^2 + max(d,|dx_s|)^2/16)  subject
#       to  sum a_f p(x_f) = d_t p  for p in {1,x,y}, over the unknowns within two graph rings of the pair (independent BFS);
#   G3  exact for affine functions (checked on the stencil); quadratic exactness NOT claimed: size of the defect for p = x_t^2 etc.;
#   G4  S = sym( sum_g meas d a'a  +  sum_w cond e e' )  assembled independently equals the code's stiffness; S exactly symmetric,
#       PSD (lambda_min >= -1e-13 lambda_max), kernel = constants on the periodic meshes; a two-point connection contributes
#       meas (u+ - u-)^2/d;
#   G5  the stiffness is the sum of non-negative terms: u'S u = sum_g meas d (grad_g u)^2 + sum_w cond u_w^2 (random u).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-gradient/scripts/code_01_gradient_stiffness.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA

function component_graph(mesh, axis)
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

function rings(adjacency, start, depth)
    sel = Set(start); frontier = collect(start)
    for _ in 1:depth
        nxt = Int[]
        for i in frontier, j in adjacency[i]
            j in sel && continue
            push!(sel, j); push!(nxt, j)
        end
        frontier = nxt
    end
    return sel
end

# independent gradient stencil of a connection: (indices, coefficients, kind)
function independent_gradient(mesh, c, adjacency, l2g)
    xm = mesh.faces[l2g[c.minus]].center; xp = mesh.faces[l2g[c.plus]].center
    s = 3 - c.axis
    off = abs(wrapped(xp[s] - xm[s], mesh.extent[s], mesh.periodic[s]))
    if off <= 256eps() * mesh.extent[s]
        return [c.minus, c.plus], [-1 / c.distance, 1 / c.distance], :two_point
    end
    idx = sort!(collect(rings(adjacency, [c.minus, c.plus], 2)))
    L = max(c.distance, off)
    C = zeros(3, length(idx)); D = zeros(length(idx))
    for (k, j) in enumerate(idx)
        p = mesh.faces[l2g[j]].center
        dx = wrapped(p[1] - xm[1], mesh.extent[1], mesh.periodic[1]); dy = wrapped(p[2] - xm[2], mesh.extent[2], mesh.periodic[2])
        C[:, k] = [1.0, dx, dy]; D[k] = dx^2 + dy^2 + L^2 / 16
    end
    t = [0.0, c.axis == 1 ? 1.0 : 0.0, c.axis == 2 ? 1.0 : 0.0]
    lam = (C .* (1 ./ D)') * C' \ t
    return idx, (1 ./ D) .* (C' * lam), :widened
end

meshes = Any[]
push!(meshes, ("walled composite", BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false), max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= 2), NoSlip()))
push!(meshes, ("periodic composite", BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, true), max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= 2), NoSlip()))
let rng = MersenneTwister(1313)
    for (periodic, levels) in (((true, true), 2), ((false, false), 2), ((true, false), 2))
        push!(meshes, ("random periodic=$periodic L=$levels", MA.random_mesh(rng; periodic, levels, refine_probability = 0.5), NoSlip()))
    end
    push!(meshes, ("walled composite FreeSlip", meshes[1][2], FreeSlip()))
end

@testset "G1-G5 gradient and stiffness" begin
    rng = MersenneTwister(14)
    for (name, mesh, boundary) in meshes
        op = StaggeredDiffusionOperator(mesh; viscosity = 1.0, boundary)
        nwide = 0; ntwo = 0; nflip = 0; worst_stencil = 0.0; worst_aff = 0.0; worst_quad = 0.0; worst_S = 0.0; worst_sq = 0.0; worst_q = 0.0
        for (component, axis) in ((op.x, 1), (op.y, 2))
            conns, adjacency, l2g = component_graph(mesh, axis)
            n = length(l2g)
            rows = Int[]; cols = Int[]; vals = Float64[]
            u = randn(rng, n)
            quad_form = 0.0
            for c in conns
                idx, a, kind = independent_gradient(mesh, c, adjacency, l2g)
                kind == :widened ? (nwide += 1) : (ntwo += 1)
                # G3 affine exactness (positions relative to minus point)
                xm = mesh.faces[l2g[c.minus]].center
                aff = zeros(3)
                quad_t = 0.0
                for (k, j) in enumerate(idx)
                    p = mesh.faces[l2g[j]].center
                    dx = wrapped(p[1] - xm[1], mesh.extent[1], mesh.periodic[1]); dy = wrapped(p[2] - xm[2], mesh.extent[2], mesh.periodic[2])
                    aff .+= a[k] .* [1.0, dx, dy]
                    quad_t += a[k] * (c.axis == 1 ? dx^2 : dy^2)
                end
                target = [0.0, c.axis == 1 ? 1.0 : 0.0, c.axis == 2 ? 1.0 : 0.0]
                # the connection runs from `minus` to `plus`; at a periodic seam `plus` can lie to the left of `minus`, so the two-point stencil is the
                # derivative up to that orientation sign (the widened stencil is solved for +d_t and has no sign) (immaterial: only squares enter the stiffness)
                flipped = kind == :two_point && wrapped(mesh.faces[l2g[c.plus]].center[c.axis] - xm[c.axis], mesh.extent[c.axis], mesh.periodic[c.axis]) < 0
                flipped && (nflip += 1)
                worst_aff = max(worst_aff, maximum(abs, aff .- (flipped ? -target : target)))
                kind == :widened && (worst_quad = max(worst_quad, abs(quad_t)))
                w = c.measure * c.distance
                for (kr, ir) in enumerate(idx), (kc, ic) in enumerate(idx)
                    push!(rows, ir); push!(cols, ic); push!(vals, w * a[kr] * a[kc])
                end
                quad_form += w * dot(a, u[idx])^2
                kind == :two_point && @test w * dot(a, u[idx])^2 ≈ c.measure * (u[c.plus] - u[c.minus])^2 / c.distance rtol = 1.0e-12 atol = 1.0e-300
            end
            wall_q = 0.0
            for c in component.wall_connections
                AMR._include_wall_connection(boundary, c) || continue
                push!(rows, c.degree_of_freedom); push!(cols, c.degree_of_freedom); push!(vals, c.conductance)
                wall_q += c.conductance * u[c.degree_of_freedom]^2
            end
            S = sparse(rows, cols, vals, n, n); S = (S + S') / 2
            worst_S = max(worst_S, maximum(abs, S - component.stiffness) / maximum(abs, S))
            worst_sq = max(worst_sq, abs(dot(u, component.stiffness * u) - quad_form - wall_q) / dot(u, component.stiffness * u))
            ev = eigvals(Symmetric(Matrix(component.stiffness)))
            worst_q = min(worst_q, ev[1] / ev[end])
            @test component.stiffness == component.stiffness'
            if all(mesh.periodic)
                @test count(<(1.0e-10 * ev[end]), ev) == 1
            end
        end
        println(rpad(name, 34), "connections: ", ntwo, " two-point, ", nwide, " widened; affine defect ", worst_aff, "; quadratic defect of widened (x_t^2) up to ", worst_quad, "; stiffness defect ", worst_S, "; u'Su identity ", worst_sq, "; lambda_min/lambda_max ", worst_q)
        @test worst_aff <= 1.0e-10
        @test worst_S <= 1.0e-10
        @test worst_sq <= 1.0e-10
        @test worst_q >= -1.0e-12
    end
end

@testset "G2: widened stencil vs the code's stencil (through the stiffness rank-one terms)" begin
    # the code's stencil enters only through S; equality of S (G4 above) with independent stencils for every connection identifies them
    # up to the null space of the outer-product sum; additionally compare directly with the private routine.
    for (name, mesh, boundary) in meshes[1:2]
        for axis in 1:2
            conns, adjacency, l2g = component_graph(mesh, axis)
            worst = 0.0
            for c in conns
                idx_i, a_i, kind = independent_gradient(mesh, c, adjacency, l2g)
                idx_c, a_c = AMR._gradient_stencil(mesh, c, l2g, adjacency)
                @test sort(idx_c) == sort(idx_i)
                dict_i = Dict(zip(idx_i, a_i))
                worst = max(worst, maximum(abs(a_c[k] - dict_i[j]) for (k, j) in enumerate(idx_c)) * c.distance)
            end
            @test worst <= 1.0e-10
        end
    end
end
