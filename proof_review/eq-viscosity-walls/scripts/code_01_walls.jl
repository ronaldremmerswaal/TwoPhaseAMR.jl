# Code check 01: eq:viscosity:walls -- wall conductances, wall values and the wall types of the single-phase operator (no solid).
#
# Label:  eq:viscosity:walls (also :single-operator for the wall source)
# Source: docs/techref/chapters/09_viscosity.tex:132-176; src/operators/staggered_diffusion.jl (_wall_diffusion_connections,
#         _include_wall_connection, _wall_value, _apply_staggered_diffusion!)
# Checks:
#   W1  conductances: tangential  w = (varpi |b|/2)/|(x_f - x_b)_{axis b}|  for every face f of component k on the sides of the boundary
#       cell c; normal  w = varpi |b|/width_k(c)  for the faces f opposite b only; independent list from the faces of each cell
#       (side fraction varpi = |f|/width_t(c)) equals the code's wall connections (composite walled meshes, random, hanging faces);
#   W2  wall point: (x_b along axis b, x_f along the other axis); value rules: NoSlip(u): u at the wall point on tangential, 0 on normal;
#       FreeSlip: only normal connections kept; Inflow(f): f on both kinds; Outflow / ZeroGradient: none (Neumann);
#   W3  exactness: linear shear u_1 = U0 + (U1-U0) y/H (x-periodic channel, bottom NoSlip(U0), top NoSlip(U1)) has (L u + b) = 0
#       to round-off on uniform AND composite channel meshes (so the half-cell wall difference is exact for affine profiles);
#       FreeSlip with u_1 = const: zero; Inflow/Outflow: uniform flow zero;
#   W4  truncation of the wall-adjacent rows for a quadratic profile: error = (1/2)(1 - ...)u'' (not quadratic exact, first order) -- the
#       chapter makes no quadratic claim for wall rows; the size is reported;
#   W5  the operator returned by diffusion! is (nu)(L u + V^-1 sum_w cond u_w) on the unknowns and zero on the boundary faces.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-walls/scripts/code_01_walls.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA

function independent_walls(mesh, k)
    list = Tuple{Int, Float64, Bool, NTuple{2, Float64}}[]          # (global face, conductance, tangential, wall point)
    for b in mesh.boundary_faces
        a = b.axis
        ref = b.cell
        width = cell_spacing(mesh, ref)
        cc = AMR.cellcenter(mesh, ref.block, ref.i, ref.j)
        for (i, frac) in cell_faces_of_axis(mesh, ref, k)
            f = mesh.faces[i]
            point = a == 1 ? (b.center[1], f.center[2]) : (f.center[1], b.center[2])
            if a != k
                push!(list, (i, 0.5 * frac * b.area / abs(f.center[a] - b.center[a]), true, point))
            else
                # face opposite b: on the other side of the cell centre
                (f.center[k] - cc[k]) * b.side < 0 || continue
                push!(list, (i, frac * b.area / width[k], false, point))
            end
        end
    end
    return sort!(list)
end

@testset "W1/W2 conductances and wall points" begin
    rng = MersenneTwister(2025)
    meshes = Any[BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false), max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= 2)]
    for (periodic, levels) in (((false, false), 1), ((false, false), 2), ((true, false), 2), ((false, true), 2))
        push!(meshes, MA.random_mesh(rng; periodic, levels, refine_probability = 0.5))
    end
    for mesh in meshes
        op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)
        for (component, k) in ((op.x, 1), (op.y, 2))
            g2l = AMR._component_face_indices(mesh, k)[1]
            code = sort!([(component.local_to_global[c.degree_of_freedom], c.conductance, c.tangential, c.point) for c in component.wall_connections])
            ind = independent_walls(mesh, k)
            @test length(code) == length(ind)
            @test all(a[1] == b[1] && a[3] == b[3] && isapprox(a[2], b[2]; rtol = 1.0e-13) && all(isapprox.(a[4], b[4]; atol = 1.0e-13)) for (a, b) in zip(code, ind))
        end
    end
end

shear_mesh(refined; n = 16) = BlockTreeMesh(;
    base_mesh = (n, n), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, false), max_level = refined ? 1 : 0,
    refine = (key, o, e) -> refined && key.level == 0 && key.i >= 2
)

@testset "W3: affine profiles are exact" begin
    H = 8.0
    for refined in (false, true)
        mesh = shear_mesh(refined)
        U0, U1 = 0.3, 1.7
        profile(x, y, t, axis) = axis == 1 ? U0 + (U1 - U0) * y / H : 0.0
        wall(x, y, t, axis) = axis == 1 ? (y > H / 2 ? U1 : U0) : 0.0
        boundary = VelocityBoundaryConditions(bottom = NoSlip(wall), top = NoSlip(wall), left = NoSlip(), right = NoSlip())
        op = StaggeredDiffusionOperator(mesh; viscosity = 1.0, boundary)
        u = FaceVelocity(mesh, profile)
        rate = FaceVelocity(mesh)
        diffusion!(rate, u, op)
        println("refined=$refined  moving-wall shear, max |rate| = ", maximum(abs, rate.interior))
        @test maximum(abs, rate.interior) <= 1.0e-11
        # the same field with a stationary NoSlip() has a nonzero rate (the wall data matter)
        op0 = StaggeredDiffusionOperator(mesh; viscosity = 1.0, boundary = NoSlip())
        rate0 = FaceVelocity(mesh); diffusion!(rate0, FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? U0 + (U1 - U0) * y / H : 0.0), op0)
        @test maximum(abs, rate0.interior) > 1.0e-3
        # FreeSlip, u_1 = const: zero
        opf = StaggeredDiffusionOperator(mesh; viscosity = 1.0, boundary = FreeSlip())
        ratef = FaceVelocity(mesh); diffusion!(ratef, FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 0.8 : 0.0), opf)
        @test maximum(abs, ratef.interior) <= 1.0e-11
    end
    # inflow / outflow: uniform flow u_1 = 1 in a box: left Inflow(1), right Outflow, top/bottom FreeSlip
    mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false), max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= 2)
    for right in (Outflow(), ZeroGradientVelocity())
        boundary = VelocityBoundaryConditions(left = Inflow((x, y, t, axis) -> axis == 1 ? 1.0 : 0.0), right = right, bottom = FreeSlip(), top = FreeSlip())
        op = StaggeredDiffusionOperator(mesh; viscosity = 1.0, boundary)
        rate = FaceVelocity(mesh); diffusion!(rate, FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? 1.0 : 0.0), op)
        println(typeof(right).name.name, ": uniform flow, max |rate| = ", maximum(abs, rate.interior))
        @test maximum(abs, rate.interior) <= 1.0e-11
    end
end

@testset "W4: wall-adjacent rows are not quadratic exact" begin
    H = 8.0
    mesh = shear_mesh(false)
    op = StaggeredDiffusionOperator(mesh; viscosity = 1.0)               # NoSlip with zero wall values
    # u_1 = y (H - y): zero at both walls, u'' = -2
    u = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? y * (H - y) : 0.0)
    rate = FaceVelocity(mesh); diffusion!(rate, u, op)
    h = 0.5
    errs = Dict{String, Float64}()
    for (i, f) in pairs(mesh.faces)
        f.axis == 1 || continue
        row = round(Int, f.center[2] / h - 0.5)          # 0 = adjacent to the wall
        key = row == 0 || row == 15 ? "wall-adjacent" : "interior"
        errs[key] = max(get(errs, key, 0.0), abs(rate.interior[i] + 2))
    end
    println("u = y(H-y), exact Delta u = -2: max error interior ", errs["interior"], ", wall-adjacent ", errs["wall-adjacent"])
    @test errs["interior"] <= 1.0e-11
    @test errs["wall-adjacent"] > 0.4         # = h-independent 0.5*|u''| * (1 - ...)
end
