# Code check 01: eq:viscosity:cut -- the single-phase operator with a solid (GeometryField): embedded conductances, covered unknowns,
# wall value zero (issue #75), wall types, clipped domain-wall measures, no corrected band.
#
# Label:  eq:viscosity:cut (also :walls for the clipped domain-wall measure, :stiffness for the cut-cell stiffness)
# Source: docs/techref/chapters/09_viscosity.tex:178-205; src/embedded/operators.jl (_embedded_wall_connections, _wall_connection_measure,
#         _cut_cell_diffusion_component, StaggeredDiffusionOperator(geometry; ...)); src/operators/staggered_diffusion.jl (_wall_value)
# Checks (fixture: walled composite mesh with the sloped solid y < 0.9 + 0.1 x, plus random discs):
#   U1  embedded wall rows: for every cut cell c (PLIC segment length l, midpoint m) and every face f of component k on its sides:
#       w = (1/2) varpi l / max(|x_f - m|_2, h_min/4); covered faces (open dual volume 0): w = 1; independent list (segment from the package,
#       everything else recomputed) equals the code's connections with boundary_axis = 0;
#   U2  #75: with NoSlip(lid) (moving wall data) the rate of the zero field has NO contribution from embedded connections, while the domain
#       wall data act: embedded connections are marked by boundary_axis = 0 and _wall_value returns 0 for them;
#   U3  wall types: NoSlip keeps cut + covered; FreeSlip keeps covered only (and the domain normal connections); VelocityBoundaryConditions
#       keeps the cut connections REGARDLESS of the sides' types (reported: differs from the chapter's "(omitted for FreeSlip)");
#   U4  no band: laplacian == -V^-1 S in every row (also at level transitions) -> consistency at transitions (K: truncation of the
#       operator on a smooth field away from the solid, rows at the refinement interface vs elsewhere);
#   U5  covered unknowns: isolated in the connection graph (no connection), row  -1 * u  (V = 1, w = 1): the step relaxes them toward 0
#       by 1/(1 + dt nu/2) only -- not an exact constraint;
#   U6  domain-wall measures are clipped by the open fraction: a solid that covers a whole wall segment removes its connections.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-cut/scripts/code_01_cut.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const MA = ViscosityAudit.MA

function embedded_connections(component)
    return [c for c in component.wall_connections if iszero(c.boundary_axis)]
end

@testset "U1: embedded conductances" begin
    rng = MersenneTwister(2323)
    geoms = Any[fixture_cut().geometry]
    for periodic in ((false, false), (true, true)), levels in (1, 2)
        mesh = MA.random_mesh(rng; periodic, levels, refine_probability = 0.5)
        push!(geoms, GeometryField(mesh, MA.random_solid(rng, mesh.extent; kind = :disc); lipschitz = 2))
    end
    nbuilt = 0; nthrown = 0
    for geometry in geoms
        mesh = geometry.mesh
        field = InterfaceField(mesh, (x, y) -> 1.0; geometry, lipschitz = 1)
        op = try
            StaggeredDiffusionOperator(geometry; viscosity = 1.0)
        catch err
            err isa ArgumentError && occursin("degenerate AMR dual diffusion connection", err.msg) || rethrow()
            nthrown += 1                       # pre-existing defect D1 (see the audit and constructor_defect.jl)
            continue
        end
        nbuilt += 1
        for (component, k) in ((op.x, 1), (op.y, 2))
            code = sort!([(component.local_to_global[c.degree_of_freedom], c.conductance, c.tangential) for c in embedded_connections(component)])
            # independent: cut rows of this axis only
            rows = Tuple{Int, Float64, Int, Symbol}[]
            for ref in cells(mesh)
                seg = AMR._local_plic_segment(geometry.solid, ref)
                seg === nothing && continue
                center = AMR.cellcenter(mesh, ref.block, ref.i, ref.j); spacing = cell_spacing(mesh, ref)
                p0 = AMR._local_to_physical(Float64, center, spacing, seg[1]); p1 = AMR._local_to_physical(Float64, center, spacing, seg[2])
                mid = ((p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2); len = hypot(p1[1] - p0[1], p1[2] - p0[2])
                len > 0 || continue
                for (i, frac) in cell_faces_of_axis(mesh, ref, k)
                    f = mesh.faces[i]
                    d = max(hypot(wrapped(f.center[1] - mid[1], mesh.extent[1], mesh.periodic[1]), wrapped(f.center[2] - mid[2], mesh.extent[2], mesh.periodic[2])), 0.25 * minimum(spacing))
                    push!(rows, (i, 0.5 * frac * len / d, 0, :cut))
                end
            end
            for (i, f) in pairs(mesh.faces)
                f.axis == k || continue
                geometry.face_dual_volumes[i] > 0 && continue
                push!(rows, (i, 1.0, 0, :covered))
            end
            ind = sort!([(i, w, kind == :cut) for (i, w, _, kind) in rows])
            @test length(code) == length(ind)
            @test all(a[1] == b[1] && a[3] == b[3] && isapprox(a[2], b[2]; rtol = 1.0e-13) for (a, b) in zip(code, ind))
        end
    end
    println("U1: ", nbuilt, " geometries built, ", nthrown, " random geometries rejected by the constructor (defect D1)")
    @test nbuilt >= 1
end

@testset "U2: #75 embedded walls do not move" begin
    # uniform mesh: a disc across a level transition cannot be built by the single-phase constructor (defect D1)
    mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false))
    geometry = GeometryField(mesh, (x, y) -> hypot(x - 4.0, y - 3.2) - 1.6; lipschitz = 1)
    lid = (x, y, time, axis) -> axis == 1 ? 1.0 : 0.0
    zero_field = FaceVelocity(mesh)
    for boundary in (NoSlip(lid), VelocityBoundaryConditions(top = NoSlip(lid)), NoSlip())
        op = StaggeredDiffusionOperator(geometry; viscosity = 1.0, boundary)
        rate = FaceVelocity(mesh); diffusion!(rate, zero_field, op)
        # unknowns that have an embedded connection only (no domain-wall connection): rate must be 0
        for (component, k) in ((op.x, 1), (op.y, 2))
            domain = Set(c.degree_of_freedom for c in component.wall_connections if !iszero(c.boundary_axis))
            emb = Set(c.degree_of_freedom for c in embedded_connections(component))
            only_emb = setdiff(emb, domain)
            @test !isempty(only_emb)
            @test all(rate.interior[component.local_to_global[i]] == 0 for i in only_emb)
            for c in embedded_connections(component)
                @test AMR._wall_value(boundary, c, 0.0, k, Float64) == 0
            end
        end
        println(typeof(boundary).name.name, ": max |rate| of the zero field = ", maximum(abs, rate.interior), " (lid data only at domain-wall unknowns)")
    end
    op = StaggeredDiffusionOperator(geometry; viscosity = 1.0, boundary = NoSlip(lid))
    rate = FaceVelocity(mesh); diffusion!(rate, zero_field, op)
    @test maximum(abs, rate.interior) > 0       # the top/bottom lid data still act
end

@testset "U3: wall types" begin
    mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, true))
    geometry = GeometryField(mesh, (x, y) -> hypot(x - 4.0, y - 3.2) - 1.6; lipschitz = 1)
    ops = Dict(
        name => StaggeredDiffusionOperator(geometry; viscosity = 1.0, boundary = b) for (name, b) in
            (("NoSlip", NoSlip()), ("FreeSlip", FreeSlip()), ("VBC(FreeSlip x4)", VelocityBoundaryConditions(left = FreeSlip(), right = FreeSlip(), bottom = FreeSlip(), top = FreeSlip())))
    )
    nnz_stiff = Dict(name => nnz(op.x.stiffness) for (name, op) in ops)
    S(name) = Matrix(ops[name].x.stiffness)
    println(
        "embedded-solid stiffness differences (component 1): |S_NoSlip - S_FreeSlip| = ", norm(S("NoSlip") - S("FreeSlip")),
        ", |S_VBC(FreeSlip) - S_FreeSlip| = ", norm(S("VBC(FreeSlip x4)") - S("FreeSlip")), ", |S_VBC(FreeSlip) - S_NoSlip| = ", norm(S("VBC(FreeSlip x4)") - S("NoSlip"))
    )
    @test norm(S("NoSlip") - S("FreeSlip")) > 1.0e-3
    @test norm(S("VBC(FreeSlip x4)") - S("NoSlip")) <= 1.0e-12 * norm(S("NoSlip"))     # VBC keeps the embedded no-slip regardless of the sides
end

@testset "U4/U5: no band, covered unknowns" begin
    # (a) composite mesh, grid-aligned solid y < 1 (builds): the corrected band of the no-solid operator is absent although the mesh has a
    #     refinement interface; the no-solid operator on the same mesh does have band rows
    meshc = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false), max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= 2)
    opc = StaggeredDiffusionOperator(GeometryField(meshc, (x, y) -> y - 1.0; lipschitz = 1); viscosity = 1.0)
    op0 = StaggeredDiffusionOperator(meshc; viscosity = 1.0)
    nband0 = 0
    for (cc, c0) in ((opc.x, op0.x), (opc.y, op0.y))
        @test maximum(abs, cc.laplacian - (-spdiagm(0 => inv.(cc.volumes)) * cc.stiffness)) == 0
        base0 = -spdiagm(0 => inv.(c0.volumes)) * c0.stiffness
        nband0 += count(r -> maximum(abs, Vector(c0.laplacian[r, :]) .- Vector(base0[r, :])) > 1.0e-12, 1:size(base0, 1))
    end
    println("composite mesh: no-solid operator has ", nband0, " band rows, the cut-cell operator none")
    @test nband0 > 0
    # (b) covered unknowns on a uniform mesh with a disc
    mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false))
    geometry = GeometryField(mesh, (x, y) -> hypot(x - 4.0, y - 3.2) - 1.6; lipschitz = 1)
    op = StaggeredDiffusionOperator(geometry; viscosity = 1.0)
    for (component, k) in ((op.x, 1), (op.y, 2))
        covered = [i for i in eachindex(component.local_to_global) if geometry.face_dual_volumes[component.local_to_global[i]] == 0]
        isempty(covered) && continue
        row_nnz = [count(!iszero, component.laplacian[i, :]) for i in covered]
        @test all(==(1), row_nnz)
        @test all(i -> component.laplacian[i, i] == -1.0 && component.volumes[i] == 1.0, covered)
        println("component ", k, ": ", length(covered), " covered unknowns, each an isolated row (-1)")
    end
end

@testset "U4b: consistency at refinement interfaces without the band (no solid in the region)" begin
    # smooth field zero at the domain walls; rows far from the solid (y > 3.5): refinement-interface rows vs the rest
    L = 8.0
    for n in (16, 32, 64)
        mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (4, 4), extent = (L, L), periodic = (false, false), max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= div(n, 8))
        geometry = GeometryField(mesh, (x, y) -> y - 1.0 - 0.0 * x; lipschitz = 1)       # solid y < 1 (grid-aligned for n=16.. cut cells off the rows tested)
        op = StaggeredDiffusionOperator(geometry; viscosity = 1.0)
        ustar(x, y) = sin(pi * x / L) * sin(pi * y / L)
        lap(x, y) = -2 * (pi / L)^2 * ustar(x, y)
        err_if = 0.0; err_other = 0.0
        for (component, k) in ((op.x, 1), (op.y, 2))
            pts = [mesh.faces[g].center for g in component.local_to_global]
            applied = component.laplacian * [ustar(p...) for p in pts]
            wall_dofs = Set(c.degree_of_freedom for c in component.wall_connections)
            conns = AMR._dual_diffusion_connections(mesh, AMR.StaggeredAdvectionWorkspace(FaceField{k}(mesh), geometry).schedule, component.local_to_global)
            ifdofs = Set{Int}(); for c in conns
                c.refinement_interface && (push!(ifdofs, c.minus); push!(ifdofs, c.plus))
            end
            for r in eachindex(pts)
                (r in wall_dofs) && continue
                pts[r][2] > 2.5 || continue
                e = abs(applied[r] - lap(pts[r]...))
                if r in ifdofs
                    err_if = max(err_if, e)
                else
                    err_other = max(err_other, e)
                end
            end
        end
        println("n=", n, ": max truncation error at refinement-interface rows ", err_if, ", other interior rows ", err_other)
    end
end

@testset "U6: clipped domain-wall measure" begin
    mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false))
    # solid covers the whole bottom wall (y < 1.3) and part of the left wall
    geometry = GeometryField(mesh, (x, y) -> y - 1.3; lipschitz = 1)
    op = StaggeredDiffusionOperator(geometry; viscosity = 1.0)
    bottom = 0; top = 0
    for component in (op.x, op.y), c in component.wall_connections
        iszero(c.boundary_axis) && continue
        c.boundary_axis == 2 && c.boundary_side < 0 && (bottom += 1)
        c.boundary_axis == 2 && c.boundary_side > 0 && (top += 1)
    end
    println("bottom wall (covered by the solid): ", bottom, " connections, top wall: ", top)
    @test bottom == 0 && top > 0
end

@testset "U4c: solution-level order of the cut-cell operator (no band) on a composite mesh, component 1" begin
    # Grid-aligned solid y < 1 carries NO embedded wall row (defect D2): there the operator imposes zero stress (Neumann); so take
    #   u* = sin(pi x/L) cos(pi (y-1)/(2(L-1)))  : d_y u* = 0 at y = 1, u* = 0 at x = 0, L and y = L,
    # (I - L) u = (I - Delta) u*  with Delta u* = -[(pi/L)^2 + (pi/(2(L-1)))^2] u*.  The no-solid operator on the same mesh is run with the
    # same field on y >= 1 (not a wall there: only rows y >= 2.5 are compared, where the Neumann face does not matter for the error order).
    L = 8.0
    sol(x, y) = sin(pi * x / L) * cos(pi * (y - 1) / (2 * (L - 1)))
    lapfac = -((pi / L)^2 + (pi / (2 * (L - 1)))^2)
    prev_cut = nothing
    for n in (16, 32, 64)
        mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (4, 4), extent = (L, L), periodic = (false, false), max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= div(n, 8))
        geometry = GeometryField(mesh, (x, y) -> y - 1.0; lipschitz = 1)
        opc = StaggeredDiffusionOperator(geometry; viscosity = 1.0)
        c = opc.x
        pts = [mesh.faces[g].center for g in c.local_to_global]
        open_ = [geometry.face_dual_volumes[g] > 0 for g in c.local_to_global]
        us = [open_[i] ? sol(pts[i]...) : 0.0 for i in eachindex(pts)]
        f = us .- lapfac .* us
        u = (sparse(I, length(us), length(us)) - c.laplacian) \ f
        e_cut = maximum(abs, (u .- us)[open_])
        println("n=", n, ": cut operator (no band) max solution error ", e_cut, prev_cut === nothing ? "" : "  order $(round(log2(prev_cut / e_cut), digits = 2))")
        prev_cut = e_cut
    end
end
