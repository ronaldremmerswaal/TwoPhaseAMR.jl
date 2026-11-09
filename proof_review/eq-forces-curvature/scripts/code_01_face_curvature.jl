# Code check 01: face curvature (eq:forces:curvature), extension (eq:forces:extension) and the unresolved set (eq:forces:unresolved).
#
# Label:  eq:forces:curvature, eq:forces:extension, eq:forces:unresolved
# Source: docs/techref/chapters/08_forces.tex:214-307; src/two_phase/forces.jl (_face_curvature, surface_tension!)
# The formulas are re-implemented here from the TeX (all leaf cells, nearest periodic image by explicit search over the 9 images, no block prefilter) and
# compared with the code on EVERY face of: drop on uniform / composite / cut meshes, the thin film of the test "Unresolved surface-tension curvature",
# a drop straddling the seam of a doubly periodic composite mesh, and the composite drop with 60 % of the cell curvatures removed (random, seeded; forces
# the extension to be used at many faces):
#   K1  face curvature: max relative difference to the code, and `nothing` exactly where the code returns `nothing`;
#   K2  classification (neighbours / extension / none) over all faces;
#   K3  convexity: kappa_f lies in [min, max] of the finite cell curvatures it averages;
#   K4  the unresolved set: surface_tension!'s count equals #{f : sigma > 0, (D psi)_f != 0, a_f > 0, no curvature}, for CSF and CGFM; those faces have zero jump;
#   K5  radius: the extension finds the nearest valid cell within 3 max(h) + h_c' -- number of unresolved faces that would acquire a curvature
#       from a cell beyond the radius (the limit of the rule, reported).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-forces-curvature/scripts/code_01_face_curvature.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
using .ForcesAudit
using TwoPhaseAMR
const FA = ForcesAudit

cmax(mesh, ref) = maximum(mesh.blocks[ref.block].spacing)

"""Independent evaluation; returns (kappa or nothing, kind, finite cell values used)."""
function curvature_independent(field, face)
    mesh = field.mesh
    total = weight = 0.0
    used = Float64[]
    for ref in (face.minus, face.plus)
        c = AMR._conditional_fraction(field, ref)
        k = AMR._get(field.curvature, ref)
        w = c * (1 - c)
        if w > 0 && isfinite(k)
            total += w * k; weight += w; push!(used, k)
        end
    end
    weight > 0 && return total / weight, :neighbours, used
    radius = 3 * max(cmax(mesh, face.minus), cmax(mesh, face.plus))
    nearest_beyond = Inf
    for ref in AMR._cell_references(mesh)
        c = AMR._conditional_fraction(field, ref)
        k = AMR._get(field.curvature, ref)
        w = c * (1 - c)
        (w > 0 && isfinite(k)) || continue
        h = cmax(mesh, ref)
        x = cellcenter(mesh, ref.block, ref.i, ref.j)
        d = Inf
        for sx in (mesh.periodic[1] ? (-1, 0, 1) : (0,)), sy in (mesh.periodic[2] ? (-1, 0, 1) : (0,))
            d = min(d, hypot(x[1] + sx * mesh.extent[1] - face.center[1], x[2] + sy * mesh.extent[2] - face.center[2]))
        end
        if d <= radius + h
            gw = w / max(d^2, (h / 4)^2)
            total += gw * k; weight += gw; push!(used, k)
        else
            nearest_beyond = min(nearest_beyond, d - h)
        end
    end
    weight > 0 && return total / weight, :extension, used
    return nothing, :none, used
end

"""Distance (centre to face, nearest image) to the nearest cell with a usable curvature, and the extension radius of the face."""
function nearest_valid(field, face)
    mesh = field.mesh
    radius = 3 * max(cmax(mesh, face.minus), cmax(mesh, face.plus))
    best = (Inf, 0.0)
    for ref in AMR._cell_references(mesh)
        c = AMR._conditional_fraction(field, ref)
        (c * (1 - c) > 0 && isfinite(AMR._get(field.curvature, ref))) || continue
        x = cellcenter(mesh, ref.block, ref.i, ref.j)
        d = hypot(x[1] - face.center[1], x[2] - face.center[2])
        d < best[1] && (best = (d, cmax(mesh, ref)))
    end
    return best, radius
end

function periodic_drop_field()
    mesh = BlockTreeMesh(;
        base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, true), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && key.i >= 2
    )
    wrapd(d, L) = d - L * round(d / L)
    levelset = (x, y) -> hypot(wrapd(x - 0.35, 8.0), wrapd(y - 4.0, 8.0)) - 1.7
    return InterfaceField(mesh, levelset; lipschitz = 1)
end

function film_field()
    mesh = BlockTreeMesh(block_size = (8, 8), periodic = (false, false))
    film = InterfaceField(mesh, (x, y) -> 1.0)
    values = interior(film.fraction, 1)
    fill!(values, 0)
    values[1, 2:6] .= (0.0, 0.167, 0.961, 0.994, 1.0)
    reconstruct!(film; force = true)
    return film
end

function thinned_field(rng)
    field = FA.drop_field(:amr)
    for ref in AMR._cell_references(field.mesh)
        c = AMR._conditional_fraction(field, ref)
        0 < c < 1 && rand(rng) < 0.6 && AMR._set!(field.curvature, NaN, ref)
    end
    return field
end

rng = MersenneTwister(20261117)
cases = Any[
    "drop uniform" => FA.drop_field(:uniform), "drop amr" => FA.drop_field(:amr), "drop cut" => FA.drop_field(:cut),
    "film" => film_field(), "periodic drop" => periodic_drop_field(), "amr, 60% removed" => thinned_field(rng),
]
rows = []
for (name, field) in cases
    mesh = field.mesh
    worst = 0.0; nothing_ok = true; convex = true
    kinds = Dict(:neighbours => 0, :extension => 0, :none => 0)
    for face in mesh.faces
        ind, kind, used = curvature_independent(field, face)
        act = AMR._face_curvature(field, face)
        kinds[kind] += 1
        if ind === nothing
            nothing_ok &= act === nothing
        else
            act === nothing && (nothing_ok = false; continue)
            worst = max(worst, abs(act - ind) / max(abs(ind), 1.0e-300))
            kind == :neighbours && (convex &= minimum(used) - 1.0e-12 <= act <= maximum(used) + 1.0e-12)
        end
    end
    # K4 unresolved count
    counts = Dict()
    for formulation in (CSF(), CGFM())
        sigma_op = isnothing(field.geometry) ? PoissonOperator(field; density = FA.DENSITY) : PoissonOperator(field, field.geometry; density = FA.DENSITY)
        force = FaceVelocity(mesh); ref = Ref(0)
        surface_tension!(force, field, sigma_op, EllipticWorkspace(sigma_op); sigma = 1.0, formulation, unresolved_curvature_faces = ref)
        J = copy(sigma_op.gradient_operator.pressure_jump)
        a = FA.apertures(field)
        expected = 0; zero_ok = true
        for (i, face) in pairs(mesh.faces)
            a[i] > 0 || continue
            cm, cp = AMR._conditional_fraction(field, face.minus), AMR._conditional_fraction(field, face.plus)
            pj = formulation isa CSF ? cp - cm : Float64((cp > 0.5) - (cm > 0.5))
            pj == 0 && continue
            if curvature_independent(field, face)[1] === nothing
                expected += 1; zero_ok &= J[i] == 0
            end
        end
        counts[nameof(typeof(formulation))] = (; code = ref[], expected, zero_ok)
        # sigma = 0 counts nothing
        z = Ref(-1); surface_tension!(force, field, sigma_op, EllipticWorkspace(sigma_op); sigma = 0.0, formulation, unresolved_curvature_faces = z)
        counts[nameof(typeof(formulation))] = (; code = ref[], expected, zero_ok, sigma0 = z[])
    end
    # K5: unresolved jump faces although some cell on the mesh has a curvature
    beyond = []
    for (i, face) in pairs(mesh.faces)
        cm, cp = AMR._conditional_fraction(field, face.minus), AMR._conditional_fraction(field, face.plus)
        cm == cp && continue
        curvature_independent(field, face)[1] === nothing || continue
        (d, h), radius = nearest_valid(field, face)
        isfinite(d) && push!(beyond, (round(d, digits = 2), round(radius + h, digits = 2)))
    end
    push!(rows, (; name, worst, nothing_ok, convex, kinds, counts, nfaces = length(mesh.faces), beyond))
    println(
        rpad(name, 18), " faces ", length(mesh.faces), ": K1 max rel. diff ", worst, ", `nothing` matches ", nothing_ok, "; K2 ", kinds, "; K3 convex ", convex,
        "; K4 unresolved (code / independent / zero jumps / sigma = 0): CSF ", counts[:CSF], " CGFM ", counts[:CGFM],
        "; K5 CSF-type jump faces unresolved although a curvature exists elsewhere (nearest distance, radius + h): ", beyond
    )
end
@testset "face curvature, extension, unresolved" begin
    for r in rows
        @test r.worst <= 1.0e-14
        @test r.nothing_ok
        @test r.convex
        for f in (:CSF, :CGFM)
            @test r.counts[f].code == r.counts[f].expected
            @test r.counts[f].zero_ok
            @test r.counts[f].sigma0 == 0
        end
    end
    film = rows[findfirst(r -> r.name == "film", rows)]
    @test film.kinds[:extension] > 0 || film.counts[:CGFM].expected > 0
    thin = rows[findfirst(r -> r.name == "amr, 60% removed", rows)]
    @test thin.kinds[:extension] > 10
    @test rows[findfirst(r -> r.name == "periodic drop", rows)].kinds[:neighbours] > 0
end
