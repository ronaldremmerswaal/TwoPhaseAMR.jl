# Code check 01: patch moment tensor of the real package on SQUARE-cell meshes.
#
# Label:  eq:setting:patch-metric
# Claim:  03_discrete_setting.tex:304-305 (symmetry of Y_v' N_v on every patch of square cells),
#         and the consequence R_v Y_v = N_v / affine exactness (03_discrete_setting.tex:261-268).
# Code:   src/operators/gradient.jl:74-137 (_wrapped_delta, _face_displacement, _vertex_key,
#         _vertex_patches), 139-184 (_local_inverse_metric), 214-310 (assembly).
# Method: for every vertex patch of every mesh (walls / periodic seams / block sizes /
#         levels / seeds / exactly dyadic and non-dyadic cell sizes):
#           * the patch rebuilt from the touching cells with the exact integer model (model.jl)
#             equals the package's patch (faces, areas, displacements, T);
#           * on every patch whose faces join cells of side ratio <= 2 ("graded patch"):
#             max |T12 - T21| / max|T| (T = Y'N from the package's own MetricFace data),
#             R_v Y_v - N_v for the package's _local_inverse_metric, Cartesian patch = diagonal;
#           * wall vertices have <= 2 face segments (diagonal closure, no tensor);
#           * the assembled operator equals the sum of the local blocks; affine pressure jumps
#             are mapped exactly to the face-normal gradient on every face whose two endpoint
#             patches are complete and graded (this includes faces across periodic seams).
#         The package is supposed to produce only 2:1-graded meshes. For some floating-point
#         extents it does not (see code_04); such meshes are NOT excluded: the lemma is checked
#         on their graded patches, and the ungraded patches are counted.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-setting-patch-metric/scripts/code_01_square_meshes.jl
#         (from the repository root, with the package environment of the repository)

import TwoPhaseAMR   # not `using`: the package also exports Cell and Face (names used by model.jl)
using LinearAlgebra: eigvals, Diagonal
using Test
const AMR = TwoPhaseAMR
include(joinpath(@__DIR__, "code_helpers.jl"))

"""Maxima over the patches of one mesh; class counts are accumulated in the two tables."""
function audit_mesh(mesh; seam_classes, interior_classes)
    patches = AMR._vertex_patches(mesh, Float64)

    # cell-level grading across every face (the hypothesis of the lemma)
    max_face_ratio = 1.0
    for face in mesh.faces
        a = mesh.blocks[face.minus.block].spacing[face.axis]
        b = mesh.blocks[face.plus.block].spacing[face.axis]
        max_face_ratio = max(max_face_ratio, a / b, b / a)
    end

    max_asym = max_ideal_mismatch = max_residual = max_cartesian_mismatch = 0.0
    max_wall_faces = 0
    nonwall_small = 0
    n_complete = n_ungraded = n_ungraded_asym = 0
    good = Set{Tuple{Int, Int}}()                 # complete AND graded patches
    for (key, faces) in patches
        if AMR._physical_vertex(mesh, key)
            max_wall_faces = max(max_wall_faces, length(faces))
            continue
        end
        if length(faces) <= 2
            nonwall_small += 1
            continue
        end
        n_complete += 1
        Y, N = moment_matrices(faces)
        T = transpose(Y) * N
        scale = maximum(abs, T)
        asym = abs(T[1, 2] - T[2, 1]) / scale
        R = AMR._local_inverse_metric(copy(faces))

        # independent integer model of the same vertex, built from the touching cells
        cells, hx, hy = lattice_cells(mesh, key, faces)
        ideal = faces_at_vertex(cells, 0, 0)
        @assert length(ideal) == length(faces) "face count differs from the model"
        Tideal = physical_moment(moment_integers(ideal), hx, hy)
        max_ideal_mismatch = max(max_ideal_mismatch, maximum(abs, T - Tideal) / scale)
        for (a, b) in zip(face_rows(faces), model_rows(ideal, hx, hy))
            @assert a[1] == b[1] && all(isapprox.(a[2:4], b[2:4]; atol = 1.0e-12 * sqrt(scale), rtol = 1.0e-12))
        end
        if !face_graded(ideal)
            n_ungraded += 1
            asym > 1.0e-12 && (n_ungraded_asym += 1)
            continue
        end
        push!(good, key)
        max_asym = max(max_asym, asym)
        max_residual = max(max_residual, maximum(abs, R * Y - N))
        seam = (mesh.periodic[1] && key[1] == 0) || (mesh.periodic[2] && key[2] == 0)
        table = seam ? seam_classes : interior_classes
        label = patch_label(cells)
        table[label] = get(table, label, 0) + 1
        if label == (1, 1, 1, 1)
            # regular Cartesian patch: the sum of eq:setting:patch-metric is the diagonal 1/(2 dualvol)
            diag_ref = [inv(2 * mesh.face_dual_volumes[f.index]) for f in faces]
            max_cartesian_mismatch = max(max_cartesian_mismatch, maximum(abs, R - Diagonal(diag_ref)) / maximum(diag_ref))
        end
    end

    # assembled operator = sum of local blocks
    operator = AMR.GradientOperator(mesh)
    K = metric_matrix(operator)
    Kref = reference_metric(mesh, patches)
    max_operator_mismatch = maximum(abs, K - Kref) / maximum(abs, K)
    @assert maximum(abs, K - transpose(K)) <= 1.0e-13 * maximum(abs, K)
    exact_face(face) = all(e -> AMR._vertex_key(mesh, face, e) in good, 1:2)

    # affine jumps: K * (area * displacement[c]) = e_axis(c) on faces with two complete graded endpoints
    jump = zeros(length(mesh.faces)); result = similar(jump)
    max_affine_jump = 0.0
    n_exact_faces = 0
    for c in 1:2
        for (index, face) in pairs(mesh.faces)
            jump[index] = face.area * AMR._face_displacement(mesh, face)[c]
        end
        AMR._mul_inverse_metric!(result, operator.inverse_metric, jump)
        for (index, face) in pairs(mesh.faces)
            exact_face(face) || continue
            max_affine_jump = max(max_affine_jump, abs(result[index] - (face.axis == c ? 1.0 : 0.0)))
            c == 1 && (n_exact_faces += 1)
        end
    end

    # affine pressure (in the non-periodic directions), actual cell values
    a = mesh.periodic[1] ? 0.0 : 0.3
    b = mesh.periodic[2] ? 0.0 : -0.7
    pressure = AMR.CellField(mesh, (x, y) -> a * x + b * y)
    gradient = AMR.FaceVelocity(mesh)
    AMR.gradient!(gradient, pressure, operator)
    max_affine_pressure = 0.0
    for (index, face) in pairs(mesh.faces)
        exact_face(face) || continue
        max_affine_pressure = max(max_affine_pressure, abs(gradient.interior[index] - (face.axis == 1 ? a : b)))
    end
    return (; n_patches = length(patches), n_complete, n_ungraded, n_ungraded_asym, nonwall_small,
        max_wall_faces, max_face_ratio, max_asym, max_cartesian_mismatch,
        max_ideal_mismatch, max_residual, max_operator_mismatch, max_affine_jump,
        max_affine_pressure, n_exact_faces)
end

# ------------------------------------------------------------------------------------------
# the mesh families
# ------------------------------------------------------------------------------------------

function designed_meshes()
    meshes = Pair{String, Any}[]
    function by_roots(name, selector; periodic, layout = (4, 4), block_size = (2, 2))
        base = (layout[1] * block_size[1], layout[2] * block_size[2])
        mesh = AMR.BlockTreeMesh(;
            base_mesh = base, block_size = block_size, extent = (Float64(base[1]), Float64(base[2])),
            periodic = periodic, max_level = 1,
            refine = (key, origin, ext) -> key.level == 0 && selector(key.i, key.j),
        )
        push!(meshes, name => mesh)
    end
    for periodic in ((false, false), (true, true))
        tag = periodic == (false, false) ? "walls" : "periodic"
        by_roots("single refined block, $tag", (i, j) -> (i, j) == (1, 1); periodic)
        by_roots("L-shaped refined region, $tag", (i, j) -> (i, j) in ((1, 1), (2, 1), (1, 2)); periodic)
        by_roots("checkerboard, $tag", (i, j) -> iseven(i + j); periodic)
        by_roots("refined band, $tag", (i, j) -> j in (1, 2); periodic)
    end
    # refinement touching the periodic seams (x seam: i = 3 | 0, y seam: j = 3 | 0)
    by_roots("seam: single block at x seam", (i, j) -> (i, j) == (0, 1); periodic = (true, true))
    by_roots("seam: single block at y seam", (i, j) -> (i, j) == (1, 0); periodic = (true, true))
    by_roots("seam: L across x seam", (i, j) -> (i, j) in ((3, 1), (0, 1), (0, 2)); periodic = (true, true))
    by_roots("seam: L across y seam", (i, j) -> (i, j) in ((1, 3), (1, 0), (2, 0)); periodic = (true, true))
    by_roots("seam: L at the corner seam", (i, j) -> (i, j) in ((3, 3), (0, 3), (3, 0)); periodic = (true, true))
    by_roots("seam: checkerboard, x only periodic", (i, j) -> iseven(i + j); periodic = (true, false))
    by_roots("seam: refined column across x seam", (i, j) -> i in (3, 0); periodic = (true, false))
    # the meshes of docs/techref/figures/generate.jl (affine_mesh with square cells, fixture())
    refined = Set([AMR.BlockKey(0, 1, 1), AMR.BlockKey(0, 2, 1), AMR.BlockKey(0, 1, 2), AMR.BlockKey(1, 3, 3)])
    push!(meshes, "figures affine_mesh (square)" => AMR.BlockTreeMesh(;
        base_mesh = (8, 8), block_size = (2, 2), extent = (1.0, 1.0), periodic = (false, false),
        max_level = 2, refine = (key, origin, ext) -> key in refined,
    ))
    push!(meshes, "figures fixture()" => AMR.BlockTreeMesh(;
        base_mesh = (6, 2), block_size = (2, 2), extent = (12.0, 4.0), periodic = (true, false),
        max_level = 2, refine = (key, origin, ext) -> key == AMR.BlockKey(0, 2, 0) || key == AMR.BlockKey(1, 4, 1),
    ))
    return meshes
end

@testset "code_01: square-cell meshes" begin
    seam_classes = Dict{Any, Int}()
    interior_classes = Dict{Any, Int}()
    rows = Any[]
    function record(name, mesh; require_graded)
        r = audit_mesh(mesh; seam_classes, interior_classes)
        push!(rows, (name, r))
        @testset "$name" begin
            require_graded && @test r.max_face_ratio <= 2 + 1.0e-12   # the hypothesis (2:1 grading)
            @test r.nonwall_small == 0                 # every non-wall vertex has >= 3 faces
            @test r.max_wall_faces <= 2                # wall vertices never carry a tensor
            @test r.max_asym <= 1.0e-13                # symmetric moment tensor on every graded patch
            @test r.max_ideal_mismatch <= 1.0e-12      # package patch == integer model
            @test r.max_residual <= 1.0e-12            # R_v Y_v = N_v
            @test r.max_cartesian_mismatch <= 1.0e-12  # regular patch: diagonal of eq:setting:wall-metric
            @test r.max_operator_mismatch <= 1.0e-12   # assembly = sum of local blocks
            @test r.max_affine_jump <= 1.0e-12
            @test r.max_affine_pressure <= 1.0e-12
        end
        return r
    end

    for (name, mesh) in designed_meshes()
        record(name, mesh; require_graded = true)
    end

    seed = 0
    for periodic in ((false, false), (true, false), (false, true), (true, true)),
            (block_size, roots) in (((2, 2), (3, 3)), ((4, 4), (2, 2)), ((2, 4), (3, 2)), ((4, 2), (2, 3)), ((6, 6), (2, 2))),
            max_level in (2, 3), h0 in (1.0, 0.37)
        block_size == (6, 6) && max_level == 3 && continue   # run-time economy
        seed += 1
        mesh = random_mesh(seed; block_size, roots, max_level, periodic, h0)
        # h0 = 1.0 (dyadic cell sizes) must always be graded; for h0 = 0.37 the package's own
        # balancing can miss contacts (code_04), so grading is recorded, not required
        record("random #$seed periodic=$periodic block=$block_size L=$max_level h0=$h0", mesh; require_graded = h0 == 1.0)
    end

    println("\nmeshes audited: ", length(rows),
        ",  vertex patches: ", sum(r.n_patches for (_, r) in rows),
        ",  tensor patches: ", sum(r.n_complete for (_, r) in rows),
        ",  of which ungraded (4:1 contact): ", sum(r.n_ungraded for (_, r) in rows),
        " (asymmetric: ", sum(r.n_ungraded_asym for (_, r) in rows), ")")
    ungraded_meshes = [name for (name, r) in rows if r.max_face_ratio > 2 + 1.0e-12]
    println("meshes with a face of size ratio > 2 (package balance defect): ", length(ungraded_meshes), " of ", length(rows))
    println("worst values over the graded patches of all meshes:")
    for field in (:max_asym, :max_ideal_mismatch, :max_residual, :max_cartesian_mismatch, :max_operator_mismatch, :max_affine_jump, :max_affine_pressure)
        println("  ", rpad(string(field), 24), maximum(getfield(r, field) for (_, r) in rows))
    end
    dyadic(name) = !occursin("h0=0.37", name)          # all cell sizes exactly representable
    println("  max_asym over the dyadic meshes only   ", maximum(r.max_asym for (name, r) in rows if dyadic(name)))
    println("  max #faces at a wall vertex       ", maximum(r.max_wall_faces for (_, r) in rows))
    println("  #faces with two complete graded patches  ", sum(r.n_exact_faces for (_, r) in rows))
    println("graded patch classes: interior vertices | periodic-seam vertices")
    for key in sort(collect(union(keys(interior_classes), keys(seam_classes))); by = string)
        println("  ", rpad(string(key), 16), lpad(get(interior_classes, key, 0), 8), lpad(get(seam_classes, key, 0), 8))
    end
    expected = Set([(1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 2), (1, 2, 1, 2), (1, 2, 2, 2), :hanging])
    seen = Set(union(keys(interior_classes), keys(seam_classes)))
    @test expected ⊆ seen                                                      # every class was met
    # graded patches outside the six classes can only be corner patterns with a 4:1 DIAGONAL
    # (face-graded but not corner-balanced; produced by the balance defect) -- all symmetric above
    @test all(k -> k in expected || (k isa NTuple{4, Int} && maximum(k) == 4), seen)
    @test all(k -> get(interior_classes, k, 0) > 0, expected)
    @test all(k -> get(seam_classes, k, 0) > 0, expected)                      # also on periodic seams
end
