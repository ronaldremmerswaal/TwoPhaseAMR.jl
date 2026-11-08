# Code check 04: the 2:1 grading hypothesis is NOT guaranteed by BlockTreeMesh for every
# floating-point extent (finding of this audit; independent of the lemma itself).
#
# Label:  eq:setting:patch-metric
# Claim:  hypothesis "2:1-graded quadtree" of 03_discrete_setting.tex:304-305.
# Code:   src/mesh/topology/mesh.jl: _tree_box_intersects (strict float comparisons, no tolerance),
#         _collect_leaf_nodes_in_window!, _balance! (605-646: finds neighbours through the window
#         search, applies the tolerant _touch (587) only AFTER the exact-comparison pruning),
#         isbalanced (1007: exact comparisons too).
# Observation: with extent = 0.7 or 3.3 (periodic seams) or generic doubles such as
#         5.353681183789219 the final mesh can contain a face between blocks that differ by TWO
#         levels (size ratio 4) while isbalanced(mesh) == true. The touching coordinates of the two
#         blocks differ by one ulp (e.g. -8.9e-16), so the pruned window search drops the neighbour.
# Consequence (checked here): on such a 4:1 face the patch tensor of square cells is NOT symmetric,
#         _local_inverse_metric symmetrises it, and R_v Y_v != N_v (affine exactness lost) -- exactly
#         as the lemma's hypothesis analysis predicts. Every GRADED patch stays symmetric.
# The script passes whether or not the defect is present (it reports it); if the balancing is
# fixed, the statistics below simply become zero.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-setting-patch-metric/scripts/code_04_balance_defect.jl

import TwoPhaseAMR   # not `using`: the package also exports Cell and Face (names used by model.jl)
using Random: MersenneTwister
using Test
const AMR = TwoPhaseAMR
include(joinpath(@__DIR__, "code_helpers.jl"))

"""Largest cell-size ratio across a face; number of faces with ratio > 2."""
function face_ratios(mesh)
    ratio = 1.0
    bad = 0
    for f in mesh.faces
        a = mesh.blocks[f.minus.block].spacing[f.axis]
        b = mesh.blocks[f.plus.block].spacing[f.axis]
        r = max(a / b, b / a)
        ratio = max(ratio, r)
        r > 2 + 1.0e-9 && (bad += 1)
    end
    return ratio, bad
end

"""Statistics of the patch tensors of a mesh, split into graded and ungraded patches."""
function patch_statistics(mesh)
    patches = AMR._vertex_patches(mesh, Float64)
    n_graded_asym = n_ungraded = n_ungraded_asym = n_throw = 0
    worst_residual = 0.0
    max_graded_asym = 0.0
    for (key, faces) in patches
        (AMR._physical_vertex(mesh, key) || length(faces) <= 2) && continue
        Y, N = moment_matrices(faces)
        T = transpose(Y) * N
        asym = abs(T[1, 2] - T[2, 1]) / maximum(abs, T)
        cells, hx, hy = lattice_cells(mesh, key, faces)
        ideal = faces_at_vertex(cells, 0, 0)
        @assert maximum(abs, T - physical_moment(moment_integers(ideal), hx, hy)) <= 1.0e-12 * maximum(abs, T)
        if face_graded(ideal)
            max_graded_asym = max(max_graded_asym, asym)
        else
            n_ungraded += 1
            if asym > 1.0e-12
                n_ungraded_asym += 1
                try
                    worst_residual = max(worst_residual, maximum(abs, AMR._local_inverse_metric(copy(faces)) * Y - N))
                catch err
                    err isa ArgumentError || rethrow()
                    n_throw += 1       # sym(Y'N) not positive definite on this ungraded patch
                end
            end
        end
    end
    return (; n_ungraded, n_ungraded_asym, n_throw, worst_residual, max_graded_asym)
end

@testset "code_04: balance on floating-point extents" begin
    corner = (key, o, e) -> key.i == 0 && key.j == 0
    function chain_mesh(extent, periodic)
        return AMR.BlockTreeMesh(;
            base_mesh = (8, 8), block_size = (2, 2), extent = (extent, extent),
            periodic = periodic, max_level = 4, refine = corner,
        )
    end

    @testset "A: corner chain, periodic seams" begin
        for extent in (1.0, 2.0, 2pi, 10.0, 0.7, 3.3), periodic in ((true, true), (true, false))
            mesh = chain_mesh(extent, periodic)
            ratio, bad = face_ratios(mesh)
            s = patch_statistics(mesh)
            println("extent=", rpad(extent, 20), " periodic=", periodic, "  max face ratio=", ratio, "  isbalanced=",
                AMR.isbalanced(mesh), "  ungraded faces=", bad, "  ungraded patches=", s.n_ungraded,
                " (asymmetric ", s.n_ungraded_asym, ", max |RY-N| = ", round(s.worst_residual; sigdigits = 3), ")")
            @test s.max_graded_asym <= 1.0e-13            # the lemma, wherever the hypothesis holds
            @test AMR.isbalanced(mesh)                    # isbalanced shares the blind spot of _balance!
            @test (ratio <= 2 + 1.0e-9) == (s.n_ungraded == 0)
        end
        ratio, bad = face_ratios(chain_mesh(0.7, (true, true)))
        println("extent 0.7 reproduces the defect: ", ratio > 2 ? "yes (ratio $ratio)" : "no (fixed?)")
    end

    @testset "B: one generic double extent, random refinement, no periodicity" begin
        extent = 5.353681183789219
        rng = MersenneTwister(355739)
        refine = (key, o, e) -> rand(rng) < (0.5, 0.45, 0.4, 0.35)[min(key.level + 1, 4)]
        mesh = AMR.BlockTreeMesh(; base_mesh = (8, 8), block_size = (2, 2), extent = (extent, extent),
            periodic = (false, false), max_level = 3, refine = refine)
        ratio, bad = face_ratios(mesh)
        s = patch_statistics(mesh)
        println("extent=", extent, ": max face ratio=", ratio, ", faces with ratio > 2: ", bad, " of ", length(mesh.faces),
            ", isbalanced=", AMR.isbalanced(mesh), ", ungraded patches ", s.n_ungraded, " (asymmetric ", s.n_ungraded_asym,
            ", max |RY-N| = ", round(s.worst_residual; sigdigits = 3), ")")
        @test s.max_graded_asym <= 1.0e-13
        @test (bad == 0) == (s.n_ungraded == 0)
        if bad > 0
            # diagnosis: the touching coordinates of the two blocks of an offending face differ by an ulp
            f = first(f for f in mesh.faces if abs(mesh.blocks[f.minus.block].key.level - mesh.blocks[f.plus.block].key.level) > 1)
            bm, bp = mesh.blocks[f.minus.block], mesh.blocks[f.plus.block]
            gaps = (bm.origin[1] + bm.extent[1] - bp.origin[1], bm.origin[2] + bm.extent[2] - bp.origin[2],
                bp.origin[1] + bp.extent[1] - bm.origin[1], bp.origin[2] + bp.extent[2] - bm.origin[2])
            println("   offending blocks ", bm.key, " / ", bp.key, ": upper - origin gaps = ", gaps)
            @test any(g -> g != 0 && abs(g) < 1.0e-12, gaps)         # a sub-ulp-scale mismatch on the touching line
        end
    end

    @testset "C: 300 random extents in [0.3, 12.3], random refinement, all periodicities" begin
        rng = MersenneTwister(2024)
        n_mesh = Dict{Any, Int}(); n_bad = Dict{Any, Int}()
        n_defect = n_throwing = 0
        worst_graded_asym = 0.0
        for trial in 1:300
            extent = 0.3 + 12 * rand(rng)
            periodic = rand(rng, ((false, false), (true, false), (false, true), (true, true)))
            rng2 = MersenneTwister(rand(rng, 1:10^6))
            refine = (key, o, e) -> rand(rng2) < (0.5, 0.45, 0.4, 0.35)[min(key.level + 1, 4)]
            mesh = AMR.BlockTreeMesh(; base_mesh = (8, 8), block_size = (2, 2), extent = (extent, extent),
                periodic = periodic, max_level = 3, refine = refine)
            ratio, bad = face_ratios(mesh)
            n_mesh[periodic] = get(n_mesh, periodic, 0) + 1
            bad > 0 && (n_bad[periodic] = get(n_bad, periodic, 0) + 1)
            s = patch_statistics(mesh)
            worst_graded_asym = max(worst_graded_asym, s.max_graded_asym)
            if bad > 0
                n_defect += 1
                throws = try
                    AMR.GradientOperator(mesh); false
                catch err
                    err isa ArgumentError || rethrow()
                    true
                end
                n_throwing += throws
                throws && @test s.n_throw > 0        # the constructor fails exactly because of an ungraded patch
            end
        end
        println("meshes with the defect: ", n_defect, " of 300; GradientOperator(mesh) throws ArgumentError on ", n_throwing, " of them")
        for periodic in sort(collect(keys(n_mesh)); by = string)
            println("periodic=", periodic, ": ", get(n_bad, periodic, 0), " of ", n_mesh[periodic], " random meshes contain a face with size ratio > 2")
        end
        @test worst_graded_asym <= 1.0e-13
    end
end
