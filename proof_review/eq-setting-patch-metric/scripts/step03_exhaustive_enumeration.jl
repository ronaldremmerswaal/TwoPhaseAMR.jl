# Step 03: exhaustive enumeration of all vertex neighbourhoods on dyadic quadtrees.
#
# Label:  eq:setting:patch-metric
# Claim:  03_discrete_setting.tex:304-305 ("symmetric on every patch of square cells ...
#         observed on the test meshes; a proof is pending"). Square cells => hx = hy.
# Checks: every quadtree on a window of side 8 with leaf sides in {8,4,2,1} (83522 trees) and
#         every interior lattice vertex v in {1..7}^2 that is a corner of some leaf: the face
#         segments ending at v are built from first principles (model.jl, mirroring _match_sides!,
#         _face_displacement, _vertex_key), and the integer moment matrix F = 4T/h^2 is tested.
#           (a) every vertex whose face segments join cells of side ratio <= 2 ("face-graded")
#               has F12 == F21  (=> T symmetric for square cells);
#           (b) the corner-balanced vertices (the package's _balance! rule) realise exactly the
#               classes FFFF, CFFF, CCFF, CFCF, CCCF and the hanging vertex (nothing else), with
#               3 (hanging) or 4 (corner) face segments;
#           (c) 4:1 face contacts exist whose tensor is NOT symmetric (grading is needed);
#           (d) a vertex inside a cell has no face (no patch).
#         Leaves of side 1/2 would only repeat the same classes at a finer scale: the touching
#         cells of a graded vertex have sides s and 2s only.
# Run:    julia --heap-size-hint=3G -t 1 proof_review/eq-setting-patch-metric/scripts/step03_exhaustive_enumeration.jl

using Test
isdefined(Main, :PatchModel) || include(joinpath(@__DIR__, "model.jl"))
using .PatchModel

const WINDOW = 8
const DEPTH = 3

# 2x2 symmetric eigenvalues (avoid LinearAlgebra so that the script needs no packages)
function eigvals_sym(T)
    a, b, d = T[1, 1], (T[1, 2] + T[2, 1]) / 2, T[2, 2]
    m = (a + d) / 2
    r = sqrt(((a - d) / 2)^2 + b^2)
    return [m - r, m + r]
end

function enumerate_all(depth::Int, window::Int)
    trees = all_trees(depth)
    total = 0
    graded = 0; graded_asym = 0
    balanced = 0; balanced_asym = 0
    ungraded = 0; ungraded_asym = 0
    corner_unbalanced_graded = 0
    interior_with_faces = 0
    graded_kinds = Dict{Tuple{Symbol, Int}, Int}()
    classes = Dict{Any, Tuple{Int, Matrix{Int}, Int}}()      # balanced: class => (count, F, #faces)
    witness = nothing
    mark = falses(window + 1, window + 1)
    for tree in trees
        cells = leaves(tree, window)
        fill!(mark, false)
        for c in cells, (dx, dy) in ((0, 0), (c.s, 0), (0, c.s), (c.s, c.s))
            mark[c.x + dx + 1, c.y + dy + 1] = true
        end
        for vy in 1:(window - 1), vx in 1:(window - 1)
            mark[vx + 1, vy + 1] || continue
            touch = touching(cells, vx, vy)
            kind = kind_of(touch, vx, vy)
            faces = faces_at_vertex(touch, vx, vy)
            if kind == :interior
                isempty(faces) || (interior_with_faces += 1)
                continue
            end
            isempty(faces) && continue          # v interior to a face segment: no patch
            total += 1
            F = moment_integers(faces)
            symmetric = F[1, 2] == F[2, 1]      # square cells: T12 = T21  <=>  F12 = F21
            is_graded = face_graded(faces)
            is_balanced = corner_balanced(touch)
            if is_graded
                graded += 1
                graded_kinds[(kind, length(faces))] = get(graded_kinds, (kind, length(faces)), 0) + 1
                symmetric || (graded_asym += 1)
                is_balanced || (corner_unbalanced_graded += 1)
            else
                ungraded += 1
                if !symmetric
                    ungraded_asym += 1
                    witness === nothing && (witness = (touch, F))
                end
            end
            if is_balanced
                balanced += 1
                symmetric || (balanced_asym += 1)
                smin = minimum(c.s for c in touch)
                Fn = F .÷ smin^2                        # F is homogeneous of degree 2 in the sides
                @assert Fn .* smin^2 == F
                key = kind == :corner4 ? class_of(sigma_pattern(touch, vx, vy)) :
                    kind == :hanging ? :hanging : (:other, kind)
                count, _, _ = get(classes, key, (0, Fn, length(faces)))
                classes[key] = (count + 1, Fn, length(faces))
            end
        end
    end
    return (;
        n_trees = length(trees), total, graded, graded_asym, balanced, balanced_asym,
        ungraded, ungraded_asym, corner_unbalanced_graded, interior_with_faces, graded_kinds,
        classes, witness,
    )
end

@testset "step03: exhaustive quadtree enumeration" begin
    s = enumerate_all(DEPTH, WINDOW)
    @test s.n_trees == 83522

    println("patches with >= 1 face:           ", s.total)
    println("  face-graded (ratio <= 2):       ", s.graded, "  asymmetric: ", s.graded_asym)
    println("  corner-balanced (package rule): ", s.balanced, "  asymmetric: ", s.balanced_asym)
    println("  graded but diagonal ratio 4:    ", s.corner_unbalanced_graded)
    println("  not face-graded (4:1 contact):  ", s.ungraded, "  asymmetric: ", s.ungraded_asym)
    println("face-graded patches by (kind, #faces): ", s.graded_kinds)
    println("\nclasses of corner-balanced patches (F = 4T/h^2; T = F/4 for hx = hy = h):")
    for key in sort(collect(keys(s.classes)); by = string)
        count, F, nf = s.classes[key]
        println(
            "  ", rpad(string(key), 14), " #faces=", nf, "  count=", lpad(count, 7),
            "  F=", F, "  eig(T)/h^2=", sort(eigvals_sym(F ./ 4))
        )
    end
    if s.witness !== nothing
        touch, F = s.witness
        println("\nexample of an asymmetric 4:1 contact: cells = ", touch)
        println("  F = ", F, "  (F12 = ", F[1, 2], " != F21 = ", F[2, 1], ")")
    end

    @test s.interior_with_faces == 0                                  # (d)
    @test s.graded_asym == 0                                          # (a)
    @test s.balanced_asym == 0
    expected = Set([(1, 1, 1, 1), (1, 1, 1, 2), (1, 1, 2, 2), (1, 2, 1, 2), (1, 2, 2, 2), :hanging])
    @test Set(keys(s.classes)) == expected                            # (b)
    for (key, (count, F, nf)) in s.classes
        @test nf == (key == :hanging ? 3 : 4)
        @test count > 0
    end
    @test s.ungraded_asym > 0                                         # (c)
    @test s.corner_unbalanced_graded > 0                              # corner balance not needed
    # all eigenvalues of T = F/4 (units h^2, h = smallest touching side) are >= 1/2: T is positive definite
    for (key, (count, F, nf)) in s.classes
        @test minimum(eigvals_sym(F ./ 4)) >= 0.5 - 1.0e-12
    end
end
