# Code check 02: STRETCHED cells (aspect ratio r = hx/hy != 1) against the closed formula.
#
# Label:  eq:setting:patch-metric
# Claim:  03_discrete_setting.tex:291-305 (stretched counterexample: non-symmetric at refinement
#         corners, 'chkStretchedAsymmetric' patches) and the guard "throws if sym(Y'N) is not
#         positive definite" (03_discrete_setting.tex:323-324).
# Derived result (steps 01-02): for four cells with a corner at v, side pattern (sNE,sNW,sSW,sSE)
#         in {h, 2h}, cell widths (r*hy, hy) for the smallest cell,
#             T12 - T21 = (r^2 - 1)/4 * hy^2 * (sigNE + sigSW - sigNW - sigSE),
#         sig = 1 for a coarse (2h) and 0 for a fine (h) cell; hanging vertices: T12 = T21 = 0.
# Method: package patches on random / designed stretched meshes against that formula; replicate
#         the figure fixture and compare with the macros in generated/checks.tex; positive
#         definiteness thresholds of sym(Y'N) (r^2 - 10 r + 1 etc.) against GradientOperator.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-setting-patch-metric/scripts/code_02_stretched.jl

import TwoPhaseAMR   # not `using`: the package also exports Cell and Face (names used by model.jl)
using LinearAlgebra: eigvals, Symmetric
using Test
const AMR = TwoPhaseAMR
include(joinpath(@__DIR__, "code_helpers.jl"))

"""Asymmetry statistics of one mesh against the closed formula."""
function stretched_audit(mesh)
    patches = AMR._vertex_patches(mesh, Float64)
    n_tensor = 0
    n_asym = 0
    formula_error = 0.0
    classes_asym = Dict{Any, Int}()
    classes_sym = Dict{Any, Int}()
    residual_sym = 0.0
    residual_asym = 0.0
    for (key, faces) in patches
        (AMR._physical_vertex(mesh, key) || length(faces) <= 2) && continue
        n_tensor += 1
        Y, N = moment_matrices(faces)
        T = transpose(Y) * N
        scale = maximum(abs, T)
        cells, hx, hy = lattice_cells(mesh, key, faces)
        touch = touching(cells, 0, 0)
        kind = kind_of(touch, 0, 0)
        r = hx / hy
        predicted = 0.0
        if kind == :corner4
            sigma = collect(sigma_pattern(touch, 0, 0)) .- 1     # (NE, NW, SW, SE): 1 = coarse
            predicted = (r^2 - 1) / 4 * hy^2 * (sigma[1] + sigma[3] - sigma[2] - sigma[4])
        end
        asym = T[1, 2] - T[2, 1]
        formula_error = max(formula_error, abs(asym - predicted) / scale)
        residual = maximum(abs, AMR._local_inverse_metric(copy(faces)) * Y - N)
        label = patch_label(cells)
        if abs(asym) > 1.0e-12 * scale
            n_asym += 1
            classes_asym[label] = get(classes_asym, label, 0) + 1
            residual_asym = max(residual_asym, residual)
        else
            classes_sym[label] = get(classes_sym, label, 0) + 1
            residual_sym = max(residual_sym, residual)
        end
    end
    return (; n_tensor, n_asym, formula_error, classes_asym, classes_sym, residual_sym, residual_asym)
end

function checks_macro(name)
    text = read(joinpath(@__DIR__, "..", "..", "..", "docs", "techref", "figures", "generated", "checks.tex"), String)
    m = match(Regex("\\\\newcommand\\\\" * name * "\\{([^}]*)\\}"), text)
    return m === nothing ? nothing : m.captures[1]
end

# class names of the asymmetric corner classes (D4 canonical patterns, 1 = fine, 2 = coarse)
const ASYMMETRIC_CLASSES = Set([(1, 1, 1, 2), (1, 2, 1, 2), (1, 2, 2, 2)])   # CFFF, CFCF, CCCF

@testset "code_02: stretched cells" begin
    @testset "random stretched meshes: formula and asymmetric classes" begin
        asym_classes = Dict{Any, Int}()
        seed = 1000
        worst_formula = 0.0
        for aspect in (2.0, 0.5, 1.5, 4.0), periodic in ((false, false), (true, true), (true, false))
            for (block_size, roots) in (((2, 2), (3, 3)), ((4, 4), (2, 2)))
                seed += 1
                mesh = random_mesh(seed; block_size, roots, max_level = 3, periodic, h0 = 1.0, aspect = aspect)
                @test AMR.isbalanced(mesh)
                # use the formula audit only (GradientOperator may legitimately throw for extreme r)
                s = stretched_audit(mesh)
                worst_formula = max(worst_formula, s.formula_error)
                @test s.formula_error <= 1.0e-12
                for (k, v) in s.classes_asym
                    asym_classes[k] = get(asym_classes, k, 0) + v
                end
                @test all(k -> k in ASYMMETRIC_CLASSES, keys(s.classes_asym))
                # the symmetric classes of a stretched mesh may only be FFFF, CCFF, hanging
                @test all(k -> !(k in ASYMMETRIC_CLASSES) || k isa Tuple, keys(s.classes_sym))
                @test s.residual_sym <= 1.0e-12
                s.n_asym > 0 && @test s.residual_asym > 1.0e-3
            end
        end
        println("worst |T12-T21| vs formula (relative): ", worst_formula)
        println("asymmetric classes met: ", sort(collect(asym_classes); by = first))
        @test Set(keys(asym_classes)) == ASYMMETRIC_CLASSES
    end

    @testset "figure fixture (affine_mesh), square vs stretched, against generated/checks.tex" begin
        refined = Set([AMR.BlockKey(0, 1, 1), AMR.BlockKey(0, 2, 1), AMR.BlockKey(0, 1, 2), AMR.BlockKey(1, 3, 3)])
        make(extent) = AMR.BlockTreeMesh(;
            base_mesh = (8, 8), block_size = (2, 2), extent, periodic = (false, false),
            max_level = 2, refine = (key, origin, ext) -> key in refined,
        )
        square = stretched_audit(make((1.0, 1.0)))
        stretched = stretched_audit(make((2.0, 1.0)))
        println("square:    patches=", square.n_tensor, " asymmetric=", square.n_asym)
        println("stretched: patches=", stretched.n_tensor, " asymmetric=", stretched.n_asym,
            " classes=", stretched.classes_asym, " residual(asym)=", stretched.residual_asym)
        @test square.n_asym == 0
        @test stretched.formula_error <= 1.0e-12
        @test all(k -> k in ASYMMETRIC_CLASSES, keys(stretched.classes_asym))
        for (name, value) in (
                ("chkSquarePatches", square.n_tensor), ("chkSquareAsymmetric", square.n_asym),
                ("chkStretchedPatches", stretched.n_tensor), ("chkStretchedAsymmetric", stretched.n_asym),
            )
            documented = checks_macro(name)
            documented === nothing && (@warn "macro $name not found in generated/checks.tex"; continue)
            println("  ", name, ": document says ", documented, ", recomputed ", value)
            @test parse(Int, documented) == value
        end
    end

    @testset "positive-definiteness guard of sym(Y'N) for stretched cells" begin
        # r^2 - 10 r + 1 < 0 (CFFF), r^2 - 6 r + 1 < 0 (CFCF), r^2 - 22 r + 1 < 0 (CCCF)
        function builds(selector, aspect)
            mesh = AMR.BlockTreeMesh(;
                base_mesh = (8, 8), block_size = (2, 2), extent = (8.0 * aspect, 8.0),
                periodic = (false, false), max_level = 1,
                refine = (key, origin, ext) -> key.level == 0 && selector(key.i, key.j),
            )
            try
                AMR.GradientOperator(mesh)
                return true
            catch err
                err isa ArgumentError && occursin("degenerate vertex patch", err.msg) && return false
                rethrow()
            end
        end
        checker = (i, j) -> iseven(i + j)                                    # corners CFCF (+ CCFF, hanging)
        lshape = (i, j) -> (i, j) in ((1, 1), (2, 1), (1, 2))                # CFFF at the re-entrant corner
        single = (i, j) -> (i, j) == (1, 1)                                  # CCCF at the four block corners
        r_cfcf = 3 + 2sqrt(2)         # 5.828...
        r_cfff = 5 + sqrt(24)         # 9.899...
        r_cccf = 11 + sqrt(120)       # 21.95...
        for (selector, rc, name) in ((checker, r_cfcf, "CFCF"), (lshape, r_cfff, "CFFF"), (single, r_cccf, "CCCF"))
            @test builds(selector, 1.0)
            @test builds(selector, rc * 0.97)
            @test !builds(selector, rc * 1.03)
            @test builds(selector, 1 / (rc * 0.97))
            @test !builds(selector, 1 / (rc * 1.03))
            println("  ", name, ": GradientOperator builds for r < ", round(rc; digits = 3), " and 1/r < that, throws beyond")
        end
    end
end
