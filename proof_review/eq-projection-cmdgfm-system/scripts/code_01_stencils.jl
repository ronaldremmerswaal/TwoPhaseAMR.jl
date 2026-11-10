# Code check 01: CMDGFM face data, staggered normal, transverse/inference faces, β, unresolved rules and diagonal dominance, against an
# independent transcription of the text (patches from face coordinates, own face data).
#
# Labels: eq:projection:mixed, eq:projection:staggered-normal, eq:projection:transverse, eq:projection:beta, eq:projection:unresolved,
#         eq:projection:dominance
# Source: docs/techref/chapters/10_projection.tex:204-216, 262-360, 396-420
# Checks (4 chapter fixtures, drop with a phase-jump predictor):
#   S1  mixed set; cell normals point into the gas (n_c·∇C < 0 on mixed cells), so η_c is the unit normal into the gas;
#   S2  classification by the rules 1-5 of (unresolved) in the printed order, resolved set, transverse set (filtered), weights, R_f, inference faces and β_f equal the code's
#       (`_cmdgfm_jump_stencils`) -- exact set equality; R, w, β to 1e-14;
#   S3  all dominated inference faces: 0 <= s <= 1 < |R|, 0 <= β < 1 (the code's clamp/error never act: counts);
#   S4  diagonal dominance: off-diagonal sum of row f = β|R| + (1-β) s (all neighbours present) <= 1, equal to 1 iff |R| >= 1; assembled jump block
#       margins = 1 - Σ|offdiag| (code) agree with the formula;
#   S5  the statement "a third of the mixed faces are unresolved, almost all by rule 2" (counts printed).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-cmdgfm-system/scripts/code_01_stencils.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit

include(joinpath(@__DIR__, "code_01_stencils_lib.jl"))

fl_active(d, g) = d.floor < d.C[g] < 1 - d.floor

"""Fixtures that exercise the branches the chapter fixtures miss (aligned normals, rule 4/5 variety)."""
function stress_fixtures()
    out = Pair{String, Any}[]
    for kind in (:uniform, :amr)
        mesh = FA.mesh_of(kind)
        push!(out, "drop r=1.3 $(kind)" => InterfaceField(mesh, (x, y) -> hypot(x - 3.9, y - 3.1) - 1.3; lipschitz = 1))
        push!(out, "thin slab $(kind)" => InterfaceField(mesh, (x, y) -> abs(y - 3.9) - 0.55; lipschitz = 1))
        push!(out, "wavy $(kind)" => InterfaceField(mesh, (x, y) -> y - 4.2 - 0.5 * sinpi(2x / 2.7); lipschitz = 1.7))
        push!(out, "grid-line interface $(kind)" => InterfaceField(mesh, (x, y) -> y - 4.0; lipschitz = 1))
    end
    return out
end

@testset "CMDGFM stencils against an independent transcription" begin
    for (name, field) in vcat(PA.fixtures(), stress_fixtures())
        mesh = field.mesh
        solver = CMDGFMProjectionSolver(field; density = PA.DENSITY)
        work = CMDGFMProjectionWorkspace(solver)
        cdata = AMR._projection_face_data(solver, work)
        d = PA.face_data(field)
        @test sort(d.mixed) == sort(collect(cdata.mixed))
        # S1: normal convention: the liquid occupies the side {ξ·x <= intercept} of the PLIC line in the reference cell, so ξ (and η_c) points into the gas
        mism = 0; tot = 0
        for ref in AMR._cell_references(mesh)
            c = AMR._conditional_fraction(field, ref)
            (field.fraction_floor < c < 1 - field.fraction_floor) || continue
            isnothing(field.geometry) || AMR._get(field.geometry.open_fraction, ref) >= 1 - 1.0e-12 || continue     # cut cells: C is relative to the open part
            ξ = (AMR._get(field.normal_x, ref), AMR._get(field.normal_y, ref)); a = AMR._get(field.intercept, ref)
            tot += 1
            liquid_side = FA.polygon_area(FA.clip([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], ξ, a; keep_less = true))
            mism += abs(liquid_side - c) > 1.0e-9 ? 1 : 0
        end
        println(rpad(name, 18), " S1: mixed cells ", tot, ", liquid on the {ξ·x <= a} side: mismatches ", mism)
        @test mism == 0
        stencils = Dict(s.face => s for s in AMR._cmdgfm_jump_stencils(solver, cdata))
        ind = classify(field, d)
        classes = Dict{Symbol, Int}()
        for (f, s) in ind
            classes[s.class] = get(classes, s.class, 0) + 1
        end
        @test sort(collect(keys(ind))) == sort(collect(keys(stencils)))
        maxβ = maxR = maxw = 0.0; clampcount = 0; nbeta_lt1 = 0; ntight = 0
        for (f, s) in ind
            c = stencils[f]
            @test c.resolved == (s.class == :resolved)
            s.class == :resolved || continue
            @test c.full_remove == s.aligned
            @test c.transverse_faces == s.trans
            @test c.gas_inference_face == s.ig && c.liquid_inference_face == s.il
            maxβ = max(maxβ, abs(c.beta - s.beta)); maxR = max(maxR, abs(c.ratio - s.R) / max(1, abs(s.R)))
            isempty(s.w) || (maxw = max(maxw, maximum(abs.(c.transverse_weights .- s.w))))
            if s.ig > 0
                @test 0 <= s.s <= 1 && abs(s.R) > 1 && 0 <= s.beta < 1               # S3
                s.beta < 1 && (nbeta_lt1 += 1)
            end
        end
        @test maxβ <= 1.0e-14 && maxR <= 1.0e-14 && maxw <= 1.0e-14
        # S4: dominance
        velocity = PA.two_predictor(field, MersenneTwister(2))
        assembled = AMR._assemble_cmdgfm_system!(solver, work, PA.DT, velocity)
        B = assembled.jump_block
        marg = zeros(size(B, 1)); diag_ = zeros(size(B, 1))
        for col in axes(B, 2), p in nzrange(B, col)
            row = rowvals(B)[p]; v = abs(nonzeros(B)[p])
            row == col ? (diag_[row] += v) : (marg[row] += v)
        end
        # rows are the resolved stencils in the order of `assembled.stencils` (face = jump column)
        worst = 0.0; below = 0; tight_ok = 0; tight = 0
        for (row, st) in pairs(assembled.stencils)
            s = ind[st.face]
            off_all = abs(s.R) * (s.aligned ? 0.0 : 1.0) * s.beta + (1 - s.beta) * s.s          # all neighbours present as unknowns
            @test diag_[row] == 1.0
            @test marg[row] <= off_all + 1.0e-14                                                  # missing unknowns only lower it
            @test off_all <= 1 + 1.0e-14                                                          # (dominance), right inequality
            if abs(s.R) >= 1 && !s.aligned
                tight += 1
                abs(off_all - 1) <= 1.0e-14 && (tight_ok += 1)
            else
                @test off_all < 1 - 1.0e-12 || abs(s.R) >= 1 - 1.0e-12                              # strict for |R| < 1
            end
            # the left inequality is an equality if every neighbour carries an unknown
            if all(g -> g in assembled.mixed_faces, vcat(s.trans, s.ig > 0 ? [s.ig, s.il] : Int[]))
                @test abs(marg[row] - off_all) <= 1.0e-13
            end
        end
        @test tight == tight_ok
        margins = diag_ .- marg
        zero_margin = count(<=(64eps()), margins)
        println("    S4: min margin of the assembled block = ", minimum(margins), ", rows with margin <= 64 eps: ", zero_margin, " (tight rows |R|>=1: ", tight, ")")
        @test minimum(margins) >= -64eps()
        # S6: rules 4, 5 read literally would also apply to faces with an aligned normal (the code exits earlier): count the faces for which it would matter
        lit = 0
        for (f, s_) in ind
            (s_.class == :resolved && s_.aligned) || continue
            tr = [g for g in PA.patch_faces(mesh, f, 3 - mesh.faces[f].axis) if fl_active(d, g) && d.al[g] + d.ag[g] > d.floor]
            ends = PA.face_endpoints(mesh.faces[f])
            onwall = any(e -> e[1] <= 1.0e-12 || e[1] >= 8 - 1.0e-12 || e[2] <= 1.0e-12 || e[2] >= 8 - 1.0e-12, ends)
            level = any(g -> mesh.faces[g].area != mesh.faces[f].area, tr)
            need = (onwall || level) ? 1 : 2
            (length(tr) < need || sum(d.C[g] * (1 - d.C[g]) for g in tr; init = 0.0) <= 0) && (lit += 1)
        end
        println("    S6: aligned resolved faces ", count(s_ -> s_.class == :resolved && s_.aligned, values(ind)), "; of which rule 4 read literally would reject: ", lit)
        nm = length(d.mixed); nunres = count(!=(:resolved), [s.class for s in values(ind)])
        println(
            "    classes: ", sort(collect(classes); by = first), "  mixed=", nm, " active-by-fraction=", length(ind), " unresolved=", nunres,
            " (", round(100 * nunres / length(ind), digits = 1), " % of the faces with a row candidate), β<1: ", nbeta_lt1, ", tight rows (|R|>=1): ", tight,
            "  max|Δβ|=", maxβ, " max|ΔR|=", maxR, " max|Δw|=", maxw
        )
    end
end
