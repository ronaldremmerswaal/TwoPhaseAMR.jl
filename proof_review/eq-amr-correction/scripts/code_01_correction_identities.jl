# Code check 01: divergence-preserving refinement of a face velocity (eq:amr:correction, eq:amr:identities) and the flux
# through the surviving line pieces (eq:amr:face-line, eq:amr:face-bracket).
#
# Labels: eq:amr:correction, eq:amr:identities, eq:amr:face-line, eq:amr:face-bracket
# Source: docs/techref/chapters/12_amr_criteria_transfer.tex:526-583 (equations), src/mesh/adaptivity/field_transfer.jl:584-665, 280-385
# Checks (random 2:1-graded composite meshes: walled, doubly periodic, periodic-in-x channel; random labels, 6 regrids per
# seed; velocity = INDEPENDENT RANDOM NOISE on every face, so no smoothness or solenoidality is used):
#   C1  the extra correction of the package equals the chapter's four numbers (eq:amr:correction) AND the minimum-norm
#       solution of the 4x4 flux system (pinv), and it touches nothing but the four internal faces. The uncorrected values
#       come from two FaceFields (same data, no divergence correction) transferred in the same regrid;
#   C2  the three divergence identities (eq:amr:identities) hold to round-off, max|D u| does not grow;
#   C3  the integrated flux through every cluster of old/new line segments that covers the same union is unchanged
#       (a split face: halves sum to the old flux; a merge: the weighted mean), measured on the lines independently
#       of the package line machinery.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-amr-correction/scripts/code_01_correction_identities.jl

using Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "_shared", "amr_audit.jl"))
using .AmrAudit
using TwoPhaseAMR
const AA = AmrAudit

# FaceVelocity data of every axis as FaceFields of one axis (local numbering = order of the faces of this axis)
function split_axes!(fields, velocity, mesh)
    local_indices = [0, 0]
    for (k, face) in pairs(mesh.faces)
        local_indices[face.axis] += 1
        fields[face.axis].interior[local_indices[face.axis]] = velocity.interior[k]
    end
    local_indices .= 0
    for (k, face) in pairs(mesh.boundary_faces)
        local_indices[face.axis] += 1
        fields[face.axis].boundary[local_indices[face.axis]] = velocity.boundary[k]
    end
    return fields
end

function merge_axes(fields, mesh)
    raw = FaceVelocity(mesh, Float64)
    local_indices = [0, 0]
    for (k, face) in pairs(mesh.faces)
        local_indices[face.axis] += 1
        raw.interior[k] = fields[face.axis].interior[local_indices[face.axis]]
    end
    local_indices .= 0
    for (k, face) in pairs(mesh.boundary_faces)
        local_indices[face.axis] += 1
        raw.boundary[k] = fields[face.axis].boundary[local_indices[face.axis]]
    end
    return raw
end

# 4x4 incidence of the internal faces (columns 12, 34, 13, 24) on the children 1..4 (rows)
const INCIDENCE = [1.0 0.0 1.0 0.0; -1.0 0.0 0.0 1.0; 0.0 1.0 -1.0 0.0; 0.0 -1.0 0.0 -1.0]

function chapter_correction(d)
    f12 = (2d[1] - d[2] + d[3]) / 4
    f34 = d[1] + d[3] - f12
    f13 = d[1] - f12
    f24 = d[2] + f12
    return [f12, f34, f13, f24]
end

function run_case(name, mesh, seed)
    rng = MersenneTwister(seed)
    u = FaceVelocity(mesh, Float64)          # corrected velocity (default registration, no integral on walls)
    u.interior .= randn(rng, length(u.interior)); u.boundary .= randn(rng, length(u.boundary))
    fx = FaceField{1}(mesh, (x, y) -> 0.0); fy = FaceField{2}(mesh, (x, y) -> 0.0)
    split_axes!((fx, fy), u, mesh)
    workspace = AMRWorkspace(mesh)
    # No periodic integral correction anywhere: it would shift the line fluxes (it keeps the divergence, tested in 03).
    register_adaptive_field!(workspace, u; conserve_integral = false)
    register_adaptive_field!(workspace, fx; conserve_integral = false)
    register_adaptive_field!(workspace, fy; conserve_integral = false)
    stats_total = (; regrids = 0, cells = 0)
    worst = (; c1_formula = 0.0, c1_pinv = 0.0, c1_untouched = 0.0, c1_target = 0.0, id = 0.0, growth = 0.0, line = 0.0)
    nsurv = nvan = 0
    for round in 1:6
        labels = AA.random_labels(rng, mesh; pr = round <= 2 ? 0.6 : 0.3, pc = round <= 2 ? 0.1 : 0.6)
        split_axes!((fx, fy), u, mesh)       # raw fields start every regrid from the corrected velocity
        old_flux = AA.net_flux_by_key(mesh, u)
        old_div = AA.divergence_by_key(mesh, u)
        old_segs = AA.face_segments(mesh, u)
        old_max = maximum(maximum(abs, d) for d in values(old_div))
        stats = adapt!(workspace, labels; step = round, force = true)
        stats.changed || continue
        @assert isbalanced(mesh)
        # --- C1
        raw = merge_axes((fx, fy), mesh)
        raw_net = AA.net_flux_by_key(mesh, raw)
        corr = (u.interior .- raw.interior)
        @test maximum(abs, u.boundary .- raw.boundary; init = 0.0) == 0.0     # boundary faces are never corrected
        nx, ny = mesh.block_size
        # new cell -> (old parent key, parent cell, quadrant)
        key_of(cell) = mesh.blocks[cell.block].key
        cellid(cell) = (key_of(cell), cell.i, cell.j)
        function parent_of(cell)
            key = key_of(cell)
            (key.level > 0 && haskey(old_flux, AA.parent_key(key))) || return nothing
            fi, fj = mod(key.i, 2) * nx + cell.i, mod(key.j, 2) * ny + cell.j
            return (AA.parent_key(key), (fi + 1) ÷ 2, (fj + 1) ÷ 2), (isodd(fi) ? 0 : 1) + 2 * (isodd(fj) ? 0 : 1) + 1
        end
        # all children of every refined parent cell, ordered 1..4 (low-low, high-low, low-high, high-high)
        children = Dict{Tuple, Vector{Any}}()
        for (b, block) in pairs(mesh.blocks), j in 1:ny, i in 1:nx
            p = parent_of(CellRef(b, i, j))
            p === nothing && continue
            slot = get!(() -> Vector{Any}(nothing, 4), children, p[1])
            slot[p[2]] = CellRef(b, i, j)
        end
        correction_by_cell = Dict{Tuple, Vector{Float64}}()
        touched = falses(length(mesh.faces))
        for (pid, kids) in children
            all(!isnothing, kids) || error("incomplete family")
            target = old_flux[pid[1]][pid[2], pid[3]] / 4
            d = [target - raw_net[key_of(k)][k.i, k.j] for k in kids]
            @test abs(sum(d)) <= 1.0e-12 * max(1, maximum(abs, d))                 # sum of defects vanishes (flux through the sides kept)
            internal = Int[]
            Fm = fill(NaN, 4)
            for (k, face) in pairs(mesh.faces)
                (face.minus in kids && face.plus in kids) || continue
                push!(internal, k)
                # axis 1: faces (c1,c2) -> 12 and (c3,c4) -> 34;  axis 2: faces (c1,c3) -> 13 and (c2,c4) -> 24
                if face.axis == 1
                    slot = face.minus == kids[1] ? 1 : 2
                else
                    slot = face.minus == kids[1] ? 3 : 4
                end
                Fm[slot] = corr[k] * face.area
                touched[k] = true
            end
            @test length(internal) == 4
            F = Fm
            ref = chapter_correction(d)
            pin = pinv(INCIDENCE) * d
            worst = merge(
                worst, (;
                    c1_formula = max(worst.c1_formula, maximum(abs, F .- ref)),
                    c1_pinv = max(worst.c1_pinv, maximum(abs, F .- pin)),
                    c1_target = max(worst.c1_target, maximum(abs, INCIDENCE * F .- d)),
                )
            )
        end
        worst = merge(worst, (; c1_untouched = max(worst.c1_untouched, maximum(abs, corr[.!touched]; init = 0.0))))
        # --- C2
        new_div = AA.divergence_by_key(mesh, u)
        ie = AA.identity_errors(mesh, old_div, new_div)
        new_max = maximum(maximum(abs, d) for d in values(new_div))
        worst = merge(worst, (; id = max(worst.id, max(ie.unchanged, ie.refined, ie.coarsened) / old_max), growth = max(worst.growth, (new_max - old_max) / old_max)))
        # --- C3
        lf = AA.line_flux_check(old_segs, AA.face_segments(mesh, u))
        lf.worst > 1.0e-12 && println("  line-flux outlier in round $round: ", lf.detail)
        worst = merge(worst, (; line = max(worst.line, lf.worst)))
        nsurv += lf.survived; nvan += lf.vanished
        stats_total = (; regrids = stats_total.regrids + 1, cells = stats_total.cells + length(children))
    end
    println(
        rpad(name * " seed $seed", 22), " regrids=", stats_total.regrids, " refined parent cells=", stats_total.cells,
        " clusters surviving=", nsurv, " vanished/created=", nvan
    )
    println(
        "    C1 |F - chapter| (flux units, O(1) data) = ", worst.c1_formula, "   |F - pinv| = ", worst.c1_pinv, "   |A F - d| = ", worst.c1_target,
        "   untouched faces: ", worst.c1_untouched
    )
    println("    C2 identities (rel. to max|D|) = ", worst.id, "   growth of max|D| = ", worst.growth, "   C3 line flux (rel.) = ", worst.line)
    @test worst.c1_formula <= 1.0e-12 && worst.c1_pinv <= 1.0e-12 && worst.c1_target <= 1.0e-12
    @test worst.c1_untouched == 0.0
    @test worst.id <= 1.0e-12
    @test worst.growth <= 1.0e-12
    @test worst.line <= 1.0e-12 && nsurv > 0
    return stats_total
end

@testset "eq:amr:correction / identities / face-line (random meshes, noise velocity)" begin
    total = 0
    for seed in 1:3
        total += run_case("walled", AA.walled_mesh(), seed).cells
        total += run_case("periodic", AA.periodic_mesh(), 10 + seed).cells
        total += run_case("channel", AA.channel_mesh(), 20 + seed).cells
    end
    @test total > 200
    println("total refined parent cells examined: ", total)
end
