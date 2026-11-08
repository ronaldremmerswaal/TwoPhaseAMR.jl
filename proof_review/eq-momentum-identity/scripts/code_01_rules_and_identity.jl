# Code check 01: rules (N), (T) and the divergence identity against the package's dual schedules.
#
# Label:  eq:momentum:identity  (eq:momentum:weights, eq:momentum:dual-divergence, eq:momentum:regular)
# Source: docs/techref/chapters/06_momentum_transport.tex:132-147, 189-228;
#         src/operators/advection/staggered_schedules.jl (_amr_dual_schedule, _uniform_dual_schedule, _bounded_phase_schedule),
#         src/embedded/operators.jl (_cut_cell_dual_schedule)
# Checks, on random 2:1-balanced composite meshes (periodic / walled / mixed, with and without a solid), both components:
#   C1  the weights of the package's schedules equal the weights built from the incidences by rules (N), (T) alone
#       (cut schedule: times the aperture), key by key, to 0;  the phase schedule of a bounded mesh equals the rules with the
#       boundary faces as extra unknowns;
#   C2  the dual-divergence matrix Q of the package minus the printed right-hand side P equals the closed-form residual of
#       step01 (S1/S2) in EVERY entry (so the identity is exact exactly where the closed form vanishes);
#   C3  Q = P on every unknown that satisfies the printed hypothesis H0 (both cells of f have unknowns on both sides);
#   C4  H0 implies T-completeness (the cells across the transverse faces have unknowns on the side where f lies) on every
#       2:1-balanced mesh tested, so H0 alone suffices in the stated setting;
#   C6  the interface flag of the package's AMR schedules equals the printed non-regularity of eq:momentum:regular (weights, uncut volumes);
#   C5  the fine segment f_1 of the hanging side of the composite fixture has 8 dual faces with the weights quoted in the text.
# Hard checks cover the claim under its hypotheses.  The T-incomplete unknowns of limitation (i) (issue 60, being fixed) are
# EXCLUDED from the hard checks and only reported: the fix is expected to add flux terms there.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-momentum-identity/scripts/code_01_rules_and_identity.jl

using Test, Random, LinearAlgebra
isdefined(Main, :MomentumAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_audit.jl"))
using .MomentumAudit
using TwoPhaseAMR
const MA = MomentumAudit
const D = MA.Docs

function fixture(rng; periodic, solid, levels)
    mesh = MA.random_mesh(rng; periodic, levels, refine_probability = 0.5)
    geometry = solid === nothing ? nothing :
        GeometryField(mesh, MA.random_solid(rng, mesh.extent; kind = solid); lipschitz = 2)
    return mesh, geometry
end

# the phase schedule of the package for a bounded mesh in the same normalised form
function phase_schedule_weights(mesh, k)
    schedule = TwoPhaseAMR._bounded_phase_schedule(mesh, Val(k), Float64)
    weights = Dict{NTuple{3, Int}, Dict{Int, Float64}}()
    for face in schedule.faces
        d = Dict{Int, Float64}()
        for p in face.first_weight:(face.first_weight + face.weight_count - 1)
            d[schedule.flux_indices[p]] = schedule.flux_weights[p]
        end
        weights[(face.minus, face.plus, face.axis)] = d
    end
    return weights
end

@testset "eq:momentum:identity: rules and identity against the package" begin
    rng = MersenneTwister(20261108)
    totals = Dict(
        :meshes => 0, :unknowns => 0, :regular => 0, :wall => 0, :cutwall => 0, :tincomplete => 0, :cut_meshes => 0,
        :walled_meshes => 0, :wall_open => 0, :wall_half => 0, :regularity_checked => 0,
    )
    worst = Dict(:weights => 0.0, :phase_weights => 0.0, :closed_form => 0.0, :identity_h0 => 0.0)
    excluded_gap = Dict(:weights => 0.0, :rows => 0.0)
    cases = vcat(
        [((true, true), nothing, 1) for _ in 1:5],
        [((true, true), nothing, 2) for _ in 1:3],
        [((true, true), :disc, 1) for _ in 1:4],
        [((false, false), nothing, 1) for _ in 1:6],
        [((false, false), nothing, 2) for _ in 1:3],
        [((true, false), nothing, 1) for _ in 1:3],
        [((false, true), nothing, 1) for _ in 1:3],
        [((false, false), :wall, 1) for _ in 1:3],
        [((false, false), :disc, 1) for _ in 1:3],
    )
    for (periodic, solid, levels) in cases
        mesh, geometry = fixture(rng; periodic, solid, levels)
        nf = length(mesh.faces)
        aper = geometry === nothing ? ones(nf) : Float64.(geometry.aperture)
        cell_faces = MA.faces_of_cells(mesh)
        totals[:meshes] += 1
        solid === nothing || (totals[:cut_meshes] += 1)
        all(periodic) || (totals[:walled_meshes] += 1)
        for k in (1, 2)
            inc = MA.component_incidence(mesh, k)
            cls = [MA.unknown_class(mesh, inc, g, aper; cell_faces) for g in inc.interior]
            ws = D.staggered_workspace(FaceField{k}(mesh), geometry)
            code = MA.schedule_weights(ws; mesh)
            rules = MA.rule_weights(mesh, inc)
            clean(key) = cls[key[1]] != :tincomplete && cls[key[2]] != :tincomplete
            gap = MA.weight_gap(rules, code; scale = aper, keep = clean)
            worst[:weights] = max(worst[:weights], gap)
            @test gap == 0
            if ws.schedule isa TwoPhaseAMR.AMRDualAdvectionSchedule
                # eq:momentum:regular: the package's interface flag is the negation of the printed regularity (uncut weights and volumes)
                regular = D.independent_regular(rules, [mesh.face_dual_volumes[g] for g in inc.interior])
                for face in ws.schedule.faces
                    key = (face.minus, face.plus, face.axis)
                    clean(key) || continue
                    totals[:regularity_checked] += 1
                    @test face.refinement_interface == !regular[key]
                end
            end
            all_gap = MA.weight_gap(rules, code; scale = aper)                   # over all dual faces, T-incomplete ones included
            excluded_gap[:weights] = max(excluded_gap[:weights], isfinite(all_gap) ? all_gap : 1.0)
            if !all(periodic) || geometry !== nothing
                # bounded / embedded phase schedule: boundary faces are unknowns; weights are the uncut rules
                incb = MA.component_incidence(mesh, k; boundary = true)
                pgap = MA.weight_gap(MA.rule_weights(mesh, incb), phase_schedule_weights(mesh, k))
                worst[:phase_weights] = max(worst[:phase_weights], pgap)
                @test pgap == 0
            end
            for t in (1, 2)
                Q = MA.dual_divergence_matrix(code, inc.n, nf, t)
                P = MA.identity_rhs_matrix(mesh, inc, nf, t) .* aper'
                R = Q .- P
                Rhat = MA.predicted_residual_matrix(mesh, inc, nf, t; cell_faces) .* aper'
                for (l, g) in enumerate(inc.interior)
                    totals[:unknowns] += 1
                    totals[cls[l]] += 1
                    row_gap = maximum(abs, R[l, :] .- Rhat[l, :])
                    row = maximum(abs, R[l, :])
                    if cls[l] == :tincomplete
                        excluded_gap[:rows] = max(excluded_gap[:rows], row_gap)
                        continue
                    end
                    worst[:closed_form] = max(worst[:closed_form], row_gap)
                    @test row_gap <= 1.0e-14                                         # C2 (outside limitation (i))
                    if cls[l] == :regular
                        worst[:identity_h0] = max(worst[:identity_h0], row)
                        @test row <= 1.0e-14                                         # C3
                        @test MA.t_complete(mesh, inc, g; cell_faces)                 # C4
                    elseif t == k && aper[g] == 1
                        # wall-adjacent unknown: the axial residual is -(1/2)(S^- - S^+) Phi_f in column f (unit flux columns)
                        totals[:wall_open] += 1
                        @test row <= 0.5 + 1.0e-14
                        row > 0.5 - 1.0e-14 && (totals[:wall_half] += 1)
                    end
                end
            end
        end
    end
    println("identity rules audit: ", totals)
    println("worst (hard checks): ", worst)
    MA.excluded(
        "weights / rows next to T-incomplete unknowns (limitation (i), issue 60): largest deviation from the rules", excluded_gap;
        note = "expected to change when the fix adds the missing flux terms"
    )
    @test totals[:regular] > 10_000 && totals[:wall] > 500
    @test totals[:wall_half] == totals[:wall_open]       # every full-aperture wall-adjacent unknown misses exactly (1/2) Phi_f
    @test worst[:identity_h0] <= 1.0e-14
end

@testset "eq:momentum:identity: the hanging fine segment of the text (C5)" begin
    mesh = D.vof_amr_mesh()
    inc = MA.component_incidence(mesh, 1)
    rules = MA.rule_weights(mesh, inc)
    f1 = D.face_index_at(mesh, (4.0, 5.25))
    l1 = inc.local_of[f1]
    connections = Dict{NTuple{2, Float64}, Vector{Tuple{Int, Float64, Float64}}}()
    centres = MA.unknown_centres(mesh, inc)
    count = 0
    for ((i, j, axis), d) in rules
        (i == l1 || j == l1) || continue
        count += 1
        other, sign = i == l1 ? (j, 1.0) : (i, -1.0)           # orientation from f_1 to the other unknown
        connections[centres[other]] = [(axis, mesh.faces[h].area, sign * w) for (h, w) in sort(collect(d); by = first)]
    end
    @test count == 8                                            # chkMomCoarseLinks
    # the other segment of the same side: Upsilon_g = 1/4 Phi_{f2} - 1/4 Phi_{f1}
    seg = connections[(4.0, 5.75)]
    @test any(x -> x[1] == 1 && abs(x[3] + 0.25) < 1.0e-15, seg) && any(x -> x[1] == 1 && abs(x[3] - 0.25) < 1.0e-15, seg)
    # couplings of weight 1/8 to the segments of the coarse sides above and below
    eighths = [c for (c, v) in connections for x in v if x[1] == 2 && abs(abs(x[3]) - 0.125) < 1.0e-15]
    @test length(eighths) == 4
    @test sort(unique(eighths)) == [(4.0, 4.25), (4.0, 4.75), (4.0, 6.25), (4.0, 6.75)]
    ws = StaggeredAdvectionWorkspace(FaceField{1}(mesh))
    @test ws.schedule.volumes[l1] == 3 / 8                      # chkMomCoarseVolume
    @test all(f -> f.refinement_interface, [f for f in ws.schedule.faces if f.minus == l1 || f.plus == l1])
    # the code's weights of this face equal the rules
    @test MA.weight_gap(rules, MA.schedule_weights(ws; mesh)) == 0
end
