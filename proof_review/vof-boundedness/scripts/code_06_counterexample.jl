# Code check 06: explicit counterexamples to 0 <= C <= 1 at the stable time step (cut coarse donor with a hanging side; frozen carrier).
#
# Label:  none (paragraph "Boundedness." of Section 5.5)
# Source: docs/techref/chapters/05_vof_transport.tex:357-366 ("For a cut coarse donor with two segments on a side neither argument applies")
# What:   configurations found by code_04 (artefact apertures: only the donor is cut, its stored solid plane is set by hand, every other
#         cell is uncut, so the neighbours' traces of the donor's faces are 1 and the package's own rule gives a_f = (trace + 1)/2) in
#         which the CFL bounds (5.15) hold with equality somewhere (dt = stable_timestep) and the package's `advect!` raises
#         "WY conditional fraction out of bounds": the donating rectangles of a hanging segment and of the opposite face overlap in the
#         open part and contain the conservatively moved phase, so its outflow exceeds its volume.
#   (1) one-velocity carrier: only the upper right segment (u = 0.9368) and the left face (u = -0.3569) donate; chi = 0 (liquid moved);
#   (2) frozen carrier: liquid/gas velocities on the right segments and the left face; chi = 1 (gas moved).
# Checks: the independent evaluation of (5.15) (docs `independent_wy_bound`) equals the package's stable step; the replica fluxes give the
#         pre-clamp fraction outside [0,1]; the rectangles overlap in the open moved phase; the package raises its error; the largest
#         step fraction lambda* for which the replica stays in [0,1].
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/vof-boundedness/scripts/code_06_counterexample.jl   (from the repository root)

using TwoPhaseAMR
using Test, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D
isdefined(Main, :build_fields) || include(joinpath(@__DIR__, "search_helpers.jl"))

function donor_cpre(field, donor, carrier, dt)
    mesh = field.mesh
    volumes, liquids, _, regions = D.replica_fluxes(field, carrier, dt, 1)
    c0 = D.cond(field, donor); chi = c0 > 0.5 ? 1.0 : 0.0
    V = L = 0.0
    for (index, o) in D.cell_faces(mesh)[donor]
        mesh.faces[index].axis == 1 || continue
        V += o * volumes[index]; L += o * liquids[index]
    end
    return c0 + (chi * V - L) / D.open_volume_of(field, donor), chi, [r for r in regions if r.result.donor == donor]
end

function overlap_report(field, donor, regions, chi)
    best = 0.0; best_phase = 0.0
    for a in 1:length(regions), b in (a + 1):length(regions)
        inter = VofAudit.rect_intersection(regions[a].result.rectangle, regions[b].result.rectangle)
        isempty(inter) && continue
        op = D.open_part(field, donor, inter)
        isempty(op) && continue
        best = max(best, G.shoelace(op))
        n, aa = D.liquid_plane(field, donor)
        best_phase = max(best_phase, G.shoelace(G.clip_halfplane(op, n, aa; keep_less = chi == 0)))
    end
    return best, best_phase
end

function refused_by_package(field, velocity, dt; frozen = false)
    trial = copy(field)
    work = frozen ? WYAdvectionWorkspace(trial, velocity; interface_transport = FrozenInterfaceFluxes()) : WYAdvectionWorkspace(trial)
    try
        advect!(trial, velocity, dt, work; check_timestep = false)
        return "no error"
    catch err
        err isa ErrorException || rethrow()
        return first(split(sprint(showerror, err), ", faces"))
    end
end

@testset "explicit counterexamples to boundedness at the stable step" begin
    mesh = build_mesh(:hanging); centre = (1.5, 2.5)
    donor = find_cell(mesh, centre...)
    right, left = donor_faces(mesh, donor)
    @testset "(1) one-velocity carrier, coarse cut donor with a hanging side" begin
        p = (tw = 3.642192944522876, ow = 0.23059525565923286, tl = 5.777533614551216, ol = -0.39261586027343004)
        field = build_fields(mesh, centre, p; artifact = true)
        velocity = FaceVelocity(mesh)
        velocity.interior[right[1]] = 0.0; velocity.interior[right[2]] = 0.9368430659316056; velocity.interior[left[1]] = -0.35686834133268
        work = WYAdvectionWorkspace(field)
        dt = stable_timestep(velocity, work)
        ind = D.independent_wy_bound(field, D.one_velocity_rates(field, velocity))
        carrier = D.one_velocity_carrier(velocity)
        cpre, chi, regions = donor_cpre(field, donor, carrier, dt)
        ov, ov_phase = overlap_report(field, donor, regions, chi)
        lo, hi = 0.0, 1.0
        for _ in 1:50
            mid = (lo + hi) / 2
            (try donor_cpre(field, donor, carrier, mid * dt)[1] >= -1.0e-12 catch; false end) ? (lo = mid) : (hi = mid)
        end
        println("(1) dt = ", dt, " (independent (5.15): ", ind.bound, "; binding numbers in/out/ext: ", ind.incoming, ", ", ind.outgoing, ", ", ind.extrusion,
            "); donor open fraction ", D.open_fraction_of(field, donor), ", C = ", D.cond(field, donor), ", chi = ", chi,
            "; apertures of the active faces ", [D.aperture_of(field, i) for i in (right[2], left[1])],
            "; depths ", [round(r.result.depth; digits = 4) for r in regions])
        println("    pre-clamp C after the sweep (replica) = ", cpre, "; overlap of the rectangles in the open part ", ov, ", in the moved phase ", ov_phase,
            "; replica stays in [0,1] for dt <= ", lo, " * dt_WY")
        println("    package: ", refused_by_package(field, velocity, dt))
        @test isapprox(ind.bound, dt; rtol = 1.0e-12)
        @test chi == 0 && 0 < D.cond(field, donor) < 0.5
        @test cpre < -0.05                                   # liquid volume becomes negative: out of bounds by 0.1 of the open volume
        @test ov > 0 && ov_phase > 0
        @test occursin("out of bounds", refused_by_package(field, velocity, dt))
        @test 0 < lo < 1                                     # a smaller step avoids this configuration
    end
    @testset "(2) frozen carrier, same donor class" begin
        p = (tw = 6.047103058103431, ow = 0.3519469972142466, tl = 1.263045452558202, ol = 0.16999975582492988)
        field = build_fields(mesh, centre, p; artifact = true)
        ul = FaceVelocity(mesh); ug = FaceVelocity(mesh)
        ul.interior[right[1]] = -0.10191943684044136; ul.interior[right[2]] = 0.0; ul.interior[left[1]] = -0.19799910106953192
        ug.interior[right[1]] = -0.10848167605356168; ug.interior[right[2]] = 0.179524103894174; ug.interior[left[1]] = 0.0
        prepared = VofAudit.prepare_carrier(:frozen, field, (ul, ug))
        ind = D.independent_wy_bound(field, D.frozen_rates((ul, ug), prepared.work))
        cpre, chi, regions = donor_cpre(field, donor, prepared.carrier, prepared.dt)
        ov, ov_phase = overlap_report(field, donor, regions, chi)
        println("(2) dt = ", prepared.dt, " (independent (5.24): ", ind.bound, "); C = ", D.cond(field, donor), ", chi = ", chi, "; pre-clamp C = ", cpre,
            "; overlap in the open part ", ov, ", in the moved phase ", ov_phase)
        println("    package: ", refused_by_package(field, (ul, ug), prepared.dt; frozen = true))
        @test isapprox(ind.bound, prepared.dt; rtol = 1.0e-12)
        @test chi == 1 && cpre > 1.05
        @test ov > 0 && ov_phase > 0
        @test occursin("out of bounds", refused_by_package(field, (ul, ug), prepared.dt; frozen = true))
    end
end
