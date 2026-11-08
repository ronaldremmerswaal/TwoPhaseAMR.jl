# Code check 01: eq:momentum:ctu at the level of one dual face, against the package's `_phase_transport`.
#
# Label:  eq:momentum:ctu  (eq:momentum:states for the bulk branch, eq:momentum:weights for the partial volumes)
# Source: docs/techref/chapters/06_momentum_transport.tex:652-690; src/two_phase/phase_face_transport.jl
#         (_phase_transport, _ctu_phase_transport, _phase_bulk_state)
# For every dual face of the package's phase schedules (uniform periodic, composite periodic, composite walled, with a solid disc /
# sloped wall) and random primal phase volumes (random signs), random staggered values and predicted fractions around the threshold
# theta = 1/2, the triple (Upsilon_g, V_m^{-1}-numerator, V_p^{-1}-numerator) returned by the package equals
#     CTU branch  (min(n_m, n_p) < theta):  Upsilon = sum_f w_{g,f} Phi_f,   sum_f p_f (q_m - q_{u(p_f)}),   sum_f p_f (q_p - q_{u(p_f)}),
#                 p_f = w_{g,f} Phi_f, u = tail if p_f >= 0 else head;
#     bulk branch:  Upsilon,  Upsilon (q_m - state),  Upsilon (q_p - state),  state of eq:momentum:states without the compressive term,
#                   Courant number |Upsilon| / min(V_m, V_p) (open volumes with a solid), scheme or interface scheme by regularity,
# with the independent states, volumes, regularity and upstream unknowns of docs/techref/figures/momentum.jl (the weights themselves are
# compared with rules (N), (T) in the audit of eq:momentum:identity).  Hard checks: the CTU branch for every face; the
# bulk branch for Upwind, Centred, LaxWendroff, Fromm, MonotonizedCentral, VanLeer on the schedules whose face classification is
# not affected by issue 62 (periodic meshes); on the bounded and embedded schedules the package currently flags every dual face as
# an AMR interface (issue 62): there the package equals the replica that treats every dual face as non-regular (upwind in the bulk), which is
# REPORTED (not asserted: the bulk state of the selected scheme is not used there).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-momentum-ctu/scripts/code_01_ctu_formula.jl

using Test, Random, LinearAlgebra
isdefined(Main, :MomentumAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_audit.jl"))
using .MomentumAudit
using TwoPhaseAMR
const MA = MomentumAudit
const D = MA.Docs
const AMR = TwoPhaseAMR

"""Package schedule of the phase transport of component `k`."""
function package_schedule(mesh, geometry, k)
    geometry === nothing && return AMR._phase_transport_connectivity(mesh, Val(k), Float64).schedule
    return AMR._geometry_phase_schedule(geometry, Val(k))
end

"""Signed partial volumes of a package dual face, oriented from `face.minus` to `face.plus`: `[(flux face, w)]`."""
function package_partials(schedule, face, mesh)
    if schedule isa AMR.AMRDualAdvectionSchedule
        return [(schedule.flux_indices[p], schedule.flux_weights[p]) for p in face.first_weight:(face.first_weight + face.weight_count - 1)]
    end
    return [(face.flux_a, face.area / (2mesh.faces[face.flux_a].area)), (face.flux_b, face.area / (2mesh.faces[face.flux_b].area))]
end

const THREE_POINT_SCHEMES = Union{Fromm, MonotonizedCentral, VanLeer, Minmod, Superbee}

"""Independent evaluation of eq:momentum:ctu and the bulk branch for the dual face `(m, p)` of the independent schedule `s`."""
function independent_dual_face(mesh, s, m, p, axis, partials, flux, values, nm, np, theta, scheme, interface_scheme, regular_flag)
    parts = [w * flux[h] for (h, w) in partials]
    volume = sum(parts)
    if min(nm, np) < theta
        mdiff = pdiff = 0.0
        for part in parts
            up = part >= 0 ? values[m] : values[p]
            mdiff += part * (values[m] - up)
            pdiff += part * (values[p] - up)
        end
        return volume, mdiff, pdiff
    end
    selected = regular_flag ? scheme : interface_scheme
    nu = abs(volume) / min(s.V[m], s.V[p])
    upstream = selected isa THREE_POINT_SCHEMES && !s.bounded ?
        (D.independent_upstream(mesh, s.dofs, s.adjacency, m, p, axis), D.independent_upstream(mesh, s.dofs, s.adjacency, p, m, axis)) :
        ((0, 0.0), (0, 0.0))
    state = D.independent_state(selected, values, m, p, volume, nu, 0.0, upstream)
    return volume, volume * (values[m] - state), volume * (values[p] - state)
end

@testset "eq:momentum:ctu: the dual-face flux against the package" begin
    rng = MersenneTwister(2027)
    theta = 0.5
    stats = Dict(:faces_ctu => 0, :faces_bulk => 0, :bulk_selection_matches => 0, :bulk_selection_checked => 0, :opposite_signs => 0)
    worst = Dict(:ctu => 0.0, :bulk_periodic => 0.0)
    fixtures = Any[]
    push!(fixtures, (:uniform, D.vof_uniform_mesh(), nothing))
    push!(fixtures, (:composite, MA.random_mesh(rng; levels = 1, refine_probability = 0.5), nothing))
    push!(fixtures, (:composite2, MA.random_mesh(rng; levels = 2, refine_probability = 0.5), nothing))
    wm = MA.random_mesh(rng; periodic = (false, false), levels = 1, refine_probability = 0.5)
    push!(fixtures, (:walled, wm, nothing))
    cm = MA.random_mesh(rng; levels = 1, refine_probability = 0.5)
    push!(fixtures, (:cut_disc, cm, GeometryField(cm, MA.random_solid(rng, cm.extent; kind = :disc); lipschitz = 2)))
    wcm = MA.random_mesh(rng; periodic = (false, false), levels = 1, refine_probability = 0.5)
    push!(fixtures, (:cut_wall, wcm, GeometryField(wcm, MA.random_solid(rng, wcm.extent; kind = :wall); lipschitz = 2)))
    schemes = (Upwind(), Centred(), LaxWendroff(), Fromm(), MonotonizedCentral(), VanLeer())
    for (name, mesh, geometry) in fixtures, k in (1, 2)
        schedule = package_schedule(mesh, geometry, k)
        s = D.phase_schedule(mesh, geometry, k)
        n = s.dofs.n
        periodic_schedule = all(mesh.periodic) && geometry === nothing
        flux = [randn(rng) * 0.3 * mesh.faces[i].area for i in eachindex(mesh.faces)]
        values = randn(rng, n)
        for face in schedule.faces
            partials = package_partials(schedule, face, mesh)                 # oriented minus -> plus
            key = (min(face.minus, face.plus), max(face.minus, face.plus), face.axis)
            regular_flag = get(s.regular, key, false)
            for (nm, np) in ((0.2, 0.9), (0.9, 0.2), (0.7, 0.8), (0.95, 0.99), (0.1, 0.3)), scheme in schemes
                interface_scheme = default_amr_interface_scheme(scheme)
                ctu = min(nm, np) < theta
                cvolume, cm_diff, cp_diff = AMR._phase_transport(
                    face, schedule, flux, mesh, values, 0.5, 0.5, nm, np, theta, scheme, interface_scheme,
                )
                ivolume, im_diff, ip_diff = independent_dual_face(
                    mesh, s, face.minus, face.plus, face.axis, partials, flux, values, nm, np, theta, scheme, interface_scheme,
                    regular_flag,
                )
                err = max(abs(cvolume - ivolume), abs(cm_diff - im_diff), abs(cp_diff - ip_diff)) /
                    (abs(cvolume) + abs(cm_diff) + abs(cp_diff) + 1.0e-300)
                if ctu
                    stats[:faces_ctu] += 1
                    signed = [w * flux[h] for (h, w) in partials]
                    any(>(0), signed) && any(<(0), signed) && (stats[:opposite_signs] += 1)
                    worst[:ctu] = max(worst[:ctu], err)
                    @test err <= 1.0e-13
                else
                    stats[:faces_bulk] += 1
                    if periodic_schedule
                        worst[:bulk_periodic] = max(worst[:bulk_periodic], err)
                        @test err <= 1.0e-11                      # limiter ratios amplify rounding of nearly equal differences
                    else
                        stats[:bulk_selection_checked] += 1
                        err <= 1.0e-13 && (stats[:bulk_selection_matches] += 1)
                    end
                end
            end
        end
    end
    println("eq:momentum:ctu dual-face audit: ", stats)
    println("worst relative differences (hard): ", worst)
    MA.excluded(
        "bulk branch on bounded / embedded schedules: the package equals the replica that treats every dual face as non-regular (upwind in the bulk, as the chapter states); the 'bulk state of the selected scheme' of eq:momentum:ctu does not hold there",
        "$(stats[:bulk_selection_matches]) of $(stats[:bulk_selection_checked]) checked bulk evaluations agree with the all-non-regular replica (issue 62)"
    )
    @test stats[:faces_ctu] > 5000
    @test stats[:opposite_signs] > 100                     # the case where CTU differs from net upwinding occurs
end
