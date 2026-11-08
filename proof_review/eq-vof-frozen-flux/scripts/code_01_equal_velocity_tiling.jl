# Code check 01: with equal phase velocities and no solid, the frozen subface rectangles tile the one-velocity rectangle and
# the fluxes coincide (end of Section 5.8.1).
#
# Label:  eq:vof:frozen-flux  (with eq:vof:subface)
# Source: docs/techref/chapters/05_vof_transport.tex:628-634 ("Without a solid and with u^l_f = u^g_f the two subface
#         rectangles tile the one-velocity rectangle and the fluxes coincide with those of Section 5.4; with a solid they
#         differ in shape") and the check sentence 647-655 (equal_velocity_reduction_check)
# Claims: (i)   a^l_f + a^g_f = a^o_f = 1 on every face (up to rounding), so the subface widths add up to the face;
#         (ii)  the subface intervals of the two phases are adjacent, cover the segment and have disjoint interiors,
#               also at hanging faces, and also in the second sweep (positions follow the current reconstruction);
#         (iii) the frozen fluxes (open, liquid, gas) of BOTH sweeps and the fractions and scalars after a full step equal
#               those of the one-velocity path, for either sweep order, to round-off, also with anisotropic cells;
#         (iv)  the stable time steps coincide;
#         (v)   the statement is false with a solid (fluxes differ at cut faces / cut donors) and trivially false for
#               unequal velocities: the hypotheses are used.
# Method: random periodic fixtures on graded AMR meshes (hanging faces, up to 2 levels), random smooth/noisy NON-solenoidal
#         velocity u used as both phase velocities; the one-velocity path (WYAdvectionWorkspace(field)) and the frozen path
#         (WYAdvectionWorkspace(field, (u, u); interface_transport = FrozenInterfaceFluxes())) run side by side on copies.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-vof-frozen-flux/scripts/code_01_equal_velocity_tiling.jl
#         (from the repository root)

using TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D

function paired_step(rng, field, velocity; order, scheme)
    one_field, two_field = copy(field), copy(field)
    w1 = WYAdvectionWorkspace(one_field)
    two = (velocity, copy(velocity))
    w2 = WYAdvectionWorkspace(two_field, two; interface_transport = FrozenInterfaceFluxes())
    dt1 = stable_timestep(velocity, w1)
    dt2 = stable_timestep(two, w2)
    isfinite(dt1) || return nothing
    dt = dt1
    q1 = PhaseScalar(one_field; liquid = (x, y) -> 2 + sin(0.7x) * cos(0.4y), gas = (x, y) -> -1.5 + cos(0.5x + 0.3y))
    q2 = PhaseScalar(two_field; liquid = (x, y) -> 2 + sin(0.7x) * cos(0.4y), gas = (x, y) -> -1.5 + cos(0.5x + 0.3y))
    advect!(one_field, velocity, dt, w1; scalars = (q1,), sweep_order = order, scheme, amr_interface_scheme = scheme)
    advect!(two_field, two, dt, w2; scalars = (q2,), sweep_order = order, scheme, amr_interface_scheme = scheme)
    cells = D.all_cells(field.mesh)
    dfrac = maximum(abs(AMR._get(one_field.fraction, r) - AMR._get(two_field.fraction, r)) for r in cells)
    dq = maximum(max(abs(AMR._get(q1.liquid, r) - AMR._get(q2.liquid, r)),
        abs(AMR._get(q1.gas, r) - AMR._get(q2.gas, r))) for r in cells)
    scale = maximum(abs, w1.open_flux)
    dflux = maximum(max(abs(w1.open_flux[i] - w2.open_flux[i]), abs(w1.liquid_flux[i] - w2.liquid_flux[i]),
        abs(w1.gas_flux[i] - w2.gas_flux[i])) for i in eachindex(w1.open_flux)) / scale
    return (; dt1, dt2, dfrac, dq, dflux)
end

@testset "eq:vof:frozen-flux equal-velocity reduction" begin
    rng = MersenneTwister(20261110)
    worst = Dict(:dt => 0.0, :frac => 0.0, :scalar => 0.0, :flux => 0.0, :partition => 0.0, :tile_gap => 0.0)
    cases = 0
    faces_checked = hanging_checked = 0
    for rep in 1:16
        levels = rep <= 8 ? 1 : 2
        field, _ = VofAudit.random_field(rng; cut = false, levels, aniso = rep % 3 == 0)
        velocity = VofAudit.random_velocity(rng, field; noise = 0.5)
        order = rep % 2 == 0 ? (1, 2) : (2, 1)
        scheme = rep % 4 < 2 ? Upwind() : LaxWendroff()
        res = paired_step(rng, field, velocity; order, scheme)
        res === nothing && continue
        cases += 1
        worst[:dt] = max(worst[:dt], abs(res.dt1 - res.dt2) / res.dt1)
        worst[:frac] = max(worst[:frac], res.dfrac)
        worst[:scalar] = max(worst[:scalar], res.dq)
        worst[:flux] = max(worst[:flux], res.dflux)
        # (i), (ii): apertures and independent subface intervals on the interface state at the start of the step
        reconstruct!(field)
        w = WYAdvectionWorkspace(field, (velocity, copy(velocity)); interface_transport = FrozenInterfaceFluxes())
        AMR._freeze_phase_apertures!(field, w)
        for (index, face) in pairs(field.mesh.faces)
            worst[:partition] = max(worst[:partition], abs(w.liquid_aperture[index] + w.gas_aperture[index] - 1))
            donor = face.minus
            lo, hi = D.tangent_interval(field.mesh, donor, face)
            ll = D.independent_subface(field, w.liquid_aperture[index], w.gas_aperture[index], index, face, donor, true)
            gg = D.independent_subface(field, w.liquid_aperture[index], w.gas_aperture[index], index, face, donor, false)
            # the two intervals cover [lo, hi] without overlap (an empty interval is a point)
            widths = (ll[2] - ll[1]) + (gg[2] - gg[1])
            overlap = max(0.0, min(ll[2], gg[2]) - max(ll[1], gg[1]))
            covered = min(ll[1], gg[1]) <= lo + 1.0e-13 && max(ll[2], gg[2]) >= hi - 1.0e-13
            worst[:tile_gap] = max(worst[:tile_gap], abs(widths - (hi - lo)) / (hi - lo), overlap / (hi - lo))
            @test covered || (ll[2] - ll[1] == 0 || gg[2] - gg[1] == 0)
            faces_checked += 1
            D.cell_level(field.mesh, face.minus) != D.cell_level(field.mesh, face.plus) && (hanging_checked += 1)
        end
    end
    println("equal-velocity reduction: cases ", cases, ", faces ", faces_checked, " (hanging ", hanging_checked, ")")
    println("worst: ", worst)
    @test cases >= 14 && hanging_checked > 100
    @test worst[:partition] <= 4eps()
    @test worst[:tile_gap] <= 1.0e-13
    @test worst[:dt] <= 1.0e-14
    @test worst[:frac] <= 1.0e-13
    @test worst[:scalar] <= 1.0e-10                      # intrinsic values; fractions near the floor amplify rounding
    @test worst[:flux] <= 1.0e-13
end

@testset "eq:vof:frozen-flux: the hypotheses are used" begin
    rng = MersenneTwister(99)
    # (v-a) with a solid the fluxes differ at the cut faces (docs fixtures: no open part smaller than 0.11 cells, so the
    #       stable step is not tiny and the shape difference of the rectangles is resolved)
    diff_cut = 0.0; diff_open = 0.0; cut_faces = open_faces = 0; rel_cut = 0.0
    halvings = Dict{Symbol, Int}()
    for kind in (:cut, :film)
        field, _ = D.vof_fixture(kind)
        velocity = D.streamfunction_velocity(field.mesh, D.vof_psi_walls)
        reconstruct!(field)
        one = WYAdvectionWorkspace(field)
        two_velocity = (velocity, copy(velocity))
        two = WYAdvectionWorkspace(field, two_velocity; interface_transport = FrozenInterfaceFluxes())
        AMR._freeze_phase_apertures!(field, two)
        dt0 = min(stable_timestep(velocity, one), stable_timestep(two_velocity, two))
        done = false
        for attempt in 0:6                      # the frozen subface strip can be too small at the stable step (vof-boundedness)
            dt = dt0 / 2^attempt
            try
                for axis in 1:2
                    AMR._compute_directional_wy_fluxes!(field, velocity, dt, one, axis)
                    AMR._compute_directional_wy_fluxes!(field, two_velocity, dt, two, axis)
                end
            catch err
                err isa ErrorException || rethrow()
                continue
            end
            halvings[kind] = attempt; done = true
            break
        end
        @test done
        done || continue
        for (index, face) in pairs(field.mesh.faces)
            donor = one.open_flux[index] >= 0 ? face.minus : face.plus
            scale = D.cell_volume(field.mesh, donor)
            d = abs(one.liquid_flux[index] - two.liquid_flux[index]) / scale
            cutface = D.aperture_of(field, index) < 1 || 0 < D.solid_fraction_of(field, face.minus) < 1 ||
                0 < D.solid_fraction_of(field, face.plus) < 1
            if cutface
                diff_cut = max(diff_cut, d); cut_faces += 1
                abs(one.liquid_flux[index]) > 0 && (rel_cut = max(rel_cut, abs(one.liquid_flux[index] - two.liquid_flux[index]) /
                    abs(one.liquid_flux[index])))
            else
                diff_open = max(diff_open, d); open_faces += 1
            end
        end
    end
    println("halvings of the stable step needed for the frozen path on the solid fixtures: ", halvings)
    println("with a solid: max flux difference (in donor volumes) at cut faces ", diff_cut, " (", cut_faces,
        " faces; relative to the flux ", rel_cut, "), at open faces ", diff_open, " (", open_faces, " faces)")
    @test diff_cut > 1.0e-4                    # the reduction does not hold at cut faces
    @test diff_open <= 1.0e-14                 # and holds on the faces away from the solid
    # (v-b) unequal velocities: the fluxes differ
    field, _ = VofAudit.random_field(rng; cut = false, levels = 1)
    ul, ug = VofAudit.random_phase_velocity(rng, field)
    reconstruct!(field)
    one = WYAdvectionWorkspace(field)
    two = WYAdvectionWorkspace(field, (ul, ug); interface_transport = FrozenInterfaceFluxes())
    AMR._freeze_phase_apertures!(field, two)
    AMR._compute_directional_wy_fluxes!(field, ul, 0.01, one, 1)
    AMR._compute_directional_wy_fluxes!(field, (ul, ug), 0.01, two, 1)
    @test maximum(abs, one.liquid_flux .- two.liquid_flux) > 1.0e-6
end

@testset "eq:vof:frozen-flux: the rectangle-area function is additive over the tiling" begin
    rng = MersenneTwister(4242)
    worst = 0.0
    for _ in 1:20000
        theta = 2pi * rand(rng)
        n = (cos(theta), sin(theta))
        a = (rand(rng) - 0.5) * 1.6
        x0, x1 = sort(rand(rng, 2) .- 0.5)
        y0, y1 = sort(rand(rng, 2) .- 0.5)
        m = x0 + rand(rng) * (x1 - x0)                # split the tangential range (here x) at m
        whole = AMR._plane_fraction(n[1], n[2], a, x0, x1, y0, y1) * (x1 - x0) * (y1 - y0)
        left = AMR._plane_fraction(n[1], n[2], a, x0, m, y0, y1) * (m - x0) * (y1 - y0)
        right = AMR._plane_fraction(n[1], n[2], a, m, x1, y0, y1) * (x1 - m) * (y1 - y0)
        worst = max(worst, abs(whole - left - right))
    end
    println("additivity of the plane/rectangle area over a tangential split: worst ", worst)
    @test worst <= 1.0e-15
end
