# Code check 01: the phase apertures (4.21)-(4.22) against the thesis' definition (2.41).
#
# Label:  eq:interface:phase-aperture  (and eq:interface:liquid-trace)
# Claim:  04_interface.tex:563-590: phi^liq_f = 0 / a_f / clamp(1/2 sum_c trace^liq_c, 0, a_f), phi^gas_f = a_f - phi^liq_f,
#         on a boundary face clamp(trace^liq, 0, a_f); the deviation (07_cutcells.tex:117-125): "the thesis defines the
#         open ... aperture as the sums of the phase quantities a^o = {{a}}, each phase aperture the average of two
#         one-sided traces (2.41) ... Without clamping and covered neighbours both give the same sum."
# Thesis: a^pi_f = (1/2) sum_{c in C(f)} |f ∩ c^pi|/|f|  (basic_interface.tex, eq. 2.41); a^o = a^liq + a^gas.
# Code:   src/two_phase/forces.jl:313-371 (_face_aperture), src/two_phase/two_velocity/transport_geometry.jl:180-262.
# Method: films above a sloped wall (contact lines, triple cells, covered cells), on a uniform and on a 2:1-graded AMR
#         mesh (hanging faces: half-side traces), isotropic and anisotropic cells.  For EVERY interior and boundary face:
#         one-sided traces from polygons built independently (Sutherland--Hodgman + Cyrus--Beck), thesis apertures
#         (1/2 sum of the liquid / gas traces), a_f from (3.6).  Checks:
#           (i)  no covered neighbour: code liquid, gas and a_f equal the thesis' values to 1e-14;
#           (ii) covered neighbour: code = (0, 0), thesis = (1/2) trace of the open cell (the deviation of ch. 7);
#           (iii) the clamp of (4.22) is inactive (|raw - clamped| <= 1e-14) except at covered neighbours;
#           (iv) pure cases are exact: absent phase = 0.0 bitwise, liquid + gas = a_f;
#           (v)  boundary faces: clamp(trace) equals the code; the thesis' formula applied literally (factor 1/2 of a single
#                trace) would halve it.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-interface-phase-aperture/scripts/code_01_phase_apertures.jl
#         (from the repository root)

import TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :film_fixture) || include(joinpath(@__DIR__, "code_helpers.jl"))

function check_fixture(field, geometry; label)
    mesh = field.mesh
    counts = Dict(:faces => 0, :covered => 0, :hanging => 0, :cut => 0, :mixed => 0, :clamp_active => 0, :pure_gas => 0, :pure_liquid => 0, :boundary => 0, :boundary_mixed => 0, :tiny => 0)
    worst = Dict(:liquid => 0.0, :gas => 0.0, :open => 0.0, :partition => 0.0, :covered_rule => 0.0, :boundary => 0.0, :clamp_gap => 0.0)
    for (index, face) in pairs(mesh.faces)
        th = thesis_apertures(field, geometry, face)
        liquid, gas = AMR._phase_face_apertures(field, index)
        a = geometry.aperture[index]
        counts[:faces] += 1
        level(r) = mesh.blocks[r.block].key.level
        level(face.minus) != level(face.plus) && (counts[:hanging] += 1)
        cutcell = any(r -> 0 < AMR._get(geometry.solid.fraction, r) < 1, (face.minus, face.plus))
        cutcell && (counts[:cut] += 1)
        covered = th.minus.covered || th.plus.covered
        worst[:partition] = max(worst[:partition], abs(liquid + gas - a))
        if covered
            counts[:covered] += 1
            # (ii) the covered rule closes the face; the thesis' sum is half the open trace of the open cell
            @test a == 0 && liquid == 0 && gas == 0
            other = th.minus.covered ? th.plus : th.minus
            both_covered = th.minus.covered && th.plus.covered
            worst[:covered_rule] = max(worst[:covered_rule], abs(th.open - (both_covered ? 0.0 : other.open / 2)))
            @test isapprox(th.open, both_covered ? 0.0 : other.open / 2; atol = 1.0e-14)
            @test isapprox(th.liquid, both_covered ? 0.0 : other.liquid / 2; atol = 1.0e-14)
            continue
        end
        # (i) the same values
        worst[:liquid] = max(worst[:liquid], abs(liquid - th.liquid))
        worst[:gas] = max(worst[:gas], abs(gas - th.gas))
        worst[:open] = max(worst[:open], abs(a - th.open))
        @test abs(liquid - th.liquid) <= 1.0e-14
        @test abs(gas - th.gas) <= 1.0e-14
        @test abs(a - th.open) <= 1.0e-14
        # (iii) the clamp is a no-op away from covered neighbours: the raw average never leaves [0, a_f]
        raw = th.liquid
        worst[:clamp_gap] = max(worst[:clamp_gap], max(-raw, raw - a, 0.0))
        (raw < -1.0e-14 || raw > a + 1.0e-14) && (counts[:clamp_active] += 1)
        # tiny positive openings of an absent phase on a face of mixed cells (information only)
        (0 < gas < 1.0e-12 || 0 < liquid < 1.0e-12) && (counts[:tiny] += 1)
        # (iv) pure phases
        relevant = [r for r in (face.minus, face.plus) if AMR._get(geometry.open_fraction, r) > 0]
        liquid_possible = any(r -> AMR._get(field.conditional_fraction, r) > 0, relevant)
        gas_possible = any(r -> AMR._get(field.conditional_fraction, r) < 1, relevant)
        if !liquid_possible
            @test liquid === 0.0
            counts[:pure_gas] += 1
        elseif !gas_possible
            @test gas === 0.0
            counts[:pure_liquid] += 1
        else
            counts[:mixed] += 1
        end
    end
    # boundary faces
    for (index, face) in pairs(mesh.boundary_faces)
        counts[:boundary] += 1
        liquid, gas = AMR._phase_boundary_face_apertures(field, index)
        a = geometry.boundary_aperture[index]
        tr = traces(field, geometry, face.cell, face, face.side / 2)
        @test isapprox(a, tr.open; atol = 1.0e-14)
        @test isapprox(liquid, clamp(tr.liquid, 0, a); atol = 1.0e-14)
        @test isapprox(liquid + gas, a; atol = 1.0e-15)
        worst[:boundary] = max(worst[:boundary], abs(liquid - clamp(tr.liquid, 0, a)))
        0 < tr.liquid < tr.open && (counts[:boundary_mixed] += 1)
        # the thesis' (2.41) applied literally to a boundary face (one adjacent cell) would give half the trace
        tr.liquid > 1.0e-3 && @test !isapprox(liquid, tr.liquid / 2; atol = 1.0e-6)
    end
    @printf(
        "  %-22s faces %4d (hanging %3d, with a cut cell %3d, covered neighbour %3d, mixed %3d, pure gas %3d, pure liquid %3d); boundary %3d (%d mixed)\n",
        label, counts[:faces], counts[:hanging], counts[:cut], counts[:covered], counts[:mixed], counts[:pure_gas], counts[:pure_liquid], counts[:boundary], counts[:boundary_mixed]
    )
    @printf(
        "  %-22s max deviations: liquid %.1e, gas %.1e, a_f %.1e, liquid+gas-a_f %.1e, covered rule %.1e, boundary %.1e; clamp active (away from covered) on %d faces, max excess %.1e\n",
        "", worst[:liquid], worst[:gas], worst[:open], worst[:partition], worst[:covered_rule], worst[:boundary], counts[:clamp_active], worst[:clamp_gap]
    )
    counts[:tiny] > 0 && @printf("  %-22s faces with a tiny positive opening (< 1e-12) of a phase: %d\n", "", counts[:tiny])
    @test counts[:clamp_active] == 0
    return counts
end

@testset "phase apertures = thesis definition (no covered neighbour); covered rule closes the face" begin
    rng = MersenneTwister(2027)
    total = Dict(:hanging => 0, :covered => 0, :mixed => 0, :cut => 0, :boundary_mixed => 0)
    for (label, kwargs) in (
            ("uniform, unit cells", (;)),
            ("uniform, aniso 1.3x0.7", (; hx = 1.3, hy = 0.7)),
            ("AMR (2:1), unit cells", (; amr = true)),
            ("AMR (2:1), aniso", (; amr = true, hx = 0.8, hy = 1.4)),
            ("AMR, steeper wall", (; amr = true, k = 0.4, y0 = 0.9, phase = 3.5, amp = 1.5)),
            ("AMR, falling wall", (; amr = true, k = -0.25, y0 = 3.6, amp = 1.0, phase = 2.0, base = 2.6)),
        )
        field, geometry = film_fixture(; kwargs...)
        counts = check_fixture(field, geometry; label)
        for key in keys(total)
            total[key] += counts[key]
        end
    end
    println("  totals: ", total)
    @test total[:hanging] > 20 && total[:covered] > 20 && total[:mixed] > 40 && total[:cut] > 40
end
