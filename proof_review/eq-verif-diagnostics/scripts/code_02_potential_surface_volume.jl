# Code check 02: potential energy from PLIC first moments, interface measure with the grid-line face term, phase volumes and
# masses, largest velocity (eq:verif:energies and the text around eq:verif:energy).
#
# Label:  eq:verif:energies
# Source: docs/techref/chapters/14_verification.tex:53-97; src/two_phase/forces.jl (_phase_first_moments,
#         _gravitational_potential_energy), src/simulation/diagnostics/interface.jl (InterfaceMeasure, PhaseVolume, PhaseMass,
#         MaximumVelocity), src/visualization/plic_geometry.jl (plic_segments), src/embedded/geometry_utils.jl (_local_plic_segment)
# Checks (11 random fixtures, see _shared/verification_audit.jl):
#   P1  PotentialEnergy = -g.(rho_l M^l + rho_g (M^o - M^l)) with independent polygon moments (open cell clipped by the solid plane,
#       liquid side n.x <= a of the PLIC line); the liquid polygon has area C |c^o| in every mixed cell (fixes the side convention);
#   P2  the 4 x 4 values of the unit test (1.75, 1.27, 0.875) are exact; the mass-at-centre alternative differs (so the sentence
#       "a mixed cell contributes the moment of its phase polygons, not its mass at its centre" is a real statement);
#   P3  oblique plane on a composite mesh: E_p equals the exact moments of the clipped box to 1.4e-5 (relative); the measure is 1.6 % too long
#   S1  InterfaceMeasure = sum of PLIC chords of the open polygons + sum over separating faces |f| a_f, independent evaluation;
#   S2  no interface is counted twice: neighbours of a separating face are never mixed cells;
#   S3  exact lengths of grid-line interfaces (8, 6.6 with a sloped solid, 7 with an aligned solid), also for offsets of 0.5 and 2
#       fraction floors (face term / PLIC segment: the measure is continuous across the floor);
#   S4  convergence of the measure of a drop (uniform meshes 16, 32, 64): observed order about 0.7 (NOT second order), and of the
#       potential energy (second order or better);
#   V1  PhaseVolume = (sum |c^o| C, |Omega^o| - V^l), PhaseMass = rho V; open volume from clipped polygons;
#   V2  MaximumVelocity: one velocity over interior and boundary faces; two velocities only over active faces.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-verif-diagnostics/scripts/code_02_potential_surface_volume.jl

using Test, Random
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
isdefined(Main, :VerifAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "verification_audit.jl"))
using TwoPhaseAMR
const VA = VerifAudit
const AMR = TwoPhaseAMR
const rel(a, b) = abs(a - b) / max(abs(b), 1.0e-300)

@testset "eq:verif:energies" begin
    # ---- P1, S1, S2, V1, V2 on the random fixtures
    for fix in VA.fixtures()
        field = fix.field
        mesh = fix.mesh
        density = (1000.0, 1.0)
        gravity = (0.31, -1.07)
        flow, state = VA.flow_state(fix; two = false, density, gravity)
        Ml, Mo, gap = VA.first_moments(field)
        @test gap <= 1.0e-12                                    # |liquid polygon| = C |c^o| in every mixed cell
        Ep = VA.potential_energy(field, density, gravity)
        @test rel(diagnostic_value(PotentialEnergy(), state), Ep) < 1.0e-12
        total, seg, facepart, doubled, skipped = VA.interface_measure(field)
        @test skipped == 0
        @test doubled == 0
        @test rel(diagnostic_value(InterfaceMeasure(), state), total) < 1.0e-12
        @test rel(diagnostic_value(SurfaceEnergy(), state), 0.7 * total) < 1.0e-12
        # V1
        liquid = sum(AMR._get(field.conditional_fraction, ref) * VA.polygon_moment(VA.open_polygon(field, ref))[1] * mesh.blocks[ref.block].volume for ref in AMR._cell_references(mesh))
        open_volume = sum(VA.polygon_moment(VA.open_polygon(field, ref))[1] * mesh.blocks[ref.block].volume for ref in AMR._cell_references(mesh))
        pv = diagnostic_value(PhaseVolume(), state)
        @test rel(pv[1], liquid) < 1.0e-12 && rel(pv[2], open_volume - liquid) < 1.0e-12
        @test diagnostic_value(PhaseMass(), state) ≈ [1000.0 * pv[1], pv[2]]
        # V2 (one velocity): interior and boundary faces
        vmax = max(maximum(abs, flow.velocity.interior), maximum(abs, flow.velocity.boundary))
        @test diagnostic_value(MaximumVelocity(), state) == vmax
        flow.velocity.boundary[1] = 7.5
        @test diagnostic_value(MaximumVelocity(), state) == max(7.5, vmax)
        # V2 (two velocities): a huge value on a face where a phase is absent is ignored
        flow2, state2 = VA.flow_state(fix; two = true, density)
        marked = 0
        for (i, face) in pairs(mesh.faces)
            C = VA.clamped_fraction(field, face)
            V = VA.open_dual_volume(field, face)
            for (p, absent) in ((1, !(C > 0) || V == 0), (2, !(C < 1) || V == 0))
                if absent && marked < 3
                    @test isnan(flow2.velocity[p].interior[i])       # the package marks exactly the faces of the text as absent
                    flow2.velocity[p].interior[i] = 1.0e3
                    marked += 1
                end
            end
        end
        expected = maximum(
            abs(flow2.velocity[p].interior[i]) for p in 1:2 for (i, face) in pairs(mesh.faces) if
                (V = VA.open_dual_volume(field, face); C = VA.clamped_fraction(field, face); V > 0 && (p == 1 ? C > 0 : C < 1))
        )
        bmax = 0.0
        for p in 1:2, (i, face) in pairs(mesh.boundary_faces)
            Cc = AMR._conditional_fraction(field, face.cell)
            op = field.geometry === nothing ? 1.0 : AMR._get(field.geometry.open_fraction, face.cell)
            (op > 0 && (p == 1 ? Cc > 0 : Cc < 1)) && (bmax = max(bmax, abs(flow2.velocity[p].boundary[i])))
        end
        @test diagnostic_value(MaximumVelocity(), state2) == max(expected, bmax)
        println(rpad(fix.name, 52), " E_p ", round(Ep, sigdigits = 8), "  |Gamma| ", round(total, sigdigits = 8), " = chords ", round(seg, sigdigits = 6), " + faces ", round(facepart, sigdigits = 6))
    end

    # ---- P3: an oblique planar interface on the composite mesh: exact moments and chord length of the box clipped by the half-plane
    let fix = VA.fixtures()[5]            # "composite oblique plane": liquid {0.6 x + 0.8 y <= 4.4}
        box = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]
        liquid = ForcesAudit.clip(box, (0.6, 0.8), 4.4; keep_less = true)
        A, mx, my = VA.polygon_moment(liquid)
        p, q = ForcesAudit.Geo.segment_in(box, (0.6, 0.8), 4.4)
        exact_length = hypot(p[1] - q[1], p[2] - q[2])
        Ml_exact = [mx, my]
        Mo_exact = [256.0, 256.0]        # ∫ x dA over the 8 x 8 box
        density, gravity = (1000.0, 1.0), (0.31, -1.07)
        _, state = VA.flow_state(fix; two = false, density, gravity)
        Ep_exact = -sum(gravity[k] * (density[1] * Ml_exact[k] + density[2] * (Mo_exact[k] - Ml_exact[k])) for k in 1:2)
        Ep = diagnostic_value(PotentialEnergy(), state)
        len = diagnostic_value(InterfaceMeasure(), state)
        println("oblique plane: E_p ", Ep, " exact ", Ep_exact, " (rel ", rel(Ep, Ep_exact), "); measure ", len, " exact ", exact_length, " (rel ", rel(len, exact_length), ")")
        @test rel(Ep, Ep_exact) < 1.0e-4          # not exact (LHF normals are not exact for this slope), but accurate to 1.4e-5
        # the PLIC chord sum is NOT exact for a straight line of rational slope 3/4 (LHF normals): it overestimates the length by 1.6 %
        @test 0.005 < rel(len, exact_length) < 0.03
        @test len > exact_length
    end

    # ---- P2: the 4 x 4 cases of test/diagnostics.jl, exact values, and the mass-at-centre alternative
    function four_by_four(offset; density = (4.0, 1.0), gravity = (-2.0, 0.0))
        mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
        field = InterfaceField(mesh, (x, y) -> x - offset)
        physics = TwoPhase(field; density, dynamic_viscosity = (0.1, 0.1), surface_tension = 0.25, gravity)
        flow = FlowSimulation(mesh, physics; diagnostics = ())
        return field, simulation_state(flow)
    end
    for (offset, exact) in ((0.5, 1.75), (0.3, 1.27))
        field, state = four_by_four(offset)
        @test diagnostic_value(PotentialEnergy(), state) ≈ exact atol = 1.0e-13
        # exact: liquid {x < offset}: M^l = offset^2/2, M^o = 1/2
        Ml = offset^2 / 2
        @test -(-2.0) * (4.0 * Ml + (0.5 - Ml)) ≈ exact atol = 1.0e-13
        # mass at the cell centre
        centre = sum(
            0.25^2 * (4.0 * AMR._conditional_fraction(field, ref) + 1.0 * (1 - AMR._conditional_fraction(field, ref))) *
                AMR.cellcenter(field.mesh, ref.block, ref.i, ref.j)[1] for ref in AMR._cell_references(field.mesh)
        ) * 2.0
        println("4x4 interface x = $offset: polygon moment $(diagnostic_value(PotentialEnergy(), state)) (exact $exact), mass at centres $centre")
        offset == 0.3 && @test abs(centre - exact) > 1.0e-3
        offset == 0.5 && @test centre ≈ exact atol = 1.0e-13          # an interface on a grid line: the two coincide
    end

    # ---- S3: interfaces on grid lines, and across the fraction floor
    function line_measure(shape, normal, intercept; kind = :amr)
        mesh = ForcesAudit.mesh_of(kind)
        geometry = shape === nothing ? nothing : GeometryField(mesh, shape.shape; lipschitz = shape.lipschitz)
        field = geometry === nothing ? InterfaceField(mesh) : InterfaceField(mesh; geometry, contact_angle = pi / 2)
        initialize!(field, normal, intercept)               # exact cell averages of the half-plane normal.x <= intercept
        physics = TwoPhase(field; density = (3.0, 1.0), dynamic_viscosity = (0.1, 0.1), surface_tension = 1.0, gravity = (0.0, -1.0))
        flow = geometry === nothing ? FlowSimulation(mesh, physics; diagnostics = ()) :
            FlowSimulation(mesh, physics; geometry, diagnostics = ())
        return diagnostic_value(InterfaceMeasure(), simulation_state(flow)), field
    end
    low = (; shape = ForcesAudit.solid_low, lipschitz = hypot(1, ForcesAudit.SOLID_LOW.slope))
    aligned = (; shape = (x, y) -> y - 1.0, lipschitz = 1.0)
    @test line_measure(nothing, (1.0, 0.0), 4.0)[1] ≈ 8.0 rtol = 1.0e-13
    @test line_measure(nothing, (0.0, 1.0), 3.5, kind = :uniform)[1] ≈ 8.0 rtol = 1.0e-13
    @test line_measure(aligned, (1.0, 0.0), 2.0)[1] ≈ 7.0 rtol = 1.0e-13
    @test line_measure(low, (1.0, 0.0), 5.0)[1] ≈ 8.0 - (0.9 + 0.1 * 5.0) rtol = 1.0e-12
    # x = 3.5 is a grid line of the coarse cells (width 0.5) of the left half; offsets of a fraction of the floor
    fl = sqrt(eps())
    h = 0.5
    for factor in (0.1, 0.5, 0.9, 1.1, 2.0, 10.0)
        value, field = line_measure(nothing, (1.0, 0.0), 3.5 + factor * fl * h)
        mixed = count(ref -> 0 < AMR._conditional_fraction(field, ref) < 1, AMR._cell_references(field.mesh))
        println("offset ", factor, " floors: measure ", value, " mixed cells ", mixed)
        @test value ≈ 8.0 rtol = 1.0e-9
        @test (factor < 1) == (mixed == 0)
    end

    # ---- S4: convergence of the measure of a drop and of the potential energy (uniform meshes)
    errs = Float64[]
    signed = Float64[]
    perrs = Float64[]
    for n in (16, 32, 64)
        mesh = BlockTreeMesh(base_mesh = (n, n), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false))
        field = InterfaceField(mesh, VA.disc(4.13, 3.87, 2.21); lipschitz = 1.0)
        physics = TwoPhase(field; density = (3.0, 1.0), dynamic_viscosity = (0.1, 0.1), surface_tension = 1.0, gravity = (0.0, -1.0))
        state = simulation_state(FlowSimulation(mesh, physics; diagnostics = ()))
        push!(signed, diagnostic_value(InterfaceMeasure(), state) - 2pi * 2.21)
        push!(errs, abs(signed[end]))
        # exact potential energy: -g.(rho_l M^l + rho_g (M^o - M^l)), M^l_y = pi r^2 cy, M^o_y = 32*4
        Ml = pi * 2.21^2 * 3.87
        Mo = 64 * 4.0
        push!(perrs, abs(diagnostic_value(PotentialEnergy(), state) - (3.0 * Ml + (Mo - Ml))))
    end
    println("drop measure errors (signed) ", signed, " rates ", log2.(errs[1:2] ./ errs[2:3]))
    println("drop potential-energy errors ", perrs, " rates ", log2.(perrs[1:2] ./ perrs[2:3]))
    # The sum of PLIC chords converges with an observed order of only about 0.7 (not second order): consistent, but slowly.
    @test all(diff(errs) .< 0)
    @test all(log2.(errs[1:2] ./ errs[2:3]) .> 0.5)
    @test all(log2.(errs[1:2] ./ errs[2:3]) .< 1.2)
    @test all(log2.(perrs[1:2] ./ perrs[2:3]) .> 1.8)
end
