# Code check 01: kinetic energy and linear momentum, one and two velocities (eq:verif:energy, eq:verif:energy-two).
#
# Label:  eq:verif:energy, eq:verif:energy-two
# Source: docs/techref/chapters/14_verification.tex:23-52; src/simulation/diagnostics/interface.jl (_one_velocity_kinetic_energy,
#         _two_velocity_kinetic_energy, _one_velocity_linear_momentum, _two_velocity_linear_momentum, KineticEnergy, LinearMomentum),
#         src/two_phase/forces.jl (_staggered_liquid_fraction)
# Checks (11 random walled fixtures: uniform / composite meshes, drops, planes on grid lines, sloped / aligned / disc solids):
#   K1  KineticEnergy and LinearMomentum, one velocity, equal the independent sums (open dual volume from the strip areas, clamped
#       fraction from the strip weights, rho = rho_l C + rho_g (1 - C)), at density ratios 1000 and 6.6;
#   K2  the same for two velocities with the weights rho_l C, rho_g (1 - C) and NaN at absent phases ("zero weight => zero product");
#   K3  the open dual volume of the text equals geometry.face_dual_volumes / mesh.face_dual_volumes;
#   K4  two velocities with u^l = u^g reproduce the one-velocity diagnostics (algebraic consequence of the two formulas);
#   K5  boundary faces do not enter (boundary values set to 1e6 leave the sums unchanged);
#   K6  single phase: 1/2 rho sum V^o v^2 and rho sum V^o v (with and without solid);
#   K7  the clamp matters nowhere (weighted fraction inside [0,1]), faces carrying NaN exist (the weight rule is exercised).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-verif-diagnostics/scripts/code_01_kinetic_momentum.jl

using Test, Random
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
isdefined(Main, :VerifAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "verification_audit.jl"))
using TwoPhaseAMR
const VA = VerifAudit
const rel(a, b) = abs(a - b) / max(abs(b), 1.0e-300)

@testset "eq:verif:energy / energy-two" begin
    nan_faces = 0
    for fix in VA.fixtures()
        field = fix.field
        mesh = fix.mesh
        # K3
        gvol = fix.solid ? fix.geometry.face_dual_volumes : mesh.face_dual_volumes
        vo = [VA.open_dual_volume(field, f) for f in mesh.faces]
        @test maximum(abs.(vo .- gvol)) <= 1.0e-12 * maximum(gvol)
        for density in ((1000.0, 1.0), (6.6, 1.0))
            # K1
            flow, state = VA.flow_state(fix; two = false, density)
            E, M = VA.kinetic_momentum(field, flow.velocity, density; two = false)
            @test rel(diagnostic_value(KineticEnergy(), state), E) < 1.0e-12
            Mp = diagnostic_value(LinearMomentum(), state)
            @test maximum(abs.(Mp .- M)) <= 1.0e-12 * max(abs(M[1]), abs(M[2]))
            # K5 (boundary faces)
            saved = copy(flow.velocity.boundary)
            flow.velocity.boundary .= 1.0e6
            @test rel(diagnostic_value(KineticEnergy(), state), E) < 1.0e-12
            flow.velocity.boundary .= saved
            # K2
            flow2, state2 = VA.flow_state(fix; two = true, density, rng = MersenneTwister(11))
            nan_faces += count(isnan, flow2.velocity[1].interior) + count(isnan, flow2.velocity[2].interior)
            E2, M2 = VA.kinetic_momentum(field, flow2.velocity, density; two = true)
            @test isfinite(E2)
            @test rel(diagnostic_value(KineticEnergy(), state2), E2) < 1.0e-12
            Mp2 = diagnostic_value(LinearMomentum(), state2)
            @test maximum(abs.(Mp2 .- M2)) <= 1.0e-12 * max(abs(M2[1]), abs(M2[2]))
            # K4: equal phase velocities (the absent phase keeps NaN: zero weight)
            for i in eachindex(flow2.velocity[1].interior)
                a = flow2.velocity[1].interior[i]
                b = flow2.velocity[2].interior[i]
                common = isnan(a) ? b : a
                isnan(a) || (flow2.velocity[1].interior[i] = common)
                isnan(b) || (flow2.velocity[2].interior[i] = common)
                flow.velocity.interior[i] = isnan(common) ? 0.0 : common
            end
            @test rel(diagnostic_value(KineticEnergy(), state2), diagnostic_value(KineticEnergy(), state)) < 1.0e-12
            @test maximum(abs.(diagnostic_value(LinearMomentum(), state2) .- diagnostic_value(LinearMomentum(), state))) <= 1.0e-12 * max(abs(M2[1]), abs(M2[2]))
        end
        # K6: single phase
        rho0 = 2.3
        extra = fix.solid ? (; geometry = fix.geometry) : (;)
        sp = FlowSimulation(
            mesh, SinglePhase(density = rho0, dynamic_viscosity = 0.1);
            extra..., velocity = (x, y, t, a) -> 1.0, diagnostics = (),
        )
        rng = MersenneTwister(3)
        for a in eachindex(sp.velocity.interior)
            sp.velocity.interior[a] = randn(rng)
        end
        Esp = 0.5 * rho0 * sum(vo[i] * sp.velocity.interior[i]^2 for i in eachindex(vo))
        Msp = [rho0 * sum(vo[i] * sp.velocity.interior[i] for i in eachindex(vo) if mesh.faces[i].axis == k) for k in 1:2]
        @test rel(diagnostic_value(KineticEnergy(), simulation_state(sp)), Esp) < 1.0e-12
        @test maximum(abs.(diagnostic_value(LinearMomentum(), simulation_state(sp)) .- Msp)) <= 1.0e-12 * maximum(abs.(Msp))
        # K7: raw weighted fraction in [0,1]
        lo, hi = Inf, -Inf
        for f in mesh.faces
            w = c = 0.0
            for ref in (f.minus, f.plus)
                s = 0.5 * ForcesAudit.open_strip_area(field, ref, f) * mesh.blocks[ref.block].volume
                w += s * TwoPhaseAMR._conditional_fraction(field, ref)
                c += s
            end
            c > 1.0e-12 && (lo = min(lo, w / c); hi = max(hi, w / c))
        end
        @test lo >= -1.0e-12 && hi <= 1 + 1.0e-12
        println(rpad(fix.name, 52), " faces ", length(mesh.faces), "  fraction range [", round(lo, sigdigits = 3), ", ", round(hi, sigdigits = 3), "]")
    end
    @test nan_faces > 0
    println("NaN entries of absent phase velocities exercised: ", nan_faces)
end
