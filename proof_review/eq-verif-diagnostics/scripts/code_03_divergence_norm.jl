# Code check 03: the divergence norm is the projection constraint (eq:verif:divergence-norm).
#
# Label:  eq:verif:divergence-norm
# Source: docs/techref/chapters/14_verification.tex:98-117; src/simulation/diagnostics/interface.jl (DivergenceNorm,
#         _diagnostic_volume_flux!, _diagnostic_divergence_velocity), src/embedded/operators.jl (divergence! with geometry),
#         src/operators/advection/cell_advection.jl (divergence!), src/two_phase/two_velocity/transport_geometry.jl (phase apertures)
# Checks:
#   D1  one velocity: DivergenceNorm = (sum_c |c^o| (D v)_c^2)^(1/2) with D = divh (no solid) / divopen (solid), independent cell sums
#       with random interior and boundary velocities (11 fixtures);
#   D2  two velocities: = the same with the phase-aperture flux a^l v^l + a^g v^g (interior and boundary faces); independent of the
#       densities (1000 vs 6.6) and of NaN at absent phases; a^l + a^g = a on every interior and boundary face;
#   D3  projection constraint, two velocities (CMDGFM and mimetic, uniform / cut, density ratio 1000, slip state): after one projected
#       step the norm is <= sqrt(|Omega^o|) tol/dt, while the density-weighted mixture divergence (the diagnostic before #101) is far
#       above that bound; the independent face sum equals the diagnostic;
#   D4  projection constraint, one velocity with a solid (periodic channel, sloped cut): after a projected step DivergenceNorm is at the
#       tolerance whereas the full-volume divergence that ignores the apertures is O(1) larger;
#   D5  uniform through-flow (u, v) = (0.7, -0.4) with a drop and NaN phases: norm 0 only if the boundary fluxes enter.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-verif-diagnostics/scripts/code_03_divergence_norm.jl

using Test, Random
isdefined(Main, :ForcesAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "forces_audit.jl"))
isdefined(Main, :VerifAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "verification_audit.jl"))
using TwoPhaseAMR
const VA = VerifAudit
const AMR = TwoPhaseAMR
const rel(a, b) = abs(a - b) / max(abs(b), 1.0e-300)

@testset "eq:verif:divergence-norm" begin
    # ---- D1, D2
    for fix in VA.fixtures()
        field = fix.field
        mesh = fix.mesh
        flow, state = VA.flow_state(fix; two = false)
        independent = VA.divergence_norm(field, flow.velocity; two = false)
        @test independent > 1.0e-3
        @test rel(diagnostic_value(DivergenceNorm(), state), independent) < 1.0e-12
        values = Float64[]
        for density in ((1000.0, 1.0), (6.6, 1.0))
            flow2, state2 = VA.flow_state(fix; two = true, density, rng = MersenneTwister(21))
            independent2 = VA.divergence_norm(field, flow2.velocity; two = true)
            @test independent2 > 1.0e-3
            @test rel(diagnostic_value(DivergenceNorm(), state2), independent2) < 1.0e-12
            push!(values, diagnostic_value(DivergenceNorm(), state2))
        end
        @test rel(values[1], values[2]) < 1.0e-13                # independent of the densities
        # a^l + a^g = a (interior and boundary faces)
        la, ga = AMR._current_phase_face_apertures!(field)
        a = fix.solid ? fix.geometry.aperture : ones(length(mesh.faces))
        @test maximum(abs.(la .+ ga .- a)) <= 1.0e-13
        ba = fix.solid ? fix.geometry.boundary_aperture : ones(length(mesh.boundary_faces))
        @test maximum(abs(sum(AMR._phase_boundary_face_apertures(field, i)) - ba[i]) for i in eachindex(mesh.boundary_faces)) <= 1.0e-13
        println(rpad(fix.name, 52), " one velocity ", round(independent, sigdigits = 8), "  two velocities ", round(values[1], sigdigits = 8))
    end

    # ---- D3: two-velocity projection constraint (as test/diagnostics.jl, with the old mass-weighted diagnostic as foil)
    dt = 1.0e-3
    tolerance = 1.0e-9
    for projection in (CMDGFMProjection(), MimeticProjection()), cut in (false, true)
        mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, false))
        geometry = cut ? GeometryField(mesh, (x, y) -> y - 0.1; lipschitz = 1) : nothing
        interface = InterfaceField(mesh, (x, y) -> y - 0.47 - 0.03cospi(2x); geometry, lipschitz = 1.02)
        liquid = (x, y, time, axis) -> axis == 1 ? 0.18 + 0.04cospi(2y) : 0.02sinpi(2x) * sinpi(y)
        gas = (x, y, time, axis) -> axis == 1 ? -0.08 + 0.03cospi(2y) : 0.02sinpi(2x) * sinpi(y)
        flow = FlowSimulation(
            mesh, TwoPhase(interface; density = (1000.0, 1.0), dynamic_viscosity = (0.0, 0.0));
            geometry, velocity_model = TwoVelocityModel(pressure_projection = projection, interface_transport = FrozenInterfaceFluxes()),
            velocity = (liquid, gas), boundary = FreeSlip(), diagnostics = (DivergenceNorm(),),
        )
        before = only(diagnostic_history(flow, :divergence_norm).values)
        stats = step!(flow.velocity, flow.pressure, flow.system, dt, flow.workspace; tolerance, max_cycles = 4000)
        @test stats.converged
        state = simulation_state(flow)
        after = diagnostic_value(DivergenceNorm(), state)
        independent = VA.divergence_norm(interface, flow.velocity; two = true)
        open_volume = isnothing(geometry) ? prod(mesh.extent) : integrate(geometry.open_fraction)
        # mass-weighted mixture divergence (the diagnostic before #101), computed independently
        ρl, ρg = 1000.0, 1.0
        mixture = FaceVelocity(mesh)
        for (i, face) in pairs(mesh.faces)
            geometry !== nothing && iszero(geometry.face_dual_volumes[i]) && continue
            C = VA.clamped_fraction(interface, face)
            wl, wg = ρl * C, ρg * (1 - C)
            mixture.interior[i] = ((wl == 0 ? 0.0 : wl * flow.velocity[1].interior[i]) + (wg == 0 ? 0.0 : wg * flow.velocity[2].interior[i])) / (wl + wg)
        end
        old = VA.divergence_norm(interface, mixture; two = false)
        println(rpad("projection $(nameof(typeof(projection))), cut=$cut", 40), " before ", round(before, sigdigits = 3), " after ", after, " bound ", sqrt(open_volume) * tolerance / dt, " mass-weighted ", round(old, sigdigits = 3))
        @test rel(after, independent) < 1.0e-8 || abs(after - independent) < 1.0e-11
        @test after <= sqrt(open_volume) * tolerance / dt + 1.0e-11
        @test before > 100 * after
        @test old > 100 * sqrt(open_volume) * tolerance / dt
    end

    # ---- D4: one-velocity projection with a solid
    for cut in (false, true)
        mesh = BlockTreeMesh(block_size = (8, 8), periodic = (true, false))
        geometry = cut ? GeometryField(mesh, (x, y) -> y - 0.1 - 0.05x; lipschitz = hypot(1, 0.05)) : nothing
        interface = InterfaceField(mesh, (x, y) -> y - 0.47 - 0.03cospi(2x); geometry, lipschitz = 1.02)
        velocity = (x, y, time, axis) -> axis == 1 ? 0.18 + 0.04cospi(2y) : 0.02sinpi(2x) * sinpi(y)
        flow = FlowSimulation(
            mesh, TwoPhase(interface; density = (1000.0, 1.0), dynamic_viscosity = (0.0, 0.0));
            geometry, velocity_model = OneVelocityModel(), velocity, boundary = FreeSlip(), diagnostics = (DivergenceNorm(),),
        )
        before = only(diagnostic_history(flow, :divergence_norm).values)
        stats = step!(flow.velocity, flow.pressure, flow.system, dt, flow.workspace; tolerance, max_cycles = 4000)
        @test stats.converged
        state = simulation_state(flow)
        after = diagnostic_value(DivergenceNorm(), state)
        independent = VA.divergence_norm(interface, flow.velocity; two = false)
        open_volume = isnothing(geometry) ? prod(mesh.extent) : integrate(geometry.open_fraction)
        # divergence that ignores the apertures: sum |f| v_f / |c| over the full cells, weighted by the open volume
        plain = CellField(mesh)
        divergence!(plain, flow.velocity)
        nxb, nyb = mesh.block_size
        h = mesh.halo
        ignoring = sqrt(
            sum(eachindex(mesh.blocks)) do b
                sum(abs2.(interior(plain, b)) .* (isnothing(geometry) ? 1.0 : interior(geometry.open_fraction, b))) * mesh.blocks[b].volume
            end
        )
        println(rpad("one-velocity projection, cut=$cut", 40), " before ", round(before, sigdigits = 3), " after ", after, " aperture-free divergence ", round(ignoring, sigdigits = 3))
        @test rel(after, independent) < 1.0e-8 || abs(after - independent) < 1.0e-11
        @test after <= sqrt(open_volume) * tolerance / dt + 1.0e-11
        cut && @test ignoring > 100 * after                   # the open divergence is the constraint that the projection enforces
    end

    # ---- D5: uniform through-flow
    mesh = BlockTreeMesh(block_size = (4, 4), periodic = (false, false))
    interface = InterfaceField(mesh, (x, y) -> x + 0.23y - 0.43)
    shared = (x, y, time, axis) -> axis == 1 ? 0.7 : -0.4
    flow = FlowSimulation(
        mesh, TwoPhase(interface; density = (1000.0, 1.0), dynamic_viscosity = (0.0, 0.0));
        velocity_model = TwoVelocityModel(), velocity = (shared, shared), diagnostics = (DivergenceNorm(),),
    )
    @test any(isnan, flow.velocity[1].interior)
    @test only(diagnostic_history(flow, :divergence_norm).values) <= 128eps()
    saved = copy(flow.velocity[1].boundary)
    flow.velocity[1].boundary[1] += 0.1
    @test diagnostic_value(DivergenceNorm(), simulation_state(flow)) > 1.0e-3          # boundary faces enter
    flow.velocity[1].boundary .= saved
end
