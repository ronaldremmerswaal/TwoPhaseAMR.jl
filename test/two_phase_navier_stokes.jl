using Test, TwoPhaseAMR
@testset "One Navier-Stokes API, transported density jump" begin
    for kind in 1:3
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 2,
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        f = InterfaceField(mesh, (x, y) -> x - 0.43)
        indicator = InterfaceRefinementIndicator(f; level = 1)
        q = PhaseScalar(f; liquid = 1.0, gas = 1.0)
        flow = FlowSimulation(
            mesh; interface = f, viscosity = (0.0, 0.0), density = (1000.0, 1.0),
            velocity = (0.1, -0.05), phase_scalars = (q,),
            refinement_indicators = kind == 3 ? (indicator,) : ()
        )
        volume = integrate(f)
        simulate!(flow, 0.02; dt = 0.002, rtol = 1.0e-11, atol = 1.0e-12)
        @test integrate(f) ≈ volume atol = 5.0e-13
        for (i, face) in pairs(mesh.faces)
            @test flow.velocity.interior[i] ≈ (face.axis == 1 ? 0.1 : -0.05) atol = 2.0e-10
        end
        for b in eachindex(mesh.blocks)
            @test interior(q.liquid, b) ≈ ones(size(interior(q.liquid, b))) atol = 2.0e-13
            @test interior(q.gas, b) ≈ ones(size(interior(q.gas, b))) atol = 2.0e-13
        end
        @test kind != 3 || nblocks(mesh) > 1
    end
end
@testset "Flat heavy interface remains hydrostatic" begin
    for kind in 1:3
        mesh = BlockTreeMesh(
            block_size = (8, 8), max_level = 2, periodic = (true, false),
            refine = (k, o, e) -> kind == 2 && (k.level == 0 || (k.level == 1 && k.i == 0 && k.j == 0))
        )
        f = InterfaceField(mesh, (x, y) -> y - 0.43)
        indicator = InterfaceRefinementIndicator(f; level = 1)
        flow = FlowSimulation(
            mesh; interface = f, viscosity = (0.0, 0.0), density = (1000.0, 1.0),
            gravity = (0.0, -1.0), refinement_indicators = kind == 3 ? (indicator,) : ()
        )
        mass = integrate(f)
        simulate!(flow, 0.02; dt = 0.002, rtol = 1.0e-11, atol = 1.0e-12)
        @test maximum(abs, flow.velocity.interior) < 1.0e-9
        @test integrate(f) ≈ mass atol = 2.0e-13
    end
end
