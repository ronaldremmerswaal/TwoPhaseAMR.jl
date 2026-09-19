using Test, TwoPhaseAMR

function anisotropic_vof_error(n)
    mesh = BlockTreeMesh(
        base_mesh = (2n, n), block_size = (8, 8),
        periodic = (true, true),
    )
    centre = (0.5, 0.5)
    radius = 0.2
    velocity_components = (0.2, -0.1)
    final_time = 0.2
    levelset = (x, y) -> hypot(x - centre[1], y - centre[2]) - radius
    interface = InterfaceField(mesh, levelset; lipschitz = 1)
    velocity = FaceVelocity(
        mesh,
        (x, y, time, axis) -> velocity_components[axis],
    )
    workspace = WYAdvectionWorkspace(interface)
    time = 0.0
    step = 0
    while time < final_time
        dt = min(
            stable_timestep(velocity, workspace; cfl = 0.4),
            final_time - time,
        )
        advect!(
            interface, velocity, dt, workspace;
            sweep_order = iseven(step) ? (1, 2) : (2, 1),
        )
        time += dt
        step += 1
    end
    translated = (
        mod(centre[1] + velocity_components[1] * final_time, 1),
        mod(centre[2] + velocity_components[2] * final_time, 1),
    )
    reference = InterfaceField(
        mesh,
        (x, y) -> hypot(x - translated[1], y - translated[2]) - radius;
        lipschitz = 1,
    )
    error = 0.0
    for block in eachindex(mesh.blocks)
        error += mesh.blocks[block].volume * sum(
            abs.(interior(interface, block) .- interior(reference, block)),
        )
    end
    return error, abs(integrate(interface) - integrate(reference))
end

function anisotropic_flow_error(n, model)
    mesh = BlockTreeMesh(
        base_mesh = (2n, n), block_size = (8, 8),
        periodic = (true, true),
    )
    viscosity = 0.01
    final_time = 0.01
    exact = (x, y, time, axis) -> exp(-8pi^2 * viscosity * time) *
        (
        axis == 1 ? sinpi(2x) * cospi(2y) :
            -cospi(2x) * sinpi(2y)
    )
    initial = (x, y, time, axis) -> exact(x, y, 0, axis)
    interface = model == :single_phase ? nothing : InterfaceField(
            mesh, (x, y) -> hypot(x - 0.5, y - 0.5) - 0.23;
            lipschitz = 1,
        )
    velocity = model == :two_velocity ?
        (FaceVelocity(mesh, initial), FaceVelocity(mesh, initial)) :
        FaceVelocity(mesh, initial)
    material = model == :single_phase ? 1.0 : (1.0, 1.0)
    diffusion = model == :single_phase ? viscosity :
        (viscosity, viscosity)
    solver = NavierStokesSolver(
        mesh; density = material, dynamic_viscosity = diffusion,
        interface, boundary = FreeSlip(), coarse_size = 32,
    )
    pressure = CellField(mesh)
    workspace = NavierStokesWorkspace(solver; dt = final_time / n)
    result = simulate!(
        velocity, pressure, solver, final_time, workspace;
        dt = final_time / n, tolerance = 1.0e-9, max_cycles = 200,
    )
    carriers = velocity isa FaceVelocity ? (velocity,) : velocity
    errors = ntuple(length(carriers)) do phase
        carrier = carriers[phase]
        squared_error = total_weight = 0.0
        for (index, face) in pairs(mesh.faces)
            phase_weight = model == :two_velocity ?
                TwoPhaseAMR._phase_face_apertures(interface, index)[phase] : 1.0
            difference = carrier.interior[index] - exact(
                face.center[1], face.center[2], final_time, face.axis,
            )
            weight = face.area * phase_weight
            squared_error += weight * difference^2
            total_weight += weight
        end
        sqrt(squared_error / total_weight)
    end
    return maximum(errors), result.pressure.converged
end

@testset "anisotropic geometric VOF advection" begin
    results = anisotropic_vof_error.((16, 32, 64))
    errors = collect(first.(results))
    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
    @test all(diff(errors) .< 0)
    @test minimum(rates) > 1.8
    @test maximum(last.(results)) <=
        64 * 64^2 * eps(Float64)
end

@testset "anisotropic Navier--Stokes model refinement" begin
    for model in (:single_phase, :two_phase, :two_velocity)
        results = anisotropic_flow_error.((8, 16), (model,))
        errors = collect(first.(results))
        observed_order = log2(errors[1] / errors[2])
        @test errors[2] < errors[1]
        @test observed_order > (model == :single_phase ? 1.8 : 1.35)
        @test all(last, results)
    end
end
