using TwoPhaseAMR

function taylor_green(; block_size=12, final_time=0.02,
                      viscosity=0.02, mixed=true)
    mesh = mixed ? BlockTreeMesh(
        block_size=(block_size, block_size), extent=(2pi, 2pi), max_level=2,
        refine=(key, origin, extent) ->
            key.level == 0 ||
            (key.level == 1 && key.i == 0 && key.j == 0),
    ) : BlockTreeMesh(
        block_size=(block_size, block_size), extent=(2pi, 2pi))
    exact = (x, y, time, axis) -> exp(-2viscosity * time) *
        (axis == 1 ? sin(x) * cos(y) : -cos(x) * sin(y))
    velocity = FaceVelocity(mesh, exact)
    pressure = CellField(mesh)
    solver = NavierStokesSolver(
        mesh; viscosity, boundary=FreeSlip(), coarse_size=16)
    workspace = NavierStokesWorkspace(solver)
    result = simulate!(
        velocity, pressure, solver, final_time, workspace;
        dt=0.02 / block_size, rtol=2e-10, max_cycles=80)

    squared_error = 0.0
    total_volume = 0.0
    for component in (solver.diffusion.x, solver.diffusion.y)
        for local_index in eachindex(component.local_to_global)
            global_index = component.local_to_global[local_index]
            face = mesh.faces[global_index]
            error = velocity.interior[global_index] -
                    exact(face.center[1], face.center[2], final_time, face.axis)
            squared_error += component.volumes[local_index] * error^2
            total_volume += component.volumes[local_index]
        end
    end
    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    maximum_divergence = maximum(
        maximum(abs, interior(divergence, block))
        for block in eachindex(mesh.blocks))
    return (result=result, velocity_l2=sqrt(squared_error / total_volume),
            maximum_divergence=maximum_divergence)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println(taylor_green())
end
