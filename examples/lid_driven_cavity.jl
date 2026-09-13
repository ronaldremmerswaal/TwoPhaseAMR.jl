using TwoPhaseAMR

function lid_driven_cavity(; block_size=32, final_time=0.2,
                           dt=0.002, reynolds=100.0)
    mesh = BlockTreeMesh(
        block_size=(block_size, block_size), periodic=(false, false))
    lid = NoSlip((x, y, time, axis) ->
        axis == 1 && y > 0.5 ? 1.0 : 0.0)
    solver = NavierStokesSolver(
        mesh; viscosity=inv(reynolds), boundary=lid, coarse_size=32)
    workspace = NavierStokesWorkspace(solver)
    velocity = FaceVelocity(mesh, (x, y, time, axis) -> 0.0)
    pressure = CellField(mesh)
    result = simulate!(
        velocity, pressure, solver, final_time, workspace;
        dt, rtol=1e-10, max_cycles=80)

    divergence = CellField(mesh)
    divergence!(divergence, velocity)
    maximum_divergence = maximum(
        maximum(abs, interior(divergence, block))
        for block in eachindex(mesh.blocks))
    return (result=result, maximum_divergence=maximum_divergence,
            velocity=velocity, pressure=pressure)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = lid_driven_cavity()
    println((time=result.result.time, steps=result.result.steps,
             pressure_cycles=result.result.pressure.cycles,
             maximum_divergence=result.maximum_divergence))
end
