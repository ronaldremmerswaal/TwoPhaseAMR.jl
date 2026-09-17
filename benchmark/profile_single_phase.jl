using Profile
using TwoPhaseAMR

function mixed_mesh(block_size = 16)
    return BlockTreeMesh(
        block_size = (block_size, block_size), max_level = 2,
        refine = (key, origin, extent) ->
        key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0)
    )
end

function build_flow(mesh)
    velocity = (x, y, time, axis) -> axis == 1 ?
        sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y)
    return FlowSimulation(
        mesh; density = 1.0, dynamic_viscosity = 0.01, velocity,
        boundary = FreeSlip(), coarse_size = 32, dt = 1.0e-4
    )
end

function repeated_setup(mesh, repetitions)
    flow = nothing
    for _ in 1:repetitions
        flow = build_flow(mesh)
    end
    return flow
end

function advance_profile!(flow, duration, dt)
    return simulate!(
        flow, flow.time + duration; dt, rtol = 1.0e-8, max_cycles = 80
    )
end

mesh = mixed_mesh()
repeated_setup(mesh, 1) # compile before collecting samples
Profile.clear()
@profile repeated_setup(mesh, 20)
println("\nFlat profile: mixed-level FlowSimulation setup")
Profile.print(format = :flat, sortedby = :count, mincount = 20)

flow = build_flow(BlockTreeMesh(block_size = (32, 32)))
advance_profile!(flow, 1.0e-4, 1.0e-4) # compile and build shifted hierarchy
Profile.clear()
@profile advance_profile!(flow, 0.02, 1.0e-4)
println("\nFlat profile: 200 fixed-mesh time steps")
Profile.print(format = :flat, sortedby = :count, mincount = 20)
