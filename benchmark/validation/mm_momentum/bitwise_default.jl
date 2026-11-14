# Default path unchanged: run short two-velocity simulations (MimeticProjection
# and CMDGFMProjection, surface tension, gravity) and write the bits of the
# final velocities and pressures to ARGS[1]. Run once with the study changes
# stashed and once with them, compare the files.
using LinearAlgebra
BLAS.set_num_threads(1)
using TwoPhaseAMR
@assert startswith(Base.active_project(), "/home/ronald/Documents/wt-mm-momentum")
bits = UInt64[]
for projection in (MimeticProjection(), CMDGFMProjection()), (sigma, gravity, U) in ((1.0, (0.0, 0.0), (0.3, 0.1)), (0.0, (0.0, -1.0), (0.0, 0.0)))
    mesh = BlockTreeMesh(block_size = (16, 16), periodic = (true, gravity[2] == 0))
    interface = InterfaceField(mesh, (x, y) -> hypot(x - 0.5, y - 0.45) - 0.2 - 0.02cospi(6atan(y - 0.45, x - 0.5) / pi); lipschitz = 1)
    physics = TwoPhase(interface; density = (1.0, 0.1), dynamic_viscosity = (0.01, 0.001),
        surface_tension = SurfaceTension(sigma; method = CGFM()), gravity)
    v = (x, y, t, axis) -> U[axis] + 0.1 * (axis == 1 ? sinpi(2x) * cospi(2y) : -cospi(2x) * sinpi(2y))
    flow = FlowSimulation(mesh, physics; velocity_model = TwoVelocityModel(pressure_projection = projection),
        boundary = FreeSlip(), velocity = (v, v))
    simulate!(flow, 0.05; tolerance = 1.0e-10, max_cycles = 1000)
    for phase in 1:2
        append!(bits, reinterpret(UInt64, flow.velocity[phase].interior))
    end
    for block in flow.pressure.blocks
        append!(bits, reinterpret(UInt64, vec(block)))
    end
    println(typeof(projection), " sigma=", sigma, " steps=", flow.steps)
end
write(ARGS[1], bits)
println("wrote ", length(bits), " words")
