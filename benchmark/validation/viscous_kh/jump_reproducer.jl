# Minimal reproducer: the two-velocity model creates a tangential velocity jump at the
# interface from a *continuous*, well-resolved velocity field.
#   julia --project=. jump_reproducer.jl
#
# Uniform periodic-in-x mesh on (0,1) × (−1/2,1/2), free-slip walls, equal densities and
# viscosities, no surface tension. The interface y = a cos(2πx) carries the resolved
# shear layer u = (ΔU/2) tanh(y/δ), v = 0, given to BOTH phases (so u_g − u_l = 0 at
# t = 0 on every face). The exact solution keeps the velocity continuous. The script
# prints the largest phase-velocity difference max|u_g − u_l| over the faces where both
# phases are present, after a number of steps, for δ/h = 1.6, 3.2, 6.4.
using TwoPhaseAMR
using Printf

function jump_history(n; δ = 0.05, a = 0.01, steps = (1, 10, 40, 160), cfl = 0.4)
    mesh = BlockTreeMesh(;
        base_mesh = (n, n), block_size = (8, 8), origin = (0.0, -0.5), extent = (1.0, 1.0),
        periodic = (true, false),
    )
    interface = InterfaceField(mesh, (x, y) -> y - a * cospi(2x); lipschitz = 1)
    physics = TwoPhase(interface; density = (1.0, 1.0), dynamic_viscosity = (1.0e-3, 1.0e-3))
    u0 = (x, y, t, axis) -> axis == 1 ? 0.5 * tanh(y / δ) : 0.0
    flow = FlowSimulation(mesh, physics; velocity_model = TwoVelocityModel(), velocity = u0, boundary = FreeSlip())
    dt = stable_timestep(flow; cfl)
    done = 0
    out = Float64[]
    for target in steps
        while done < target
            step!(flow, dt)
            done += 1
        end
        ul, ug = flow.velocity
        jump = 0.0
        for (i, face) in pairs(mesh.faces)
            face.axis == 1 || continue
            l, g = ul.interior[i], ug.interior[i]
            isfinite(l) && isfinite(g) && (jump = max(jump, abs(g - l)))
        end
        push!(out, jump)
    end
    return steps, out, dt
end

for n in (32, 64, 128)
    steps, jumps, dt = jump_history(n)
    @printf("δ/h = %.1f (dt = %.2e): ", 0.05n, dt)
    for (s, j) in zip(steps, jumps)
        @printf(" step %3d: max|u_g − u_l| = %.3f;", s, j)
    end
    println()
end
